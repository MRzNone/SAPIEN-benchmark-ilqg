import jax.numpy as np
from jax import jit, jacfwd, jacrev, grad, jacobian
import jax
import time
from joblib import Parallel, delayed
import numpy as onp
from tqdm import tqdm, trange
from joblib import Parallel, delayed, wrap_non_picklable_objects
# from tqdm.notebook import tqdm, trange

from Tools import SimWorker, misc, ModelDerivator


class ILQG_Human:
    def __init__(self, final_cost, running_cost, model, u_range, horizon, per_iter, model_der: ModelDerivator = None,
                 model_sim: SimWorker = None, DEBUG=False, d0=2.0, a0=0.9, max_loops=10):
        """
            final_cost:     v(x)    ->  cost, float
            running_cost:   l(x, u) ->  cost, float
            model:          f(x, u) ->  new state, [n_x]
        """
        self.mu_min = 1e-6
        self.max_loops = max_loops
        self.DEBUG = DEBUG
        self.packs = None
        self.d0 = d0
        self.a0 = a0
        self.num_x = None
        self.num_u = None

        if model_sim is None:
            self.f = model
        else:
            self.model_sim = model_sim
            self.f = lambda x, u: self.model_sim.sim(u)

        self.v = final_cost
        self.l = running_cost

        self.u_range = u_range
        self.horizon = horizon
        self.per_iter = per_iter

        # specify derivatives
        self.l_x = grad(self.l, 0)
        self.l_u = grad(self.l, 1)
        self.l_xx = jacfwd(self.l_x, 0)
        self.l_uu = jacfwd(self.l_u, 1)
        self.l_ux = jacrev(self.l_u, 0)

        self.v_x = jacrev(self.v, 0)
        self.v_xx = jacfwd(self.v_x, 0)

        if model_der is None:
            self.f_x = jacrev(self.f, 0)
            self.f_u = jacfwd(self.f, 1)

            (self.f, self.f_u, self.f_x,) = [jit(e) for e in [self.f, self.f_u, self.f_x, ]]
        else:
            # using provided function for step
            self.f_x = lambda x, u: model_der.fx(x, u)
            self.f_u = lambda x, u: model_der.fu(x, u)
            self.model_der = model_der

        # speed up

        (self.l, self.l_u, self.l_uu, self.l_ux, self.l_x, self.l_xx,
         self.v, self.v_x, self.v_xx, self.f_u, self.f_x) = \
            [jit(e, backend='gpu') for e in
             [self.l, self.l_u, self.l_uu, self.l_ux, self.l_x, self.l_xx,
              self.v, self.v_x, self.v_xx, self.f_u, self.f_x]]

        self.vmap_l = jit(jax.vmap(self.l, in_axes=(0, 0)), backend='gpu')

    @jax.partial(jax.jit, static_argnums=(0,), backend='gpu')
    def cal_Qs(self, lx, lu, lxx, luu, lux, fx, fu, vx, vxx, mu):
        adjusted_vxx = vxx + np.eye(self.num_x) * mu

        q_x = lx + fx.T @ vx
        q_u = lu + fu.T @ vx
        q_xx = lxx + fx.T @ adjusted_vxx @ fx
        q_uu = luu + fu.T @ adjusted_vxx @ fu
        q_ux = lux + fu.T @ adjusted_vxx @ fx

        return q_x, q_u, q_xx, q_uu, q_ux

    @jax.partial(jax.jit, static_argnums=(0,), backend='gpu')
    def cal_Ks(self, q_x, q_u, q_xx, q_uu, q_ux):
        # cal Ks
        inv_quu = np.linalg.inv(q_uu)

        k = - inv_quu @ q_u
        kk = - inv_quu @ q_ux

        # cal Vs
        # new_v = q_u @ k / 2
        new_vx = q_x + kk.T @ q_uu @ k + kk.T @ q_u + q_ux.T @ k
        new_vxx = q_xx + kk.T @ q_uu @ kk + kk.T @ q_ux + q_ux.T @ kk

        return k, kk, new_vx, new_vxx

    def increase_val(self, val, factor, scale):
        factor = max(scale, scale * factor)
        val = max(self.mu_min, val * factor)

        return val, factor

    def decrease_val(self, val, factor, scale):
        factor = min(1 / scale, factor / scale)

        dm = val * factor
        val = val * factor if dm > self.mu_min else 0

        return val, factor

    def backward(self, last_x, derivs, mu, d, root_pos, jaco):
        """
            Calculate all the necessary derivatives, and compute the Ks
        """
        orig_v_x = self.v_x(last_x, root_pos, jaco)
        orig_v_xx = self.v_xx(last_x, root_pos, jaco)
        accept = False
        valid_pair = None

        for _ in range(self.max_loops):
            k_seq = [None] * self.horizon
            kk_seq = [None] * self.horizon
            v_x = orig_v_x
            v_xx = orig_v_xx
            safe = True
            j1, j2 = 0, 0

            for i in range(self.horizon - 2, -1, -1):
                # get all grads
                lx, lu, lxx, luu, lux, fx, fu = derivs[i]

                # cal Qs
                q_x, q_u, q_xx, q_uu, q_ux = self.cal_Qs(lx, lu, lxx, luu, lux, fx, fu, v_x, v_xx, mu)

                if not misc.is_PD(q_uu):
                    mu, d = self.increase_val(mu, d, self.d0)
                    safe = False
                    break

                # cal Ks
                k, kk, new_vx, new_vxx = self.cal_Ks(q_x, q_u, q_xx, q_uu, q_ux)

                # for m, name in zip([q_x, q_u, q_xx, q_uu, q_ux, k, kk, new_vx, new_vxx],
                #                    ["q_x", "q_u", "q_xx", "q_uu", "q_ux", "k", "kk", "new_vx", "new_vxx"]):
                #     if misc.check_val(m):
                #         print(name)
                #         raise RuntimeError("fasdds")

                if k is None or kk is None:
                    mu, d = self.increase_val(mu, d, self.d0)
                    safe = False
                    break

                # record
                k_seq[i] = k
                kk_seq[i] = kk

                # accumulate j for forward pass
                j1 += k.T @ q_u
                j2 += k.T @ q_uu @ k

            if safe:
                accept = True
                valid_pair = (k_seq, kk_seq, mu, d, j1, j2)
                mu, d = self.decrease_val(mu, d, self.d0)

        if not accept:
            raise RuntimeError("Backward failed")

        return valid_pair

    def cal_traj(self, x_seq, u_seq, k_seq, kk_seq, a):
        new_x_seq = [onp.array(x_seq[0])]
        new_u_seq = []

        if self.packs is not None:
            packs = [self.packs[0]]
            self.model_sim.set(self.packs[0])
        else:
            packs = None

        u_seq = onp.array(u_seq[:-1])
        k_seq = onp.array(k_seq[:-1])

        uak_seq = onp.array(u_seq + a * k_seq)
        x_seq = onp.array(x_seq)
        kk_seq = onp.array(kk_seq)

        # for i in trange(self.horizon - 1, desc='forward', leave=False):
        for i in range(self.horizon - 1):
            x = new_x_seq[i]

            new_u = uak_seq[i] + kk_seq[i] @ (x - x_seq[i])
            new_u = onp.clip(new_u, self.u_range[0], self.u_range[1])
            new_x = self.f(x, new_u)

            if self.packs is not None:
                pack = self.model_sim.get_pack()
                packs.append(pack)

            new_u_seq.append(new_u)
            new_x_seq.append(new_x)

        new_u_seq.append(new_u_seq[-1])  # filling
        if self.packs is not None:
            packs.append(packs[-1])

        return new_x_seq, new_u_seq, packs

    # @jax.partial(jax.jit, static_argnums=(0,), backend='gpu')
    def cost(self, x_seq, u_seq, root_pos, jaco):
        # total_cost = 0

        # for x, u in zip(x_seq[:-1], u_seq[:-1]):
        #     total_cost += self.l(x, u)

        x_batch = np.array(x_seq[:-1])
        u_batch = np.array(u_seq[:-1])

        total_cost = np.sum(self.vmap_l(x_batch, u_batch))

        total_cost += self.v(x_seq[-1], root_pos, jaco)

        return total_cost

    def forward(self, x_seq, u_seq, k_seq, kk_seq, j1, j2, last_cost, root_pos, jaco):
        accept = False
        loop_num = 0
        a = 1

        min_pair = None
        min_cost = None
        while not accept and loop_num < self.max_loops:
            loop_num += 1

            new_x_seq, new_u_seq, packs = self.cal_traj(x_seq, u_seq, k_seq, kk_seq, a)

            new_cost = self.cost(new_x_seq, new_u_seq, root_pos, jaco)

            dj = a * j1 + a ** 2 / 2.0 * j2

            z = (last_cost - new_cost) / dj

            # check if safe
            a = a * self.a0
            if z < 0:
                accept = True
            elif min_cost is None or min_cost > new_cost:
                min_cost = new_cost
                min_pair = (False, new_x_seq, new_u_seq, packs, new_cost)

        if not accept:
            print("Forward failed")
            return min_pair

        return True, new_x_seq, new_u_seq, packs, new_cost

    # @jax.partial(jax.jit, static_argnums=(0,), backend='gpu')
    # def compute_der(self, x, u):
    #     lx = self.l_x(x, u)
    #     lu = self.l_u(x, u)
    #     lxx = self.l_xx(x, u)
    #     luu = self.l_uu(x, u)
    #     lux = self.l_ux(x, u)
    #     fx = self.f_x(x, u)
    #     fu = self.f_u(x, u)
    #
    #     return lx, lu, lxx, luu, lux, fx, fu
    #
    # @jax.partial(jax.jit, static_argnums=(0,), backend='gpu')
    # def compute_derivatives(self, x_seq, u_seq):
    #     derivs = []
    #     for x, u, pack in zip(x_seq, u_seq, self.packs):
    #         self.model_der.set_pack(pack)
    #         derivs.append(self.compute_der(x, u))
    #     return derivs

    @jax.partial(jax.jit, static_argnums=(0,), backend='gpu')
    def compute_der_math(self, x, u):
        lx = self.l_x(x, u)
        lu = self.l_u(x, u)
        lxx = self.l_xx(x, u)
        luu = self.l_uu(x, u)
        lux = self.l_ux(x, u)
        fx = self.f_x(x, u)
        # fu = self.f_u(x, u)

        return lx, lu, lxx, luu, lux, fx

    @jax.partial(jax.jit, static_argnums=(0,), backend='gpu')
    def compute_derivatives_math(self, x_seq, u_seq):
        derivs = []
        for x, u in zip(x_seq, u_seq):
            derivs.append(self.compute_der_math(x, u))
        return derivs

    def compute_derivatives_sim(self, x_seq, u_seq):
        derivs = []
        for x, u, pack in zip(x_seq, u_seq, self.packs):
            self.model_der.set_pack(pack)
            fu = self.model_der.fu(x, u)

            derivs.append(fu)
        return derivs


    def compute_derivatives(self, x_seq, u_seq):
        deriv_math = self.compute_derivatives_math(x_seq, u_seq)
        deriv_num = self.compute_derivatives_sim(x_seq, u_seq)

        derivs = []
        for i in range(len(deriv_num)):
            derivs.append((*deriv_math[i], deriv_num[i]))

        return derivs

    def zero_traj(self, first_x, first_pack, root_pos, jaco):
        x_seq = [first_x]
        u_seq = []
        packs = [first_pack]

        self.model_sim.set(first_pack)

        for i in range(self.horizon - 1):
            x = x_seq[i]
            u = [0.0] * self.num_u
            u_seq.append(np.array(u))

            new_x = self.model_sim.sim(u)
            x_seq.append(np.array(new_x))
            packs.append(self.model_sim.get_pack())

        u_seq.append(u_seq[-1])

        cost = self.cost(x_seq, u_seq, root_pos, jaco)

        return x_seq, u_seq, packs, cost

    def complete_last_traj(self, x_seq, u_seq, packs, root_pos, jaco):
        x_seq = x_seq[1:]
        u_seq = u_seq[1:]
        packs = packs[1:]

        self.model_sim.set(packs[-1])
        new_x = self.model_sim.sim(u_seq[-1])
        x_seq.append(new_x)
        packs.append(self.model_sim.get_pack())
        u_seq.append(u_seq[-1])

        cost = self.cost(x_seq, u_seq, root_pos, jaco)

        return x_seq, u_seq, packs, cost

    def predict(self, x_seq, u_seq, packs, last_cost, root_pos, jaco):
        self.packs = packs

        d = 1
        mu = 1

        if self.num_x is None:
            self.num_x = len(x_seq[0])
        if self.num_u is None:
            self.num_u = len(u_seq[0])

        # for _ in trange(self.per_iter, desc='ILQR', leave=False):
        for _ in range(self.per_iter):
            # compute derivatives
            derivs = self.compute_derivatives(x_seq, u_seq)

            # compute k
            k_seq, kk_seq, mu, d, j1, j2 = self.backward(x_seq[-1], derivs, mu, d, root_pos, jaco)

            if None in k_seq[:-1] or None in kk_seq[:-1]:
                print("Invalid backward")
                print("Using last trajectory")
                return self.complete_last_traj(x_seq, u_seq, packs, root_pos, jaco)

            # compute u
            ret, new_x_seq, new_u_seq, new_packs, new_cost = self.forward(x_seq, u_seq, k_seq, kk_seq, j1, j2,
                                                                          last_cost, root_pos, jaco)

            if new_cost > last_cost:
                print("Using last trajectory")
                return self.complete_last_traj(x_seq, u_seq, packs, root_pos, jaco)
            else:
                x_seq, u_seq, self.packs, cost = new_x_seq, new_u_seq, new_packs, new_cost

        return x_seq, u_seq, self.packs, cost
