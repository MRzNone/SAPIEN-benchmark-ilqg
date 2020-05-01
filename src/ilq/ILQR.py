import jax.numpy as np
from jax import jit, jacfwd, jacrev, grad, jacobian
import jax
import time
from joblib import Parallel, delayed
import numpy as onp
from tqdm import tqdm, trange
# from tqdm.notebook import tqdm, trange

from Tools import SimWorker, misc, ModelDerivator


class ILQR:
    def __init__(self, final_cost, running_cost, model, u_range, horizon, per_iter, model_der: ModelDerivator = None,
                 model_sim=None, DEBUG=False):
        """
            final_cost:     v(x)    ->  cost, float
            running_cost:   l(x, u) ->  cost, float
            model:          f(x, u) ->  new state, [n_x]
        """
        self.DEBUG = DEBUG
        self.packs = None
        self.model_sim = None

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

        self.v_x = jacrev(self.v)
        self.v_xx = jacfwd(self.v_x)

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
            [jit(e) for e in [self.l, self.l_u, self.l_uu, self.l_ux, self.l_x, self.l_xx,
                              self.v, self.v_x, self.v_xx, self.f_u, self.f_x]]

    @jax.partial(jax.jit, static_argnums=(0,))
    def cal_Ks(self, lx, lu, lxx, luu, lux, fx, fu, vx, vxx):
        q_x = lx + fx.T @ vx
        q_u = lu + fu.T @ vx
        q_xx = lxx + fx.T @ vxx @ fx
        q_uu = luu + fu.T @ vxx @ fu
        q_ux = lux + fu.T @ vxx @ fx

        # cal Ks
        inv_quu = np.linalg.inv(q_uu)

        k = - inv_quu @ q_u
        kk = - inv_quu @ q_ux

        # cal Vs
        # new_v = q_u @ k / 2
        new_vx = q_x + q_u @ kk
        new_vxx = q_xx + q_ux.T @ kk

        return k, kk, new_vx, new_vxx

    def backward(self, last_x, derivs):
        """
            Calculate all the necessary derivatives, and compute the Ks
        """
        v_x = self.v_x(last_x)
        v_xx = self.v_xx(last_x)

        k_seq = [None] * self.horizon
        kk_seq = [None] * self.horizon

        for i in range(self.horizon - 2, -1, -1):

            # get all grads
            lx, lu, lxx, luu, lux, fx, fu = derivs[i]

            # cal Qs
            k, kk, v_x, v_xx = self.cal_Ks(lx, lu, lxx, luu, lux, fx, fu, v_x, v_xx)

            # record
            k_seq[i] = k
            kk_seq[i] = kk

        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        new_x_seq = [None] * self.horizon
        new_u_seq = [None] * self.horizon

        new_x_seq[0] = x_seq[0]

        if self.model_sim is not None and self.packs is not None:
            packs = [self.packs[0]]
        else:
            packs = None

        # for i in trange(self.horizon - 1, desc='forward', leave=False):
        for i in range(self.horizon - 1):
            x = new_x_seq[i]

            new_u = u_seq[i] + k_seq[i] + kk_seq[i] @ (x - x_seq[i])
            new_u = np.clip(new_u, self.u_range[0], self.u_range[1])
            new_x = self.f(x, new_u)

            if self.packs is not None:
                pack = self.model_sim.get_pack()
                packs.append(pack)

            new_u_seq[i] = new_u
            new_x_seq[i + 1] = new_x

        new_u_seq[-1] = new_u_seq[-2]  # filling
        if self.packs is not None:
            packs[-1] = packs[-2]

        return new_x_seq, new_u_seq, packs

    @jax.partial(jax.jit, static_argnums=(0,))
    def compute_der(self, x, u):
        # x = xu[0]
        # u = xu[1]
        lx = self.l_x(x, u)
        lu = self.l_u(x, u)
        lxx = self.l_xx(x, u)
        luu = self.l_uu(x, u)
        lux = self.l_ux(x, u)
        fx = self.f_x(x, u)

        fu = self.f_u(x, u)

        return lx, lu, lxx, luu, lux, fx, fu

    @jax.partial(jax.jit, static_argnums=(0,))
    def compute_derivatives(self, x_seq, u_seq):
        derivs = []
        for x, u, pack in zip(x_seq, u_seq, self.packs):
            if self.model_der is not None:
                self.model_der.set_pack(pack)
            derivs.append(self.compute_der(x, u))
        return derivs

    def predict(self, x_seq, u_seq, packs):
        self.packs = packs

        # for _ in trange(self.per_iter, desc='ILQR', leave=False):
        for _ in range(self.per_iter):
            # compute derivatives
            derivs = self.compute_derivatives(x_seq, u_seq)

            # compute k
            k_seq, kk_seq = self.backward(x_seq[-1], derivs)

            if self.model_sim is not None:
                self.model_sim.set(self.packs[0])

            # compute u
            x_seq, u_seq, self.packs = self.forward(x_seq, u_seq, k_seq, kk_seq)

        return x_seq, u_seq, self.packs
