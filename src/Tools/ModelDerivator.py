import jax.numpy as np


class ModelDerivator:

    def set_pack(self, pack):
        raise NotImplementedError

    def fu(self, x: np.array, u: np.array, eps: float = 1e-3) -> np.array:
        """
            derivative respect to u
        """
        raise NotImplementedError

    def fx(self, x: np.array, u: np.array, eps: float = 1e-3) -> np.array:
        """
            derivative respect to x
        """
        raise NotImplementedError
