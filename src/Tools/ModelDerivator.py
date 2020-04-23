import jax.numpy as np


class ModelDerivator:

    def set_pack(self, pack):
        raise NotImplementedError

    def fu(self, u: np.array, x: np.array, eps: float = 1e-3) -> np.array:
        """
            derivative respect to u
        """
        raise NotImplementedError

    def fx(self, u: np.array, x: np.array, eps: float = 1e-3) -> np.array:
        """
            derivative respect to x
        """
        raise NotImplementedError
