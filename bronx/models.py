from functools import partial
import math
import torch
import dgl
import pyro
from pyro import poutine
from pyro.contrib.gp.models.vsgp import VariationalSparseGP as VSGP
# from .kernels import conditional

class GraphVariationalSparseGaussianProcess(VSGP):
    def __init__(self, graph, X, y, kernel, Xu, likelihood, latent_shape=None):
        kernel = partial(kernel, graph=graph)
        super().__init__(
            X, y, kernel, Xu.float(), likelihood, latent_shape=latent_shape,
        )
        del self.Xu
        self.register_buffer("Xu", Xu)


# class GraphVariationalSparseGaussianProcess(Parametrized):
#     def __init__(self, kernel, likelihood, jitter=1e-6):
#         super().__init__()
#         self.kernel = kernel
#         self.likelihood = likelihood
#         self.jitter = jitter