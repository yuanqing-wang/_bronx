from functools import partial
import math
import torch
import dgl
import pyro
from pyro import poutine
from pyro.contrib.gp.models.vsgp import VariationalSparseGP as VSGP
# from .kernels import conditional

class GraphVariationalSparseGaussianProcess(VSGP):
    def __init__(self, graph, X, y, kernel, Xu, likelihood):
        kernel = partial(kernel, graph=graph)
        super().__init__(X, y, kernel, Xu, likelihood)
        self.Xu = Xu

