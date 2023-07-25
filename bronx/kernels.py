from functools import cache, partial
from pyro.contrib.gp.kernels.kernel import Kernel
from pyro.contrib.gp.util import conditional as _conditional

class GraphLinearDiffusion(Kernel):
    def __init__(self, input_dim, variance=None, active_dims=None):
        super().__init__(input_dim, active_dims)

    @cache
    def graph_exp(self, graph):
        a = g.adj().to_dense()
        d = a.sum(-1, keepdims=True)
        a = a / d
        a = a - torch.eye(a.shape[0], dtype=a.dtype)
        a = torch.linalg.matrix_exp(a)
        return a

    def forward(self, X, Z=None, graph=None):
        if Z is None:
            Z = X
        variance = self.graph_exp(graph)
        return variance[X, Z]

def conditional(graph, X_new, X, kernel, **kwargs):
    kernel = partial(kernel, graph=graph)
    return _conditional(X_new, X, kernel, **kwargs)






        
        


