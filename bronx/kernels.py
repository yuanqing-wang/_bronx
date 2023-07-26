from functools import lru_cache, partial
import torch
from pyro.contrib.gp.kernels.kernel import Kernel
# from pyro.contrib.gp.util import conditional as _conditional



class GraphLinearDiffusion(Kernel):
    def __init__(self, input_dim, variance=None, active_dims=None):
        super().__init__(input_dim, active_dims)

    @lru_cache(maxsize=1)
    def graph_exp(self, graph):
        # a = graph.adj().to_dense()
        a = torch.zeros(
            graph.number_of_nodes(), graph.number_of_nodes(),
            dtype=torch.float32, device=graph.device,
        )
        src, dst = graph.edges()
        a[src, dst] = 1
        d = a.sum(-1, keepdims=True)
        a = a / d
        a = a - torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
        a = torch.linalg.matrix_exp(a)
        return a

    @lru_cache(maxsize=8)
    def forward(self, X, Z=None, graph=None, diag=False):
        if Z is None:
            Z = X
        variance = self.graph_exp(graph)
        variance = variance[X.long(), :][:, Z.long()]
        if diag:
            variance = variance.diag()
        return variance

# def conditional(graph, X_new, X, kernel, **kwargs):
#     kernel = partial(kernel, graph=graph)
#     return _conditional(X_new, X, kernel, **kwargs)






        
        


