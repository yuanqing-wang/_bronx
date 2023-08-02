from functools import lru_cache, partial
from typing import Optional
import torch
from torch import Tensor
from gpytorch.kernels.kernel import Kernel
import linear_operator
from dgl import DGLGraph

class CombinedGraphDiffusion(Kernel):
    def __init__(self, base_kernel: Kernel):
        super().__init__()
        self.base_kernel = base_kernel

    @lru_cache(maxsize=1)
    def graph_exp(self, graph):
        # a = graph.adj().to_dense()
        a = torch.zeros(
            graph.number_of_nodes(),
            graph.number_of_nodes(),
            dtype=torch.float32,
            device=graph.device,
        )
        src, dst = graph.edges()
        a[src, dst] = 1
        d = a.sum(-1, keepdims=True).clamp(min=1)
        a = a / d
        # a = a - torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
        a = torch.linalg.matrix_exp(a)
        return a

    def forward(
        self,
        x1: Tensor,
        x2: Optional[Tensor] = None,
        diagonal: Optional[bool] = False,
        last_dim_is_batch: Optional[bool] = False,
        graph: Optional[DGLGraph] = None,
        ix1: Optional[Tensor] = None,
        ix2: Optional[Tensor] = None,
        **params,
    ):
        assert graph is not None, "Must provide a graph"
        assert ix1 is not None, "Must provide ix1"
        if ix2 is None:
            ix2 = ix1

        variance = self.base_kernel(
            x1,
            x2,
            diagonal=diagonal,
            last_dim_is_batch=last_dim_is_batch,
            **params,
        )
     
        a = self.graph_exp(graph)
        variance = a @ variance @ a.transpose(-1, -2)
        
        if diagonal:
            variance = variance[ix1.long()]
        else:
            variance = variance[ix1.long(), :][:, ix2.long()]
        
        return variance
