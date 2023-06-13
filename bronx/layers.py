import math
import torch
import pyro
from pyro import poutine
import dgl
# dgl.use_libxsmm(False)
from dgl.nn import GraphConv
from dgl import function as fn
from dgl.nn.functional import edge_softmax

@torch.jit.script
def approximate_matrix_exp(a, k:int=6):
    result = a
    for i in range(k-1):
        a = a @ a
        result = result + a / math.factorial(i)
    return result

class LinearDiffusion(torch.nn.Module):
    def __init__(self, gamma=0.0, dropout=0.0):
        super().__init__()
        self.gamma = gamma
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, g, h, e=None):
        a = g.adj().to_dense()
        if e.dim() == 2:
            a = a.unsqueeze(-3).repeat_interleave(e.shape[-2], dim=-3)
        src, dst = g.edges()
        a[..., src, dst] = e
        src = dst = torch.arange(g.number_of_nodes())
        a[..., src, dst] = self.gamma
        a = a / a.sum(-1, keepdims=True)
        # a = torch.linalg.matrix_exp(a)
        a = approximate_matrix_exp(a)
        a = self.dropout(a)
        h = a @ h
        return h

class BronxLayer(pyro.nn.PyroModule):
    def __init__(
            self, 
            in_features, out_features, activation=torch.nn.SiLU(), 
            dropout=0.0, idx=0, num_heads=4, gamma=0.0, edge_drop=0.0,
        ):
        super().__init__()
        self.fc_mu = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(in_features, out_features, bias=False)
        self.activation = activation
        self.idx = idx
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_diffusion = LinearDiffusion(
            dropout=edge_drop, gamma=gamma,
        )

    def guide(self, g, h):
        with pyro.plate(f"nodes{self.idx}", g.number_of_nodes(), device=g.device):
            with pyro.poutine.scale(None, float(g.ndata["train_mask"].sum() / g.number_of_nodes())):

                k = pyro.sample(
                    f"k{self.idx}",
                    pyro.distributions.Normal(
                        self.fc_mu(h),
                        self.fc_log_sigma(h).exp(),
                    ).to_event(1),
                )

        if k.dim() == 3:
            parallel = True
        else:
            parallel = False

        if parallel:
            k = k.swapaxes(0, 1)

        g.ndata["k"] = k
        g.apply_edges(fn.u_dot_v("k", "k", "e"))
        # e = edge_softmax(g, g.edata["e"]).squeeze(-1)
        e = g.edata["e"].exp().squeeze(-1)
        if parallel:
            e = e.swapaxes(0, 1)
        return e

    def mp(self, g, h, e):
        h = self.linear_diffusion(g, h, e)
        h = self.dropout(h)
        return h

    def forward(self, g, h):
        with pyro.plate(f"nodes{self.idx}", g.number_of_nodes(), device=g.device):
            with pyro.poutine.scale(None, float(g.ndata["train_mask"].sum() / g.number_of_nodes())):
                k = pyro.sample(
                    f"k{self.idx}",
                    pyro.distributions.Normal(
                        torch.zeros(g.number_of_nodes(), self.out_features, device=g.device),
                        torch.ones(g.number_of_nodes(), self.out_features, device=g.device),
                    ).to_event(1),
                )
        if k.dim() == 3:
            parallel = True
        else:
            parallel = False

        if parallel:
            k = k.swapaxes(0, 1)

        g.ndata["k"] = k
        g.apply_edges(fn.u_dot_v("k", "k", "e"))
        # e = edge_softmax(g, g.edata["e"]).squeeze(-1)
        e = g.edata["e"].exp().squeeze(-1)
        if parallel:
            e = e.swapaxes(0, 1)
        h = self.mp(g, h, e)
        return h

