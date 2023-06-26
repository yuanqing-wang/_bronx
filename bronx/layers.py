import math
from typing import Optional, Callable
from functools import partial
import torch
import pyro
from pyro import poutine
import dgl
# dgl.use_libxsmm(False)
from dgl.nn import GraphConv
from dgl import function as fn
from dgl.nn.functional import edge_softmax


def linear_diffusion(g, h, e, k:int=6):
    g = g.local_var()
    parallel = e.dim() == 4
    if parallel:
        if h.dim() == 2:
            h = h.broadcast_to(e.shape[0], *h.shape)
        e, h = e.swapaxes(0, 1), h.swapaxes(0, 1)

    h = h.reshape(*h.shape[:-1], e.shape[-2], -1)
    g.edata["e"] = e
    g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
    # g.update_all(fn.copy_e("e", "m"), fn.sum("m", "e_sum"))
    # g.apply_edges(lambda edges: {"e": edges.data["e"] / edges.dst["e_sum"]})
    g.ndata["x"] = h
    result = h
    for i in range(1, k+1):
        g.update_all(fn.u_mul_e("x", "e", "m"), fn.sum("m", "x"))
        result = result + g.ndata["x"] / math.factorial(i)
    if parallel:
        result = result.swapaxes(0, 1)

    result = result.flatten(-2, -1)
    return result


class Linear(pyro.nn.PyroModule):
    def __init__(self, in_features, out_features, idx):
        super().__init__()
        self.w_mu = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.w_log_sigma = torch.nn.Parameter(1e-3 * torch.randn(in_features, out_features))
        self.idx = idx
        self.in_features = in_features
        self.out_features = out_features


    def guide(self, x):
        with pyro.plate(f"weight{self.idx}", self.in_features, device=x.device):
            w = pyro.sample(
                    f"w{self.idx}",
                    pyro.distributions.Normal(
                        torch.zeros_like(self.w_mu), 
                        torch.ones_like(self.w_log_sigma.exp()),
                    ).to_event(1),
            )
        return x @ w

    def forward(self, x):
        with pyro.plate(f"weight{self.idx}", self.in_features, device=x.device):
            w = pyro.sample(
                    f"w{self.idx}",
                    pyro.distributions.Normal(
                        self.w_mu, 
                        self.w_log_sigma.exp(),
                    ).to_event(1),
            )
        return x @ w



class BronxLayer(pyro.nn.PyroModule):
    def __init__(
            self, 
            in_features, out_features, activation=torch.nn.SiLU(), 
            dropout=0.0, idx=0, num_heads=4, edge_drop=0.2,
        ):
        super().__init__()
        self.fc_mu = torch.nn.Linear(in_features, in_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(in_features, in_features, bias=False)
        self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, out_features, bias=False)
        self.activation = activation
        self.idx = idx
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        torch.nn.init.normal_(self.fc_mu.weight, std=1e-3)
        torch.nn.init.normal_(self.fc_log_sigma.weight, std=1e-3)

    def guide(self, g, h):
        g = g.local_var()
        h0 = h
        pyro.module(f"fc_mu{self.idx}", self.fc_mu)
        pyro.module(f"fc_log_sigma{self.idx}", self.fc_log_sigma)
        pyro.module(f"fc_k{self.idx}", self.fc_k)
        pyro.module(f"fc_q{self.idx}", self.fc_q)

        mu, log_sigma = self.fc_mu(h), self.fc_log_sigma(h)
        with pyro.plate(f"nodes{self.idx}", g.number_of_nodes()):
            h = pyro.sample(
                    f"h{self.idx}",
                    pyro.distributions.Normal(
                        mu, 
                        log_sigma.exp(),
                    ).to_event(1),
            )
        
        k, q = self.fc_k(h), self.fc_q(h)
        k = k.reshape(*k.shape[:-1], self.num_heads, -1)
        q = q.reshape(*q.shape[:-1], self.num_heads, -1)

        parallel = h.dim() == 3
        if parallel:
            k, q = k.swapaxes(0, 1), q.swapaxes(0, 1)
        g.ndata["k"], g.ndata["q"] = k, q
        g.apply_edges(fn.u_dot_v("k", "q", "e"))
        # g.edata["e"] = g.edata["e"] / math.sqrt(self.out_features / self.num_heads)
        e = edge_softmax(g, g.edata["e"])
        if parallel:
            e = e.swapaxes(0, 1)
        h = linear_diffusion(g, h0, e)
        return h

    def forward(self, g, h):
        g = g.local_var()
        h0 = h
        with pyro.plate(f"nodes{self.idx}", g.number_of_nodes()):
            h = pyro.sample(
                    f"h{self.idx}",
                    pyro.distributions.Normal(
                        torch.zeros_like(h), 
                        torch.ones_like(h),
                    ).to_event(1),
            )

        k, q = self.fc_k(h), self.fc_q(h)
        k = k.reshape(*k.shape[:-1], self.num_heads, -1)
        q = q.reshape(*q.shape[:-1], self.num_heads, -1)

        parallel = h.dim() == 3
        if parallel:
            k, q = k.swapaxes(0, 1), q.swapaxes(0, 1)
        g.ndata["k"], g.ndata["q"] = k, q
        g.apply_edges(fn.u_dot_v("k", "q", "e"))
        # g.edata["e"] = g.edata["e"] / math.sqrt(self.out_features / self.num_heads)
        e = edge_softmax(g, g.edata["e"])
        if parallel:
            e = e.swapaxes(0, 1)
        h = linear_diffusion(g, h0, e)
        return h
