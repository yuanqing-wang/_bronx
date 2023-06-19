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
    parallel = e.dim() == 3
    if parallel:
        if h.dim() == 2:
            h = h.broadcast_to(e.shape[0], *h.shape)
        e, h = e.swapaxes(0, 1), h.swapaxes(0, 1)

    g.edata["e"] = e
    g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
    g.update_all(fn.copy_e("e", "m"), fn.sum("m", "e_sum"))
    g.apply_edges(lambda edges: {"e": edges.data["e"] / edges.dst["e_sum"]})
    g.ndata["x"] = h
    result = h
    for i in range(1, k+1):
        g.update_all(fn.u_mul_e("x", "e", "m"), fn.sum("m", "x"))
        result = result + g.ndata["x"] / math.factorial(i)
    if parallel:
        result = result.swapaxes(0, 1)
    return result

class BronxLayer(pyro.nn.PyroModule):
    def __init__(
            self, 
            in_features, out_features, activation=torch.nn.SiLU(), 
            dropout=0.0, idx=0, num_heads=4, edge_drop=0.0,
        ):
        super().__init__()
        # self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_mu = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(in_features, out_features, bias=False)
        self.activation = activation
        self.idx = idx
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(in_features)

    def guide(self, g, h):
        g = g.local_var()
        h = h - h.mean(-1, keepdims=True)
        h = torch.nn.functional.normalize(h, dim=-1)
        mu, log_sigma = self.fc_mu(h), self.fc_log_sigma(h)
     
        parallel = h.dim() == 3

        if parallel:
            mu, log_sigma = mu.swapaxes(0, 1), log_sigma.swapaxes(0, 1)

        g.ndata["mu"], g.ndata["log_sigma"] = mu, log_sigma
        g.apply_edges(dgl.function.u_dot_v("mu", "mu", "mu"))
        g.apply_edges(dgl.function.u_dot_v("log_sigma", "log_sigma", "log_sigma"))
        mu, log_sigma = g.edata["mu"], g.edata["log_sigma"]

        if parallel:
            mu, log_sigma = mu.swapaxes(0, 1), log_sigma.swapaxes(0, 1)

        with pyro.plate(f"edges{self.idx}", g.number_of_edges(), device=g.device):
            with pyro.poutine.scale(None, float(g.ndata["train_mask"].sum() / (2 * g.number_of_edges()))):
                e = pyro.sample(
                        f"e{self.idx}", 
                        pyro.distributions.LogNormal(
                        mu, log_sigma.exp(),
                    ).to_event(1)
                )

        return e

    def mp(self, g, h, e):
        h = linear_diffusion(g, h, e)
        return h

    def forward(self, g, h):
        g = g.local_var()
        with pyro.plate(f"edges{self.idx}", g.number_of_edges(), device=g.device):
            with pyro.poutine.scale(None, float(g.ndata["train_mask"].sum() / (2 * g.number_of_edges()))):
                e = pyro.sample(
                        f"e{self.idx}", 
                        pyro.distributions.LogNormal(
                            torch.zeros(g.number_of_edges(), 1, device=g.device),
                            torch.ones(g.number_of_edges(), 1, device=g.device),
                    ).to_event(1)
                )

        h = self.mp(g, h, e)
        h = self.dropout(h)
        return h

