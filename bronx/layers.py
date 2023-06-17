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

@torch.jit.script
def approximate_matrix_exp(a, k:int=6):
    result = a
    for i in range(1, k):
        a = a @ a
        result = result + a / math.factorial(i)
    return result

def expmm(a, b, k:int=6):
    x = b
    result = x
    for i in range(1, k+1):
        x = a @ x
        result = result + x / math.factorial(i)
    return result

class LinearDiffusion(torch.nn.Module):
    def __init__(self, dropout=0.0, gamma=0.7):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.gamma = gamma

    def forward(self, g, h, e=None):
        num_heads = e.shape[-2]
        a = g.adj().to_dense()
        a = a.unsqueeze(-3).repeat_interleave(e.shape[-2], dim=-3)

        if e.dim() == 3:
            a = a.unsqueeze(-4).repeat_interleave(e.shape[-3], dim=-4)
        src, dst = g.edges()

        a[..., src, dst] = e
        a[..., dst, src] = e
        a = a / a.sum(-1, keepdims=True)
        h = h.reshape(*h.shape[:-2], num_heads, g.number_of_nodes(), -1)
        h = expmm(a, h)
        h = h.reshape(*h.shape[:-3], g.number_of_nodes(), -1)
        return h

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
        self.linear_diffusion = LinearDiffusion()

    def guide(self, g, h):
        h = h - h.mean(-1, keepdims=True)
        h = torch.nn.functional.normalize(h, dim=-1)
        mu, log_sigma = self.fc_mu(h), self.fc_log_sigma(h)
     
        src, dst = g.edges()
        mu = mu[..., src, :] * mu[..., dst, :]
        log_sigma = log_sigma[..., src, :] * log_sigma[..., dst, :]

        mu = mu.reshape(
            *mu.shape[:-2], self.num_heads, g.number_of_edges(), -1
        ).sum(-1, keepdims=True)
        log_sigma = log_sigma.reshape(
            *log_sigma.shape[:-2], self.num_heads, g.number_of_edges(), -1
        ).sum(-1, keepdims=True)
        

        with pyro.plate(f"edges{self.idx}", g.number_of_edges(), device=g.device):
            with pyro.plate(f"heads{self.idx}", self.num_heads, device=g.device):
                with pyro.poutine.scale(None, float(g.ndata["train_mask"].sum() / (2 * g.number_of_edges()))):
                    e = pyro.sample(
                            f"e{self.idx}", 
                            pyro.distributions.LogNormal(
                            mu, log_sigma.exp(),
                        ).to_event(1)
                    )

        return e

    def mp(self, g, h, e):
        h = self.linear_diffusion(g, h, e.squeeze(-1))
        return h

    def forward(self, g, h):
        with pyro.plate(f"edges{self.idx}", g.number_of_edges(), device=g.device):
            with pyro.plate(f"heads{self.idx}", self.num_heads, device=g.device):
                with pyro.poutine.scale(None, float(g.ndata["train_mask"].sum() / (2 * g.number_of_edges()))):
                    e = pyro.sample(
                            f"e{self.idx}", 
                            pyro.distributions.LogNormal(
                                torch.zeros(self.num_heads, g.number_of_edges(), 1, device=g.device),
                                torch.ones(self.num_heads, g.number_of_edges(), 1, device=g.device),
                        ).to_event(1)
                    )

        h = self.mp(g, h, e)
        h = self.dropout(h)
        return h

