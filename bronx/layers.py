from typing import Optional, Callable
from functools import partial
import torch
import pyro
from pyro import poutine
import dgl
from dgl.nn import GraphConv
from dgl import function as fn
from dgl.nn.functional import edge_softmax

def linear_diffusion(g, h, e=None, gamma=0.1):
    a = g.adj()
    if e is not None:
        a.val.mul_(e)
    a = a.to_dense()
    a.fill_diagonal_(gamma)
    a = a / a.sum(-1, keepdims=True)
    a = torch.linalg.matrix_exp(a)
    return a @ h

class BronxLayer(pyro.nn.PyroModule):
    def __init__(
            self, 
            in_features, out_features, activation=torch.nn.SiLU(), 
            dropout=0.0,
            idx=0,
        ):
        super().__init__()
        self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_mu = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(in_features, out_features, bias=False)
        self.activation = activation
        self.idx = idx
        self.out_features = out_features
        self.dropout = torch.nn.Dropout(dropout)

    def guide(self, g, h):
        k, mu, log_sigma = self.fc_k(h), self.fc_mu(h), self.fc_log_sigma(h)
        g.ndata["k"], g.ndata["mu"], g.ndata["log_sigma"] = k, mu, log_sigma
        g.apply_edges(fn.u_dot_v("k", "mu", "mu"))
        g.apply_edges(fn.u_dot_v("k", "log_sigma", "log_sigma"))
        with pyro.plate(f"edges{self.idx}", g.number_of_edges(), device=g.device):
            e = pyro.sample(
                    f"e{self.idx}", 
                    pyro.distributions.Normal(
                    g.edata["mu"], g.edata["log_sigma"].exp(),
               ).to_event(1)
            )
        return e

    def mp(self, g, h, e):
        e = e / (self.out_features ** 0.5)
        e = edge_softmax(g, e).squeeze(-1)
        h = linear_diffusion(g, h, e=e)
        return h

    def forward(self, g, h):
        with pyro.plate(f"edges{self.idx}", g.number_of_edges(), device=g.device):
            e = pyro.sample(
                    f"e{self.idx}", 
                    pyro.distributions.Normal(
                        torch.zeros(g.number_of_edges(), 1, device=g.device),
                        torch.ones(g.number_of_edges(), 1, device=g.device),
               ).to_event(1)
            )

        h = self.mp(g, h, e)
        h = self.dropout(h)
        return h

