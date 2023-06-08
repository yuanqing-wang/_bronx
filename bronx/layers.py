from math import exp
from typing import Optional, Callable
from functools import partial, lru_cache
import torch
import pyro
from pyro import poutine
import dgl
# dgl.use_libxsmm(False)
from dgl.nn import GraphConv
from dgl import function as fn
from dgl.nn.functional import edge_softmax

@lru_cache(maxsize=1)
def adj(g):
    a = g.adj().to_dense()
    a = a / a.sum(-1, keepdim=True).clamp_min(1)
    return a

class BronxLayer(torch.nn.Module):
    def __init__(
            self, 
            in_features, out_features, gamma=1.0, idx=0,
        ):
        super().__init__()
        self.fc_mu = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(in_features, out_features, bias=False)
        self.idx = idx
        self.out_features = out_features
        self.gamma = gamma
        self.dropout = torch.nn.Dropout(0.5)

    def guide(self, g, h):
        pyro.module(f"fc_mu{self.idx}", self.fc_mu)
        pyro.module(f"fc_log_sigma{self.idx}", self.fc_log_sigma)
        with pyro.plate(f"nodes{self.idx}", g.number_of_nodes(), device=g.device):
            with pyro.poutine.scale(None, scale=float(g.ndata["train_mask"].sum() / g.number_of_nodes())):
                h = pyro.sample(
                        f"h{self.idx}", 
                        pyro.distributions.Normal(
                            self.fc_mu(h), 
                            self.fc_log_sigma(h).exp(),
                        ).to_event(1),
                    )
        h = self.dropout(h)
        h = adj(g) @ h
        return h

    def forward(self, g, h):
        with pyro.plate(f"nodes{self.idx}", g.number_of_nodes(), device=g.device):
            with pyro.poutine.scale(None, scale=float(g.ndata["train_mask"].sum() / g.number_of_nodes())):
                h = pyro.sample(
                        f"h{self.idx}", 
                            pyro.distributions.Normal(
                                torch.zeros(g.number_of_nodes(), self.out_features, device=g.device),
                                torch.ones(g.number_of_nodes(), self.out_features, device=g.device),
                            ).to_event(1),
                )
        h = self.dropout(h)
        h = adj(g) @ h
        return h

