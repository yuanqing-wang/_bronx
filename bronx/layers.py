import math
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
def exp_adj(g, gamma=1.0):
    a = g.adj().to_dense()
    a.fill_diagonal_(gamma)
    a = a / a.sum(-1, keepdims=True)
    a = torch.linalg.matrix_exp(a)
    return a

@torch.jit.script
def approximate_matrix_exp(a, k:int=6):
    result = a
    for i in range(1, k):
        a = a @ a
        result = result + a / math.factorial(i)
    return result

class InLayer(torch.nn.Module):
    def __init__(
            self, in_features, out_features, gamma=1.0,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        self.gamma = gamma
    
    def forward(self, g, h):
        h = self.fc(h)
        h = exp_adj(g, gamma=self.gamma) @ h
        return h

class BronxLayer(torch.nn.Module):
    def __init__(
            self, 
            in_features, out_features, gamma=1.0, idx=0, embedding_features=8,
        ):
        super().__init__()
        self.fc_mu = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_k = torch.nn.Linear(out_features, embedding_features, bias=False)
        self.fc_q = torch.nn.Linear(out_features, embedding_features, bias=False) 
        self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        self.idx = idx
        self.out_features = out_features
        self.embedding_features = embedding_features
        self.gamma = gamma

    def guide(self, g, h):
        pyro.module(f"fc_mu{self.idx}", self.fc_mu)
        pyro.module(f"fc_log_sigma{self.idx}", self.fc_log_sigma)
        pyro.module(f"fc_k{self.idx}", self.fc_k)
        pyro.module(f"fc_q{self.idx}", self.fc_q)
        pyro.module(f"fc{self.idx}", self.fc)
        h0 = self.fc(h)
        with pyro.plate(f"nodes{self.idx}", g.number_of_nodes(), device=g.device):
            with pyro.poutine.scale(None, scale=float(g.ndata["train_mask"].sum() / g.number_of_nodes())):
                h = pyro.sample(
                        f"h{self.idx}", 
                        pyro.distributions.Normal(
                            self.fc_mu(h), 
                            self.fc_log_sigma(h).exp(),
                        ).to_event(1),
                    )
        h = h + h0
        k, q = self.fc_k(h), self.fc_q(h)
        a_att = (k @ q.transpose(-1, -2)).softmax(-1)
        a_str = exp_adj(g, gamma=self.gamma)
        a = a_str * a_att
        a = a / a.sum(-1, keepdims=True)
        h = a @ h
        return h

    def forward(self, g, h):
        pyro.module(f"fc_k{self.idx}", self.fc_k)
        pyro.module(f"fc_q{self.idx}", self.fc_q)
        pyro.module(f"fc{self.idx}", self.fc)
        h0 = self.fc(h)
        with pyro.plate(f"nodes{self.idx}", g.number_of_nodes(), device=g.device):
            with pyro.poutine.scale(None, scale=float(g.ndata["train_mask"].sum() / g.number_of_nodes())):
                h = pyro.sample(
                        f"h{self.idx}", 
                            pyro.distributions.Normal(
                                torch.zeros(g.number_of_nodes(), self.out_features, device=g.device),
                                torch.ones(g.number_of_nodes(), self.out_features, device=g.device),
                            ).to_event(1),
                )

        h = h + h0

        k, q = self.fc_k(h), self.fc_q(h)
        a_att = (k @ q.transpose(-1, -2)).softmax(-1)
        a_str = exp_adj(g, gamma=self.gamma)
        a = a_str * a_att
        a = a / a.sum(-1, keepdims=True)
        h = a @ h
        return h

