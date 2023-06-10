from math import factorial
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

def get_candidates(a, k=4):
    return torch.stack([torch.matrix_power(a, i) for i in range(1, k + 1)], dim=-3)

@lru_cache(maxsize=1)
def get_coefficients(k=4):
    return torch.tensor([float(1 / factorial(i)) for i in range(1, k + 1)])

class Rewire(pyro.nn.PyroModule):
    def __init__(self, in_features, out_features, k=6, idx=0):
        super().__init__()
        self.k = k
        self.register_buffer("coefficients", torch.tensor(get_coefficients(k=k)))
        self.register_buffer("_mu", torch.zeros(k))
        self.register_buffer("_log_sigma", torch.ones(k))
        self.mu = pyro.nn.PyroParam(
            self._mu, constraint=torch.distributions.constraints.real
        )
        self.log_sigma = pyro.nn.PyroParam(
            self._log_sigma, 
            constraint=torch.distributions.constraints.positive
        )
        self.idx = idx 
        
        self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, g, h):
        pyro.module(f"fc_k{self.idx}", self.fc_k)
        pyro.module(f"fc_q{self.idx}", self.fc_q)
        a = g.adj().to_dense()
        k = self.fc_k(h)
        q = self.fc_q(h)
        att = (torch.matmul(k, q.transpose(-1, -2))).softmax(-1)
        a = att * a
        candidates = get_candidates(a, k=self.k)
        theta = pyro.sample(
            f"theta{self.idx}",
            pyro.distributions.LogNormal(
                torch.zeros(self.k, device=candidates.device), 
                torch.ones(self.k, device=candidates.device),
            ).to_event(1),
        ).squeeze()
        theta = (self.coefficients * theta).unsqueeze(-1).unsqueeze(-1)

        
        a = (candidates * theta).sum(-3)
        return a
    
    def guide(self, g, h):
        a = g.adj().to_dense()
        candidates = get_candidates(a, k=self.k)
        theta = pyro.sample(
            f"theta{self.idx}",
            pyro.distributions.LogNormal(
                self.mu, self.log_sigma,
            ).to_event(1),
        ).squeeze()
        theta = (self.coefficients * theta).unsqueeze(-1).unsqueeze(-1)
        a = (candidates * theta).sum(-3)
        return a

class BronxLayer(torch.nn.Module):
    def __init__(
            self, 
            in_features, out_features, gamma=1.0, idx=0, k=4,
        ):
        super().__init__()
        self.fc_mu = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(in_features, out_features, bias=False)
        self.idx = idx
        self.out_features = out_features
        self.gamma = gamma
        self.rewire = Rewire(out_features, 8, k=k, idx=idx)

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

        a = self.rewire.guide(g, h)
        h = a @ h
        return h

    def forward(self, g, h):
        pyro.module(f"fc_mu{self.idx}", self.fc_mu)
        with pyro.plate(f"nodes{self.idx}", g.number_of_nodes(), device=g.device):
            with pyro.poutine.scale(None, scale=float(g.ndata["train_mask"].sum() / g.number_of_nodes())):
                h = pyro.sample(
                        f"h{self.idx}", 
                            pyro.distributions.Normal(
                                torch.zeros(g.number_of_nodes(), self.out_features, device=g.device),
                                torch.ones(g.number_of_nodes(), self.out_features, device=g.device),
                            ).to_event(1),
                )

        a = self.rewire(g, h)
        h = a @ h
        return h

