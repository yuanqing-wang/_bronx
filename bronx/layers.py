from typing import Optional, Callable
import torch
import pyro
import dgl
from dgl.nn import GraphConv
from dgl import function as fn
from dgl.nn.functional import edge_softmax

class BronxLayer(pyro.nn.PyroModule):
    def __init__(
            self, in_features, out_features, 
            embedding_features=None, num_heads=1, index=0,
        ):
        super().__init__()
        if embedding_features is None: embedding_features = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.index = index

        self.fc = pyro.nn.PyroModule[torch.nn.Linear](
            in_features, out_features
        )

        self.fc_k = pyro.nn.PyroModule[torch.nn.Linear](
            in_features, embedding_features,
        )

        self.fc_q_mu = pyro.nn.PyroModule[torch.nn.Linear](
            in_features, embedding_features,
        )

        self.fc_q_log_sigma = pyro.nn.PyroModule[torch.nn.Linear](
            in_features, embedding_features,
        )

        self.num_heads = num_heads

    def mp(self, g, h, e=None):
        g = g.local_var()
        h = self.fc(h).reshape(*h.shape[:-1], self.num_heads, self.in_features)

        if e is None:
            e = h.new_ones(g.number_of_edges(), self.num_heads, 1)

        g.ndata["h"] = h
        g.edata["e"] = edge_softmax(g, e / self.out_features ** 0.5)
        g.update_all(
            fn.u_mul_e("h", "e", "a"),
            fn.sum("a", "h"),
        )
        h = g.ndata["h"].flatten(-2, -1)
        return h

    def model(self, g, h):
        g = g.local_var()
        e = pyro.sample(
            f"e{self.index}",
            pyro.distributions.Normal(
                h.new_zeros(g.number_of_edges(), self.num_heads),
                h.new_zeros(g.number_of_edges(), self.num_heads).exp(),
            ).to_event(2)
        )
        h = self.mp(g, h, e)
        return h

    def guide(self, g, h):
        g = g.local_var()
        k = self.fc_k(h)
        mu, log_sigma = self.fc_q_mu(h), self.fc_q_log_sigma(h)
        g.ndata["mu"], g.ndata["log_sigma"] = mu, log_sigma
        g.apply_edges(fn.u_dot_v("mu", "mu", "mu"))
        g.apply_edges(
            fn.u_dot_v("log_sigma", "log_sigma", "log_sigma"),
        )

        e = pyro.sample(
            f"e{self.index}",
            pyro.distributions.Normal(
                g.edata["mu"], g.edata["log_sigma"],
            ).to_event(2)
        )

        return e

class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, num_heads=1):
        super().__init__()
        if out_features is None: out_features = in_features
        if hidden_features is None: hidden_features = out_features
        self.fc_k = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc = torch.nn.Linear(in_features, out_features)
        self.num_heads = num_heads

    def forward(self, g, h):
        g = g.local_var()
        k = self.fc_k(h).reshape(*h.shape[:-1], self.num_heads, -1)
        q = self.fc_q(h).reshape(*h.shape[:-1], self.num_heads, -1)
        g.ndata["k"] = k
        g.ndata["q"] = q
        g.apply_edges(dgl.function.u_dot_v("k", "q", "e"))
        g.edata["a"] = edge_softmax(g, g.edata["e"])
        g.ndata["h"] = h
        g.update_all(dgl.function.u_mul_e("h", "a", "m"), dgl.function.sum("m", "h"))
        h = self.fc(g.ndata["h"]).flatten(-2, -1)
        return h
