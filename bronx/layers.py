from typing import Optional, Callable
from functools import partial
import torch
import pyro
from pyro import poutine
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
        self.embedding_features = int(out_features / num_heads)
        self.index = index

        self.fc = pyro.nn.PyroModule[torch.nn.Linear](
            in_features, out_features, bias=False,
        )

        self.fc_k = pyro.nn.PyroModule[torch.nn.Linear](
            embedding_features, embedding_features, bias=False,
        )

        self.fc_q_mu = pyro.nn.PyroModule[torch.nn.Linear](
            embedding_features, embedding_features, bias=False,
        )

        self.fc_q_log_sigma = pyro.nn.PyroModule[torch.nn.Linear](
            embedding_features, embedding_features, bias=False,
        )

        self.num_heads = num_heads

    def mp(self, g, h, e=None):
        g = g.local_var()
        h = h.reshape(*h.shape[:-1], self.num_heads, -1)
        if e is None:
            e = h.new_zeros(g.number_of_edges(), self.num_heads, 1)
        h = self.fc(h)
        g.ndata["h"] = h
        g.edata["e"] = edge_softmax(g, e / self.embedding_features ** 0.5)
        g.update_all(
            fn.u_mul_e("h", "e", "a"),
            fn.sum("a", "h"),
        )
        g.update_all(
            fn.copy_e("e", "m"),
            fn.sum("m", "e_sum")
        )
        h = g.ndata["h"] / (g.ndata["e_sum"].relu() + 1e-8)
        h = h.flatten(-2, -1)
        return h

    def model(self, g, h):
        g = g.local_var()

        with pyro.plate(f"_e{self.index}", g.number_of_edges()):
            e = pyro.sample(
                f"e{self.index}",
                pyro.distributions.Normal(
                    h.new_zeros(size=(g.number_of_edges(), self.num_heads, self.out_features),),
                    h.new_ones(size=(g.number_of_edges(), self.num_heads, self.out_features),),
                ).to_event(2)
            )

        h = self.mp(g, h, e)
        return h

    def guide(self, g, h):
        g = g.local_var()
        h = self.fc(h)
        h = h.reshape(*h.shape[:-1], self.num_heads, -1)
        k = self.fc_k(h)# .tanh()
        mu, log_sigma = self.fc_q_mu(h).tanh(), self.fc_q_log_sigma(h).tanh()
        g.ndata["mu"], g.ndata["log_sigma"] = mu, log_sigma
        g.apply_edges(fn.u_dot_v("mu", "mu", "mu"))
        g.apply_edges(
            fn.u_dot_v("log_sigma", "log_sigma", "log_sigma"),
        )

        with pyro.plate(f"_e{self.index}", g.number_of_edges()):
            e = pyro.sample(
                f"e{self.index}",
                pyro.distributions.Normal(
                    g.edata["mu"].expand(g.number_of_edges(), self.num_heads, self.out_features), 
                    torch.nn.functional.softplus(g.edata["log_sigma"].expand(g.number_of_edges(), self.num_heads, self.out_features)),
                ).to_event(2)
            ).relu()
        return e
