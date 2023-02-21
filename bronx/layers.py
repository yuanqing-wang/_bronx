from typing import Optional, Callable
from functools import partial
import torch
import pyro
from pyro import poutine
from pyro import distributions as dist
import dgl
from dgl.nn import GraphConv
from dgl import function as fn
from dgl.nn.functional import edge_softmax

class BayesianLinear(torch.nn.Linear, pyro.nn.PyroModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if torch.cuda.is_available():
            self.weight = pyro.nn.PyroSample(
                dist.Normal(
                    torch.tensor(0.0, device="cuda:0"),
                    torch.tensor(1.0, device="cuda:0"),
                ).expand(self.weight.shape).to_event(2),
            )

        else:

            self.weight = pyro.nn.PyroSample(
                dist.Normal(
                    torch.tensor(0.0),
                    torch.tensor(1.0),
                ).expand(self.weight.shape).to_event(2),
            )

        self.guide = pyro.infer.autoguide.AutoNormal(self)



class BronxLayer(pyro.nn.PyroModule):
    def __init__(
            self, in_features, out_features, 
            embedding_features=None, num_heads=1, index=0,
            bayesian_weights=False,
        ):
        super().__init__()
        if embedding_features is None: embedding_features = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.embedding_features = int(out_features / num_heads)
        self.index = index

        if bayesian_weights:
            Linear = BayesianLinear
        else:
            Linear = pyro.nn.PyroModule[torch.nn.Linear]

        self.fc = Linear(
            in_features, embedding_features, bias=False, # out_features
        )

        self.fc_k = Linear(
            in_features, embedding_features, bias=False,
        )

        self.fc_q_mu = Linear(
            in_features, embedding_features, bias=False,
        )

        self.fc_q_log_sigma = Linear(
            in_features, embedding_features, bias=False,
        )

        self.num_heads = num_heads
        self.bayesian_weights = bayesian_weights

    def mp(self, g, h, e=None):
        g = g.local_var()
        h = h.reshape(*h.shape[:-1], self.num_heads, -1)
       
        if e is None:
            e = h.new_zeros(g.number_of_edges(), self.num_heads, 1)
            
        g.ndata["h"] = h
        g.edata["e"] = edge_softmax(g, e / self.out_features ** 0.5)
        g.update_all(
            fn.u_mul_e("h", "e", "a"),
            fn.sum("a", "h"),
        )
        # h = self.fc(h)
        h = h.flatten(-2, -1) 
        return h

    def forward(self, g, h):
        g = g.local_var()
        
        with pyro.plate(f"_d{self.index}", self.embedding_features):
            with pyro.plate(f"_e{self.index}", g.number_of_edges()):
                e = pyro.sample(
                    f"e{self.index}",
                    pyro.distributions.Normal(
                        h.new_zeros(size=(),),
                        h.new_ones(size=(),),
                    ).expand([self.num_heads]).to_event(1)
                ).swapaxes(-1, -2)


        h = self.mp(g, h, e)
        return h


    def guide(self, g, h):
        g = g.local_var()

        h = h.reshape(*h.shape[:-1], self.num_heads, -1)
        k = self.fc_k(h)
        mu, log_sigma = self.fc_q_mu(h), self.fc_q_log_sigma(h)

        g.ndata["mu"], g.ndata["log_sigma"] = mu, log_sigma
        g.apply_edges(fn.u_dot_v("mu", "mu", "mu"))
        g.apply_edges(
            fn.u_dot_v("log_sigma", "log_sigma", "log_sigma"),
        )

        with pyro.plate(f"_d{self.index}", self.embedding_features):
            with pyro.plate(f"_e{self.index}", g.number_of_edges()):
                e = pyro.sample(
                    f"e{self.index}",
                    pyro.distributions.Normal(
                        g.edata["mu"], 
                        g.edata["log_sigma"].exp(),
                    ).to_event(1)
                ).swapaxes(-1, -2)
        
        return e
