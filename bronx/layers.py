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

def linear_diffusion(g, h, e, k=6, t=1, gamma=0.0):
    g = g.local_var()
    parallel = e.dim() == 4
    if parallel:
        if h.dim() == 2:
            h = h.broadcast_to(e.shape[0], *h.shape)
        e, h = e.swapaxes(0, 1), h.swapaxes(0, 1)

    h = h.reshape(*h.shape[:-1], e.shape[-2], -1)
    g.edata["e"] = e
    g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
    g.update_all(fn.copy_e("e", "m"), fn.sum("m", "e_sum"))
    g.apply_edges(lambda edges: {"e": edges.data["e"] / edges.dst["e_sum"]})

    g = dgl.add_self_loop(g)
    src, dst = g.edges()
    g.edata["e"][src==dst, ...] = gamma

    g.ndata["x"] = h
    result = h
    g.edata["e"] = t * g.edata["e"]
    for i in range(1, k + 1):
        g.update_all(fn.u_mul_e("x", "e", "m"), fn.sum("m", "x"))
        result = result + g.ndata["x"] / math.factorial(i)
    if parallel:
        result = result.swapaxes(0, 1)
    result = result.flatten(-2, -1)
    return result

class BronxLayer(pyro.nn.PyroModule):
    def __init__(
            self, 
            in_features, 
            out_features, 
            activation=torch.nn.SiLU(), 
            idx=0, 
            num_heads=4,
            sigma_factor=1.0,
            kl_scale=1.0,
            t=1.0,
            gamma=1.0,
        ):
        super().__init__()
        self.fc_mu = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(in_features, out_features, bias=False)
        self.activation = activation
        self.idx = idx
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.sigma_factor = sigma_factor
        self.kl_scale = kl_scale
        self.t = t
        self.gamma = gamma

    def guide(self, g, h, kl_anneal=1.0):
        g = g.local_var()
        h0 = h
        h = h - h.mean(-1, keepdims=True)
        h = torch.nn.functional.normalize(h, dim=-1)
        mu, log_sigma = self.fc_mu(h), self.fc_log_sigma(h)
        mu = mu.reshape(*mu.shape[:-1], self.num_heads, -1)
        log_sigma = log_sigma.reshape(
            *log_sigma.shape[:-1], self.num_heads, -1
        )

        parallel = h.dim() == 3

        if parallel:
            mu, log_sigma = mu.swapaxes(0, 1), log_sigma.swapaxes(0, 1)

        g.ndata["mu"], g.ndata["log_sigma"] = mu, log_sigma
        g.apply_edges(dgl.function.u_dot_v("mu", "mu", "mu"))
        g.apply_edges(
            dgl.function.u_dot_v("log_sigma", "log_sigma", "log_sigma")
        )
        mu, log_sigma = g.edata["mu"], g.edata["log_sigma"]

        if parallel:
            mu, log_sigma = mu.swapaxes(0, 1), log_sigma.swapaxes(0, 1)

        with pyro.plate(
            f"edges{self.idx}", g.number_of_edges(), device=g.device
        ):
            with pyro.poutine.scale(None, self.kl_scale * kl_anneal):
                e = pyro.sample(
                    f"e{self.idx}",
                    pyro.distributions.TransformedDistribution(
                        pyro.distributions.Normal(
                            mu,
                            self.sigma_factor * log_sigma.exp(),
                        ),
                        pyro.distributions.transforms.SigmoidTransform(),
                    ).to_event(2),
                )

        h = linear_diffusion(g, h0, e, t=self.t, gamma=self.gamma)
        return h

    def forward(self, g, h, kl_anneal=1.0):
        g = g.local_var()
        h0 = h
        with pyro.plate(
            f"edges{self.idx}", g.number_of_edges(), device=g.device
        ):
            with pyro.poutine.scale(None, self.kl_scale * kl_anneal):
                e = pyro.sample(
                    f"e{self.idx}",
                    pyro.distributions.TransformedDistribution(
                        pyro.distributions.Normal(
                            torch.zeros(
                                g.number_of_edges(),
                                self.num_heads,
                                1,
                                device=g.device,
                            ),
                            self.sigma_factor * torch.ones(
                                g.number_of_edges(),
                                self.num_heads,
                                1,
                                device=g.device,
                            ),
                        ),
                        pyro.distributions.transforms.SigmoidTransform(),
                    ).to_event(2),
                )

        h = linear_diffusion(g, h, e, t=self.t, gamma=self.gamma)
        return h
