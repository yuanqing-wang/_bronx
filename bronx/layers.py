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

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

class ODEFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._g = None
        self._e = None

    @property
    def g(self):
        return self._g

    @property
    def e(self):
        return self._e
    
    @g.setter
    def g(self, g):
        self._g = g

    @e.setter
    def e(self, e):
        self._e = e

    def forward(self, t, x):
        g = self.g
        e = self.e
        g = g.local_var()
        g.edata["e"] = e
        g.ndata["x"] = x
        g.update_all(fn.u_mul_e("x", "e", "m"), fn.sum("m", "x"))
        return g.ndata["x"]

class ODEBlock(torch.nn.Module):
    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc

    def forward(self, g, h, e, t=1.0):
        g = g.local_var()
        self.odefunc.g = g
        self.odefunc.e = e
        t = torch.tensor([0, t], device=h.device, dtype=h.dtype)
        y = odeint(self.odefunc, h, t, method="rk4")[1]
        return y

class LinearDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ode_block = ODEBlock(ODEFunc())

    def forward(self, g, h, e, t=1.0, gamma=0.0):
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

        result = self.ode_block(g, h, g.edata["e"], t=t)

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

        torch.nn.init.constant_(self.fc_mu.weight, 1e-3)
        torch.nn.init.constant_(self.fc_log_sigma.weight, 1e-3)

        self.activation = activation
        self.idx = idx
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.sigma_factor = sigma_factor
        self.kl_scale = kl_scale
        self.t = t
        self.gamma = gamma
        self.linear_diffusion = LinearDiffusion()

    def guide(self, g, h):
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
            with pyro.poutine.scale(None, self.kl_scale):
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

        h = self.linear_diffusion(g, h0, e, t=self.t, gamma=self.gamma)
        return h

    def forward(self, g, h):
        g = g.local_var()
        h0 = h
        with pyro.plate(
            f"edges{self.idx}", g.number_of_edges(), device=g.device
        ):
            with pyro.poutine.scale(None, self.kl_scale):
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

        h = self.linear_diffusion(g, h, e, t=self.t, gamma=self.gamma)
        return h

class NodeRecover(pyro.nn.PyroModule):
    def __init__(self, in_features, out_features, scale=1.0):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        self.scale = scale

    def forward(self, g, h, y):
        g = g.local_var()
        h = h - h.mean(-1, keepdims=True)
        h = torch.nn.functional.normalize(h, dim=-1)
        h = self.fc(h)
        with pyro.poutine.scale(None, self.scale):
            with pyro.plate("nodes_recover", g.number_of_nodes(), device=g.device):
                pyro.sample(
                    "node_recover",
                    pyro.distributions.Bernoulli(h.sigmoid()).to_event(1),
                    obs=1.0 * (y > 0),
                )

class EdgeRecover(pyro.nn.PyroModule):
    def __init__(self, in_features, out_features, scale=1.0):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        self.scale = scale

    def forward(self, g, h):
        g = g.local_var()
        h = h - h.mean(-1, keepdims=True)
        h = torch.nn.functional.normalize(h, dim=-1)
        h = self.fc(h)
        src, dst = g.edges()
        src_fake = torch.randint(high=g.number_of_nodes(), size=(g.number_of_edges(),), device=g.device)
        dst_fake = torch.randint(high=g.number_of_nodes(), size=(g.number_of_edges(),), device=g.device)

        with pyro.poutine.scale(None, self.scale):
            with pyro.plate("real_edges", g.number_of_edges(), device=g.device):
                pyro.sample(
                    "edge_recover_real",
                    pyro.distributions.Bernoulli(
                            (h[..., src, :] * h[..., dst, :]).sum(-1).sigmoid(),
                    ),
                    obs=torch.ones(g.number_of_edges(), device=g.device),
                )

                pyro.sample(
                    "edge_recover_fake",
                    pyro.distributions.Bernoulli(
                        torch.sigmoid(
                            (h[..., src_fake, :] * h[..., dst_fake, :]).sum(-1)
                        ),
                    ),
                    obs=torch.zeros(g.number_of_edges(), device=g.device),
                )

class BatchedLSTM(torch.nn.LSTM):
    def forward(self, h):
        if h.dim() > 3:
            event_shape = h.shape[2:]
            batch_shape = h.shape[:2]
            h = h.reshape(-1, *event_shape)
            _, (h, __) = super().forward(h)
            h = h.reshape(*batch_shape, -1)
            return h
        else:
            output, (h, c) = super().forward(h)
            h = h.squeeze(0)
            return h





                

