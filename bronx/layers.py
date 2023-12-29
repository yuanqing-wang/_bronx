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

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

class ODEFunc(torch.nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.g = None
        self.edge_shape = None
        self.node_shape = None
        self.h0 = None
        self.register_buffer("gamma", torch.tensor(gamma))
        
    def forward(self, t, x):
        h, e = x[:self.node_shape.numel()], x[self.node_shape.numel():]
        h, e = h.reshape(*self.node_shape), e.reshape(*self.edge_shape)
        h0 = h
        g = self.g.local_var()
        g.edata["e"] = e
        g.ndata["h"] = h 
        g.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))
        h = g.ndata["h"]
        # h = h / g.in_degrees().float().clamp(min=1).view(-1, *((1,) * (h.dim()-1)))
        # h = h.tanh()
        h = h - h0 * self.gamma
        if self.h0 is not None:
            h = h + self.h0
        h, e = h.flatten(), e.flatten()
        x = torch.cat([h, torch.zeros_like(e)])
        return x

class LinearDiffusion(torch.nn.Module):
    def __init__(self, t, adjoint=False, physique=False, gamma=1.0):
        super().__init__()
        self.odefunc = ODEFunc(gamma=gamma)
        self.register_buffer("t", torch.tensor(t))
        self.physique = physique
        if adjoint:
            self.integrator = odeint_adjoint
        else:
            self.integrator = odeint
        

    def forward(self, g, h, e):
        g = g.local_var()

        parallel = e.dim() == 4
        if parallel:
            if h.dim() == 2:
                h = h.broadcast_to(e.shape[0], *h.shape)
            e, h = e.swapaxes(0, 1), h.swapaxes(0, 1)

        h = h.reshape(*h.shape[:-1], e.shape[-2], -1)
        g.edata["e"] = e
        g.update_all(fn.copy_e("e", "m"), fn.sum("m", "e_sum"))
        g.apply_edges(lambda edges: {"e": edges.data["e"] / edges.dst["e_sum"]})
        node_shape = h.shape
        if self.physique:
            self.odefunc.h0 = h.detach().clone()
        self.odefunc.node_shape = node_shape
        self.odefunc.edge_shape = g.edata["e"].shape
        self.odefunc.g = g
        t = torch.tensor([0.0, self.t], device=h.device, dtype=h.dtype)
        x = torch.cat([h.flatten(), g.edata["e"].flatten()])
        x = self.integrator(self.odefunc, x, t, method="dopri5")[-1]
        # x = self.integrator(self.odefunc, x, t, method="rk4", options={"step_size": 0.1})[-1]
        h, e = x[:h.numel()], x[h.numel():]
        h = h.reshape(*node_shape)
        if parallel:
            h = h.swapaxes(0, 1)
        h = h.flatten(-2, -1)
        return h

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
            adjoint=False,
            physique=False,
            gamma=1.0,
            norm=False,
            dropout=0.0,
            node_prior=False,
        ):
        super().__init__()
        self.fc_mu = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)

        torch.nn.init.constant_(self.fc_k.weight, 1e-5)
        torch.nn.init.constant_(self.fc_log_sigma.weight, 1e-5)
        torch.nn.init.constant_(self.fc_mu.weight, 1e-5) 

        self.activation = activation
        self.idx = idx
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.sigma_factor = sigma_factor
        self.kl_scale = kl_scale
        self.linear_diffusion = LinearDiffusion(
            t, adjoint=adjoint, physique=physique, gamma=gamma,
        )


        if norm:
            self.norm = torch.nn.LayerNorm(in_features)
        else:
            self.norm = None

        self.dropout = torch.nn.Dropout(dropout)

    def guide(self, g, h):
        g = g.local_var()
        if self.norm:
            h = self.norm(h)
        h = self.dropout(h)
        mu, log_sigma = self.fc_mu(h), self.fc_log_sigma(h)

        mu = mu.reshape(*mu.shape[:-1], self.num_heads, -1)
        log_sigma = log_sigma.reshape(
            *log_sigma.shape[:-1], self.num_heads, -1
        )

        parallel = h.dim() == 3

        if parallel:
            mu, log_sigma = mu.swapaxes(0, 1), log_sigma.swapaxes(0, 1)

        g.ndata["mu"], g.ndata["log_sigma"], g.ndata["k"] = mu, log_sigma, k

        if parallel:
            mu, log_sigma = mu.swapaxes(0, 1), log_sigma.swapaxes(0, 1)

        with pyro.plate(
            f"nodes{self.idx}", g.number_of_nodes(), device=g.device
        ):
            with pyro.poutine.scale(None, self.kl_scale):
                pyro.sample(
                    f"h{self.idx}",
                    pyro.distributions.Normal(
                        mu, self.sigma_factor * log_sigma.exp(),
                    ),
                )

        k, q = self.fc_k(h), self.fc_q(h)
        k, q = k.reshape(*k.shape[:-1], self.num_heads, -1), q.reshape(
            *q.shape[:-1], self.num_heads, -1
        )

        g.ndata["k"], g.ndata["q"] = k, q
        g.apply_edges(fn.u_mul_v("k", "q", "e"))
        e = g.edata["e"]
        h = self.linear_diffusion(g, h, e)
        return h

    def forward(self, g, h, he=None):
        g = g.local_var()

        mu = torch.zeros(
            g.number_of_nodes(),
            self.num_heads,
            1,
            device=g.device,
        )

        sigma = self.sigma_factor * torch.ones(
            g.number_of_nodes(),
            self.num_heads,
            1,
            device=g.device,
        )

        with pyro.plate(
            f"nodes{self.idx}", g.number_of_nodes(), device=g.device
        ):
            with pyro.poutine.scale(None, self.kl_scale):
                h = pyro.sample(
                    f"h{self.idx}",
                    pyro.distributions.Normal(
                        mu, sigma,
                    ),
                ).to_event(2),

        h = self.linear_diffusion(g, h, e)
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
        h = self.fc(h)
        g = dgl.add_reverse_edges(g)
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

class NeighborhoodRecover(pyro.nn.PyroModule):
    def __init__(self, in_features, scale=1.0):
        super().__init__()
        self.fc_mu = torch.nn.Linear(in_features, in_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(in_features, in_features, bias=False)
        self.scale = scale

    def forward(self, g, h):
        g = g.local_var()
        g = dgl.add_reverse_edges(g)
        mu, log_sigma = self.fc_mu(h), self.fc_log_sigma(h)
        src, dst = g.edges()
        mu, log_sigma = mu[..., src, :], log_sigma[..., src, :]
        h = h[..., dst, :]

        with pyro.poutine.scale(None, self.scale):
            with pyro.plate("neighborhood_recover_plate", g.number_of_edges(), device=g.device):
                pyro.sample(
                    "neighborhood_recover",
                    pyro.distributions.Normal(
                        mu, 
                        log_sigma.exp(),
                    ).to_event(1),
                    obs=h,
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

class ConsistencyRegularizer(torch.nn.Module):
    def __init__(self, temperature, factor):
        super().__init__()
        self.temperature = temperature
        self.factor = factor

    def forward(self, probs):
        if probs.dim() == 2:
            avg_probs = probs
        else:
            avg_probs = probs.mean(0)

        sharpened_probs = avg_probs.pow(1.0 / self.temperature)
        sharpened_probs = sharpened_probs / sharpened_probs.sum(-1, keepdims=True)
        loss = (sharpened_probs - probs).pow(2).sum()
        pyro.factor("consistency_regularizer", -loss * self.factor)
        return loss





                

