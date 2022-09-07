from typing import Optional, Callable
import torch

class BronxLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_features,
        adjacency_matrix,
        diffusion_matrix,
        prior_mu=1.0,
        prior_sigma=1.0,
        num_heads=1,
        activation=torch.nn.ELU(),
        n_samples=1,
    ):
        super().__init__()
        self.fc_k = torch.nn.Linear(hidden_features, hidden_features, bias=False)
        self.fc_mu = torch.nn.Linear(hidden_features, hidden_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(hidden_features, hidden_features, bias=False)
        self.fc_v = torch.nn.Linear(hidden_features * num_heads, hidden_features)
        self.activation = activation
        self.hidden_features = hidden_features
        self.num_heads = num_heads
        self.register_buffer("prior_mu", torch.tensor(prior_mu))
        self.register_buffer("prior_sigma", torch.tensor(prior_sigma))
        self.norm = torch.nn.LayerNorm(hidden_features)
        self.diffusion_matrix = diffusion_matrix
        self.adjacency_matrix = adjacency_matrix
        self.n_samples = n_samples

    def forward(self, h):
        h = self.norm(h)
        k = self.fc_k(h)
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)

        # (n, d, n_heads)
        k = k.reshape(k.shape[0], int(self.hidden_features / self.num_heads), self.num_heads)
        mu = mu.reshape(mu.shape[0], int(self.hidden_features / self.num_heads), self.num_heads)
        log_sigma = log_sigma.reshape(log_sigma.shape[0], int(self.hidden_features / self.num_heads), self.num_heads)

        mu = torch.einsum("xyb,zyb->xzb", k, mu) * (self.hidden_features ** (-0.5))
        log_sigma = torch.einsum("xyb,zyb->xzb", k, log_sigma) * (self.hidden_features ** (-0.5))
        a_distribution = torch.distributions.LogNormal(mu, torch.nn.Softplus()(log_sigma))
        prior = torch.distributions.LogNormal(self.prior_mu.log(), self.prior_sigma)
        kl = torch.distributions.kl_divergence(a_distribution, prior)
        kl = (kl * self.diffusion_matrix.sign()).sum(-1).mean()
        self.kl = kl

        a = a_distribution.rsample([self.n_samples]) * self.diffusion_matrix
        a = torch.nn.functional.normalize(a, p=1, dim=-2)
        h = torch.einsum("...xyb,...yz->...xbz", a, h)
        h = h.flatten(-2, -1)
        h = self.fc_v(h)
        h = self.activation(h)
        h = h.mean(0)
        return h
