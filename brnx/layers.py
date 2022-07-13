from typing import Optional, Callable
import torch

class GCN(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[Callable] = None
    ):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        self.activation = activation
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, a, h):
        d = torch.sparse.sum(a, dim=-1).values().float().clamp(min=1)
        norm = d.pow(-0.5).unsqueeze(-1)
        h = h * norm
        h = self.fc(h)
        h = a @ h
        h = h * norm
        if self.activation is not None:
            h = self.activation(h)
        return h


class VGAE(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.gcn0 = GCN(in_features, out_features, activation=torch.nn.ReLU())
        self.gcn_mu = GCN(out_features, out_features)
        self.gcn_log_sigma = GCN(out_features, out_features)
        self.p_z = torch.distributions.Normal(0, 1)

    def encode(self, a, h):
        h = self.gcn0(a, h)
        mu = self.gcn_mu(a, h)
        log_sigma = self.gcn_log_sigma(a, h)
        sigma = log_sigma.exp()
        return torch.distributions.Normal(mu, sigma)

    def decode(self, q_z):
        z = q_z.rsample()
        p_a = torch.distributions.Bernoulli(
            logits=(z.unsqueeze(0) * z.unsqueeze(1)).sum(-1),
        )
        return p_a

    def loss(self, a, h):
        q_z = self.encode(a, h)
        p_a = self.decode(q_z)
        elbo = p_a.log_prob(a.to_dense()).sum(-1).mean()\
            - torch.distributions.kl_divergence(
                q_z, self.p_z,
            )

        return -elbo
