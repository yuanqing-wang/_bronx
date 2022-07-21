from typing import Optional, Callable
import torch

def _sum(x, *args, **kwargs):
    if x.is_sparse:
        return torch.sparse.sum(x, *args, **kwargs)
    else:
        return torch.sum(x, *args, **kwargs)

class ShiftedELU(torch.nn.Module):
    def forward(self, x):
        return torch.nn.ELU()(x) + 1.5

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
        d = _sum(a, dim=-1).values().float().clamp(min=1)
        norm = d.pow(-0.5).unsqueeze(-1)
        h = h * norm
        h = self.fc(h)
        h = a @ h
        h = h * norm
        if self.activation is not None:
            h = self.activation(h)
        return h

class BVGAE(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
    ):
        super().__init__()
        self.gcn0 = GCN(in_features, hidden_features, activation=torch.nn.ReLU())
        self.gcn1 = GCN(hidden_features, hidden_features)
        self.layer_alpha = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, 1),
            ShiftedELU(),
        )
        self.layer_beta = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, 1),
            ShiftedELU(),
        )
        self.p_z = torch.distributions.Beta(0.5, 0.5)
        self.epsilon = 1e-3

    def _forward(self, a, h):
        h = self.gcn0(a, h)
        h = self.gcn1(a, h)
        alpha, beta = self.layer_alpha(h).squeeze(-1), self.layer_beta(h).squeeze(-1)
        return alpha, beta

    def forward(self, a, h):
        alpha, beta = self._forward(a, h)
        alpha = alpha.unsqueeze(0) + alpha.unsqueeze(1)
        beta = beta.unsqueeze(0) + beta.unsqueeze(1)
        return torch.distributions.Beta(alpha, beta)

    def loss(self, a, h):
        alpha, beta = self._forward(a, h)
        src, dst = a.coalesce().indices()
        src, dst = src[src != dst], dst[src != dst]

        alpha1 = alpha[src] + alpha[dst]
        beta1 = beta[src] + beta[dst]

        _src, _dst = torch.randint(high=a.shape[0], size=src.shape), torch.randint(high=a.shape[0], size=src.shape)
        alpha0 = alpha[_src] + alpha[_dst]
        beta0 = beta[_src] + beta[_dst]

        beta1 = torch.distributions.Beta(alpha1, beta1)
        beta0 = torch.distributions.Beta(alpha0, beta0)

        mll = beta1.log_prob(torch.tensor(1.0 - self.epsilon)) + beta0.log_prob(torch.tensor(self.epsilon))
        return -mll.mean()

class GAE(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gcn0 = GCN(in_features, hidden_features, activation=torch.nn.ReLU())
        self.gcn1 = GCN(hidden_features, out_features)

    def encode(self, a, h):
        h = self.gcn0(a, h)
        h = self.gcn1(a, h)
        return h

    def forward(self, a, h):
        h = self.encode(a, h)
        a_hat = h @ h.t()
        return a_hat

    def loss(self, a, h):
        h = self.encode(a, h)
        a_hat = h @ h.t()
        a = a.to_dense()
        pos_weight = (a.shape[0] * a.shape[0] - a.sum()) / a.sum()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=a_hat,
            target=a,
            pos_weight=pos_weight,
        )
        return loss

class VGAE(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
    ):
        super().__init__()
        self.gcn0 = GCN(in_features, hidden_features, activation=torch.nn.ReLU())
        self.gcn_mu = GCN(hidden_features, out_features)
        self.gcn_log_sigma = GCN(hidden_features, out_features)
        self.p_z = torch.distributions.Normal(0, 1)

    def encode(self, a, h):
        h = self.gcn0(a, h)
        mu, log_sigma = self.gcn_mu(a, h), self.gcn_log_sigma(a, h)
        q_z = torch.distributions.Normal(mu, log_sigma.exp())
        return q_z

    def decode(self, q_z):
        z = q_z.rsample()
        z = z @ z.transpose(1, 0)
        p_a = torch.distributions.Bernoulli(logits=z)
        return p_a

    def forward(self, a, h):
        q_z = self.encode(a, h)
        p_a = self.decode(q_z)
        return p_a

    def loss(self, a, h):
        q_z = self.encode(a, h)
        p_a = self.decode(q_z)
        p_a = p_a.logits

        pos_weight = (a.shape[0] * a.shape[0] - _sum(a)) / _sum(a)
        scaling = a.shape[0] * a.shape[0] / float((a.shape[0] * a.shape[0] - _sum(a)) * 2)

        nll_loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
        )(
            p_a.flatten(),
            a.to_dense().flatten(),
        )

        kl_divergence = torch.distributions.kl_divergence(q_z, self.p_z).sum(-1).mean() / p_a.shape[0]
        loss = nll_loss + kl_divergence
        return loss
