from typing import Optional, Callable
import torch

def _sum(x, *args, **kwargs):
    if x.is_sparse:
        return torch.sparse.sum(x, *args, **kwargs).values()
    else:
        return torch.sum(x, *args, **kwargs)

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
        d = _sum(a, dim=-1).float().clamp(min=1)
        norm = d.pow(-0.5).unsqueeze(-1)
        h = h * norm
        h = self.fc(h)
        h = a @ h
        h = h * norm
        if self.activation is not None:
            h = self.activation(h)
        return h

class GraphAutoEncoder(torch.nn.Module):
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
        a_hat = self(a, h)
        a = a.to_dense()
        pos_weight = (a.shape[0] * a.shape[0] - a.sum()) / a.sum()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=a_hat,
            target=a,
            pos_weight=pos_weight,
        )
        return loss

class VariationalGraphAutoEncoder(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gcn0 = GCN(in_features, hidden_features, activation=torch.nn.ReLU())
        self.gcn_mu = GCN(hidden_features, out_features)
        self.gcn_log_sigma = GCN(hidden_features, out_features)
        self.p_z = torch.distributions.Normal(0, 1)

    def encode(self, a, h):
        h = self.gcn0(a, h)
        mu, sigma = self.gcn_mu(a, h), self.gcn_log_sigma(a, h).exp()
        p_z = torch.distributions.Normal(mu, sigma)
        return p_z

    def decode(self, q_z, n_samples=1):
        z = q_z.rsample((n_samples, ))
        a_hat = torch.bmm(z, z.transpose(-2, -1))
        return a_hat

    def forward(self, a, h, n_samples=1):
        return self.decode(self.encode(a, h), n_samples=n_samples)

    def loss(self, a, h):
        q_z = self.encode(a, h)
        a_hat = self.decode(q_z).mean(dim=0)

        a = a.to_dense()
        pos_weight = (a.shape[0] * a.shape[0] - a.sum()) / a.sum()
        norm = a.shape[0] * a.shape[0] / ((a.shape[0] * a.shape[0] - _sum(a)) * 2)
        ll = torch.nn.functional.binary_cross_entropy_with_logits(
            input=a_hat,
            target=a,
            pos_weight=pos_weight,
        )

        kl_divergence = torch.distributions.kl_divergence(q_z, self.p_z).sum(-1).mean()

        loss = norm * ll + kl_divergence / a.shape[0]

        return loss

class SharedVariationalGraphAutoEncoder(VariationalGraphAutoEncoder):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__(in_features, hidden_features, out_features)
        del self.gcn_mu
        del self.gcn_log_sigma
        self.gcn0 = GCN(in_features, hidden_features, activation=torch.nn.ReLU())
        self.gcn1 = GCN(hidden_features, hidden_features)
        self.fc_mu = torch.nn.Linear(hidden_features, out_features, bias=False)
        self.fc_log_sigma = torch.nn.Linear(hidden_features, out_features, bias=False)
        self.p_z = torch.distributions.Normal(0, 1)

    def _forward(self, a, h):
        h = self.gcn0(a, h)
        h = self.gcn1(a, h)
        return h

    def encode(self, a, h):
        h = self._forward(a, h)
        mu, sigma = self.fc_mu(h), self.fc_log_sigma(h).exp()
        p_z = torch.distributions.Normal(mu, sigma)
        return p_z

class Bronx(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, neighborhood_size=3):
        super().__init__()
        self.vgae = SharedVariationalGraphAutoEncoder(in_features, hidden_features, hidden_features)
        self.fc = torch.nn.Linear(hidden_features, out_features, bias=False)
        self.a_candidate = None
        self.neighborhood_size = neighborhood_size

    def reconstruct(self, a, h, n_samples=1):
        a_hat = self.vgae(a, h, n_samples=n_samples)
        return a_hat

    def candidate(self, a, k=None):
        if self.a_candidate is None:
            if k is None:
                k = self.neighborhood_size
            self.a_ref = a
            for _ in range(k - 1):
                a = a @ a
            self.a_candidate = a.to_dense().clamp(0, 1)
        return self.a_candidate

    def sparsify(self, a):
        k = self.a_ref._nnz() - self.a_ref.shape[0]
        a = a - torch.eye(a.shape[0])
        a_flatten = a.flatten(-2, -1)
        topk_value, _ = torch.topk(a_flatten, k, -1)
        threshold = topk_value.min(dim=-1)[0]
        a = torch.where(a > threshold, a, torch.zeros_like(a))
        a = a + torch.eye(a.shape[0])
        return a

    def forward(self, a, h, n_samples=1):
        a_candidate = self.candidate(a)
        a_hat = self.reconstruct(a, h, n_samples=n_samples).sigmoid()
        a_hat = a_hat * a_candidate
        h = self.vgae._forward(a_hat, h)
        y_hat = self.fc(h).mean(dim=0)
        return y_hat

    def loss_vae(self, a, h):
        loss_vae = self.vgae.loss(a, h)
        return loss_vae
