from typing import Optional, Callable
import torch

class BronxLayer(torch.nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            num_heads=1,
            semantic_weight=-1.0,
            activation=torch.nn.ELU(),
            residual=True,
        ):
        super().__init__()
        if hidden_features is None:
            hidden_features = in_features
        if out_features is None:
            out_features = hidden_features

        # attention weights
        self.fc_k = torch.nn.Linear(in_features, hidden_features * num_heads, bias=False)
        self.fc_q = torch.nn.Linear(in_features, hidden_features * num_heads, bias=False)
        self.fc_v = torch.nn.Linear(in_features+3, out_features, bias=False)

        # mixing
        mixing = torch.ones(num_heads + 1, num_heads +1) * semantic_weight
        mixing[0] = 0.0
        self.mixing = torch.nn.Parameter(mixing)
        self.hidden_features = hidden_features
        self.activation = activation
        self.norm = torch.nn.LayerNorm(in_features)
        self.residual = residual
        self.num_heads = num_heads

    def forward(self, h, x):
        h0, x0 = h, x
        h = self.norm(h)
        x = torch.nn.functional.normalize(x, p=1, dim=-1)
        k = self.fc_k(h)
        q = self.fc_q(h)
        k = k.reshape(k.shape[0], int(self.hidden_features / self.num_heads), self.num_heads)
        q = q.reshape(q.shape[0], int(self.hidden_features / self.num_heads), self.num_heads)

        # (n_heads, n, n)
        a_h = (torch.bmm(k.permute(2, 0, 1), q.permute(2, 1, 0)) * self.hidden_features ** (-0.5)).softmax(-1)
        a_h = a_h.permute(1, 2, 0)
        a_x = x @ x.t()

        i = torch.cat(
            [
                a_x.diag().unsqueeze(-1),
                a_x.sum(-1, keepdims=True),
                a_x.std(-1, keepdims=True),
            ],
            dim=-1,
        )

        a = torch.cat([a_x.unsqueeze(-1), a_h], dim=-1)
        a = a @ self.mixing.softmax(0)
        a_x, a_h = a[..., 0], a[..., 1:]
        a_h = a_h.transpose(-1, 0)
        h = a_h @ h
        x = a_x @ x
        h = h.reshape(h.shape[1], self.hidden_features)
        h = torch.cat([h, i], dim=-1)
        h = self.fc_v(h)
        h = self.activation(h)
        if self.residual:
            h = h + h0
            x = x + x0
        return h, x
