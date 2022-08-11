from typing import Optional, Callable
import torch

class BronxLayer(torch.nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
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
        self.fc_k = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_v = torch.nn.Lienar(in_features, out_features, bias=False)

        # mixing
        mixing = torch.zeros(2, 2)
        mixing[0, 0] = semantic_weight
        mixing[0, 1] = semantic_weight

        self.hidden_features = hidden_features
        self.activation = activation
        self.norm = torch.nn.LayerNorm(in_features)
        self.residual = residual

    def forward(self, h, x):
        h0, x0 = h, x
        h = self.norm(h)
        k = self.fc_k(h)
        q = self.fc_q(h)
        a_h = (k @ q.t() * self.hidden_features ** (-0.5)).softmax(-1)
        a_x = x @ x.t()
        a_x = torch.nn.functional.normalize(a_x, p=2, dim=-1)
        a = torch.stack([a_h, a_x], dim=-1)
        a = a @ self.mixing.softmax(0)
        a_h, a_x = a[..., 0], a[..., 1]
        h = a_h @ h
        x = a_x @ x
        h = self.fc_v(h)
        h = self.activation(h)
        if self.residual:
            h = h + h0
            x = x + x0
        return h, x
