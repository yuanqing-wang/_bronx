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
        self.fc_k = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_v = torch.nn.Linear(in_features+3, hidden_features, bias=False)
        self.fc = torch.nn.Linear(hidden_features, out_features)

        # mixing
        mixing = torch.ones(num_heads + 1, num_heads +1) * semantic_weight
        mixing[0] = 0.0
        # self.mixing = torch.nn.Parameter(mixing)
        self.register_buffer("mixing", mixing)
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

        # (n, d, n_heads)
        k = k.reshape(k.shape[0], int(self.hidden_features / self.num_heads), self.num_heads)
        q = q.reshape(q.shape[0], int(self.hidden_features / self.num_heads), self.num_heads)

        # (n_heads, n, n)
        a_h = torch.einsum("xyb,zyb->xzb", k, q).softmax(-2)
        a_x = x @ x.t()

        i = torch.cat(
            [
                a_x.diag().unsqueeze(-1),
                a_x.sum(-1, keepdims=True),
                a_x.std(-1, keepdims=True),
            ],
            dim=-1,
        )
        
        h = torch.cat([h, i], dim=-1)
        v = self.fc_v(h)
        v = v.reshape(v.shape[0], int(self.hidden_features / self.num_heads), self.num_heads)

        a = torch.cat([a_x.unsqueeze(-1), a_h], dim=-1)
        a = a @ self.mixing.softmax(0)
        a_x, a_h = a[..., 0], a[..., 1:]

        h = torch.einsum("xyb,yzb->xzb", a_h, v).reshape(v.shape[0], self.hidden_features)
        x = a_x @ x
        h = self.fc(h)
        h = self.activation(h)

        if self.residual:
            h = h + h0
            x = x + x0
        return h, x
