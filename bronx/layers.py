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
            epsilon=1.0,
            activation=torch.nn.ELU(),
            residual=True,
            a_h_drop=0.0,
            a_x_drop=0.0,
            fc_drop=0.0,
            last=False,
        ):
        super().__init__()
        if hidden_features is None:
            hidden_features = in_features
        if out_features is None:
            out_features = hidden_features
        if last is True:
            out_features = out_features * num_heads

        # attention weights
        self.fc_k = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_v = torch.nn.Linear(in_features, hidden_features, bias=False)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, out_features),
            torch.nn.Dropout(fc_drop),
            activation,
        )

        # mixing
        mixing = torch.ones(num_heads + 1, num_heads +1) * semantic_weight
        mixing[0] = 0.0
        mixing = mixing + torch.randn_like(mixing) * epsilon
        self.mixing = torch.nn.Parameter(mixing)
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.activation = activation
        self.norm = torch.nn.LayerNorm(in_features)
        self.norm_fc = torch.nn.LayerNorm(hidden_features)
        self.residual = residual
        self.num_heads = num_heads
        self.a_h_drop = torch.nn.Dropout(a_h_drop)
        self.a_x_drop = torch.nn.Dropout(a_x_drop)
        self.last = last

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
        a_h = torch.einsum("xyb,zyb->xzb", k, q)
        a_h = a_h.softmax(-2)
        a_h = self.a_h_drop(a_h)
        a_x = x @ x.t()

        '''
        i = torch.cat(
            [
                a_x.diag().unsqueeze(-1),
                a_x.sum(-1, keepdims=True),
                a_x.std(-1, keepdims=True),
                a_x.max(-1, keepdims=True)[0],
            ],
            dim=-1,
        )
        '''

        a_x = self.a_x_drop(a_x)
        v = self.fc_v(h)
        v = v.reshape(v.shape[0], int(self.out_features / self.num_heads), self.num_heads)

        a = torch.cat([a_x.unsqueeze(-1), a_h], dim=-1)
        a = a @ self.mixing.softmax(0)
        a = torch.nn.functional.normalize(a, p=1, dim=-2)
        a_x, a_h = a[..., 0], a[..., 1:]
        h = torch.einsum("xyb,yzb->xzb", a_h, v)
        h = h.reshape(v.shape[0], self.hidden_features)

        x = a_x @ x

        if self.residual:
            h = h + h0
            x = x + x0

        _h0 = h
        h = self.norm_fc(h)
        # h = torch.cat([h, i], dim=-1)
        h = self.fc(h)

        if self.residual:
            h = h + _h0

        return h, x
