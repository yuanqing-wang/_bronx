from typing import Optional, Callable
import torch

class BronxLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_features=None,
        num_heads=1,
        activation=torch.nn.ELU(),
    ):
        super().__init__()
        self.fc_k = torch.nn.Linear(hidden_features, hidden_features, bias=False)
        self.fc_q = torch.nn.Linear(hidden_features, hidden_features, bias=False)
        self.fc = torch.nn.Linear(hidden_features, hidden_features*num_heads, bias=False)
        self.norm = torch.nn.LayerNorm(hidden_features)
        self.activation = activation

    def forward(self, h, x):
        h = self.norm(h)
        x = torch.nn.functional.normalize(x, p=1, dim=-1)
        x = torch.stack([torch.matrix_power(x.swapaxes(0, -1), idx) for idx in range(self.num_heads-1)], -1).swapaxes(0, -1)

        k = self.fc_k(h)
        q = self.fc_q(h)

        # (n, d, n_heads)
        k = k.reshape(k.shape[0], int(self.hidden_features / self.num_heads), self.num_heads)
        q = q.reshape(q.shape[0], int(self.hidden_features / self.num_heads), self.num_heads)

        # (n, n, n_heads)
        a = torch.einsum("xyb,zyb->xzb", k, q)
        a = a.softmax(-2)

        a = (a * x).softmax(-2)
        h = torch.einsum("xyb,yzb->xzb", a, h)
        h = h.flatten(-2, -1)

        h = self.fc(h)
        h = self.activation(h)
        return h, x
