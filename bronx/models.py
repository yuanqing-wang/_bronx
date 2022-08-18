import torch
from .layers import BronxLayer

class BronxModel(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        depth,
        **kwargs,
    ):
        super().__init__()
        self.embedding_in = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.layer = BronxLayer(hidden_features, **kwargs)
        self.embedding_out = torch.nn.Sequential(
            torch.nn.Linear(4 * depth + hidden_features*depth, hidden_features),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.log_identity_weights = torch.nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def pool_x(x):
        x = torch.cat(
            [
                x.diag().unsqueeze(-1),
                x.sum(-1, keepdims=True),
                x.std(-1, keepdims=True),
                x.max(-1, keepdims=True)[0],
            ],
            dim=-1
        ).swapaxes(0, -1).flatten(-2, -1)

    def forward(self, h, a):
        i = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
        x = a + i * self.log_identity_weights.exp()
        h = self.embedding_in(h)
        xs = []
        hs = []
        for _ in range(self.depth):
             h, x = self.layer(h, x)
             xs.append(x)
             hs.append(h)
        xs = torch.stack(xs, 0)
        hs = torch.stack(hs, -1).flatten(-2, -1)
        xs = self.pool_x(xs)
        h = self.embedding_out(torch.cat([hs, xs]))
        return h
