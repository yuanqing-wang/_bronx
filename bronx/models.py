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
        self.embedding_in = torch.nn.Linear(in_features, hidden_features)
        layers = []
        # layers.append(BronxLayer(in_features, hidden_features, hidden_features, **kwargs))
        for _ in range(depth):
            layers.append(
                BronxLayer(hidden_features, **kwargs),
            )
        # layers.append(BronxLayer(hidden_features, hidden_features, out_features, last=True, **kwargs))
        self.layers = torch.nn.ModuleList(layers)
        self.embedding_out = torch.nn.Linear(hidden_features, out_features)
        self.log_identity_weights = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, h, a):
        i = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
        x = a + i * self.log_identity_weights.exp()
        h = self.embedding_in(h)
        for layer in self.layers:
            h, x = layer(h, x)
        h = self.embedding_out(h)
        return h
