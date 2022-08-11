import torch
from .layers import BronxLayer

class BronxModel(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        depth,
        residual=True,
    ):
        super().__init__()
        self.embedding_in = torch.nn.Linear(in_features, hidden_features)
        layers = []
        for _ in range(depth):
            layers.append(
                BronxLayer(hidden_features),
            )
        self.layers = torch.nn.ModuleList(layers)
        self.embedding_out = torch.nn.Linear(hidden_features, out_features)
        self.log_identity_weights = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, h, a):
        i = torch.eye(a.shape[-1], device=a.device, dtype=a.device)
        x = a + i * self.log_identity_weights.exp()
        h = self.embedding_in(h)
        for layer in self.layers:
            h, x = self.layer(h, x)
        h = self.embedding_out(h)
        return h
