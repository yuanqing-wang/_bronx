import torch
from .layers import BronxLayer

class BronxModel(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        depth,
        adjacency_matrix,
        diffusion_matrix,
        **kwargs,
    ):
        super().__init__()
        self.embedding_in = torch.nn.Linear(in_features, hidden_features, bias=False)
        layers = []

        for _ in range(depth):
            layers.append(
                BronxLayer(
                    hidden_features,
                    adjacency_matrix=adjacency_matrix,
                    diffusion_matrix=diffusion_matrix,
                    **kwargs
                ),
            )
        self.layers = torch.nn.ModuleList(layers)
        '''

        self.layer = BronxLayer(hidden_features, adjacency_matrix=adjacency_matrix, diffusion_matrix=diffusion_matrix)
        self.layers = [self.layer for _ in range(depth)]
        '''

        self.embedding_out = torch.nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, h):
        kl = 0.0
        h = self.embedding_in(h)
        for layer in self.layers:
             h = layer(h)
             kl += layer.kl
        h = self.embedding_out(h)
        self.kl = kl
        return h
