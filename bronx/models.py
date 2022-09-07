import torch
from .layers import BronxLayer

class BronxModel(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        depth,
        num_heads,
        **kwargs,
    ):
        super().__init__()
        self.embedding_in = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.num_heads = num_heads

        layers = []

        '''
        for _ in range(depth):
            layers.append(
                BronxLayer(hidden_features, num_heads=num_heads, **kwargs),
            )
        self.layers = torch.nn.ModuleList(layers)
        '''

        self.layer = BronxLayer(hidden_features, num_heads=num_heads, **kwargs)
        self.layers = [self.layer for _ in range(depth)]
        self.embedding_out = torch.nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, h, x):
        with torch.no_grad():
            x = torch.nn.functional.normalize(x, p=1, dim=-1)
            x = torch.stack([torch.matrix_power(x, idx+1) for idx in range(self.num_heads)], -1)
            x = torch.nn.functional.normalize(x, p=1, dim=-2)
        h = self.embedding_in(h)
        for layer in self.layers:
             h, x = layer(h, x)
        h = self.embedding_out(h)
        return h
