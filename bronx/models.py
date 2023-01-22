import torch
from .layers import BronxLayer, GraphAttentionLayer
import torchsde


class BronxModel(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, depth, num_heads=1):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_out = torch.nn.Linear(hidden_features, out_features, bias=False)
        layers = [GraphAttentionLayer(in_features=hidden_features, out_features=hidden_features)]
        for idx in range(depth-1):
            layers.append(GraphAttentionLayer(in_features=hidden_features*num_heads, out_features=hidden_features))
        self.layers = torch.nn.ModuleList(layers)
    def forward(self, g, h):
        g = g.local_var()
        h = self.fc_in(h)
        for layer in self.layers:
            h = layer(g, h)
        h = self.fc_out(h)
        return h
