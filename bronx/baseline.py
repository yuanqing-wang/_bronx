import torch
import dgl
from dgl.nn.pytorch import GraphConv

class DropEdge(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        p=0.5,
        activation=torch.nn.SiLU(),
        depth=1,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for idx in range(depth):
            _in_features = hidden_features
            _out_features = hidden_features
            if idx == 0:
                _in_features = in_features
            if idx == depth-1:
                _out_features = out_features
            layer = GraphConv(
                _in_features, _out_features, activation=activation,
            )
            self.layers.append(layer)
        self.p = p
        self.activation = activation


    def forward(self, graph, feat):
        weight = torch.distributions.Bernoulli(
            torch.ones(graph.number_of_edges()) * (1-self.p)
        ).sample().to(feat.device)

        for layer in self.layers[:-1]:
            feat = layer(graph, feat, edge_weight=weight)
            feat = self.activation(feat)
        feat = self.layers[-1](graph, feat, edge_weight=weight)

        return feat



