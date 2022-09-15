from typing import Optional, Callable
import torch
import dgl
from dgl.nn import GraphConv
import torchsde

class BronxLayer(torchsde.SDEIto):
    def __init__(self, hidden_features):
        super().__init__(noise_type="general")
        self.gcn = GraphConv(hidden_features + 1, hidden_features)
        self.graph = None

    def f(self, t, y):
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        return self.gcn(
            self.graph,
            torch.cat([t, y], dim=-1),
        ) - y

    def g(self, t, y):
        return 1e-2 * torch.ones_like(y).unsqueeze(-1)
