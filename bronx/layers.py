from typing import Optional, Callable
import torch
import dgl
from dgl.nn import DotGatConv, GraphConv
import torchsde

class BronxLayer(torchsde.SDEStratonovich):
    def __init__(self, hidden_features, num_heads=1, gamma=0.0):
        super().__init__(noise_type="general")
        # self.gcn = DotGatConv(hidden_features + 2, int(hidden_features / num_heads), num_heads)
        # self.gcn1 = DotGatConv(hidden_features, int(hidden_features / num_heads), num_heads)
        self.gcn = GraphConv(hidden_features + 2, hidden_features, bias=False)
        # self.gcn1 = GraphConv(hidden_features + 2, hidden_features, bias=False)

        self.fc_log_sigma = torch.nn.Linear(
            hidden_features + 2, hidden_features, bias=False,
        )

        self.graph = None
        self.num_heads = num_heads
        self.gamma = gamma

    def ty(self, t, y):
        # y = torch.nn.functional.normalize(y, dim=-2)
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        ty = torch.cat([t.cos(), t.sin(), y], dim=-1)
        return ty

    def f(self, t, y):
        ty = self.ty(t, y)
        return (self.gcn(self.graph, ty) - self.gamma * y).tanh()

    def g(self, t, y):
        ty = self.ty(t, y)
        return self.fc_log_sigma(ty).tanh().unsqueeze(-1)

    def h(self, t, y):
        return -y