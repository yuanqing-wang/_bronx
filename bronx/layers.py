from typing import Optional, Callable
import torch
import dgl
from dgl.nn import DotGatConv
import torchsde

class BronxLayer(torchsde.SDEIto):
    def __init__(self, hidden_features, num_heads=1):
        super().__init__(noise_type="general")
        self.gcn = DotGatConv(
            hidden_features + 2, int(hidden_features / num_heads), num_heads, # bias=False,
        )

        self.fc_log_sigma = torch.nn.Linear(
            hidden_features + 2, hidden_features, bias=False,
        )

        # self.norm = torch.nn.LayerNorm(hidden_features+2)
        # self.norm = torch.nn.InstanceNorm1d(hidden_features+2)
        self.graph = None
        self.num_heads = num_heads

    def ty(self, t, y):
        y = torch.nn.functional.normalize(y, dim=-1)
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        ty = torch.cat([t.cos(), t.sin(), y], dim=-1)
        return ty

    def f(self, t, y):
        ty = self.ty(t, y)
        return self.gcn(self.graph, ty).flatten(-2, -1).tanh()

    def g(self, t, y):
        ty = self.ty(t, y)
        return self.fc_log_sigma(ty).unsqueeze(-1)

    def h(self, t, y):
        return -y
