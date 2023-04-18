from typing import Optional, Callable
import torch
import dgl
from dgl.nn import DotGatConv
import torchsde


class BronxLayer(torchsde.SDEIto):
    def __init__(self, hidden_features):
        super().__init__(noise_type="general")
        self.gcn = DotGatConv(
            hidden_features + 1, hidden_features, 1, # bias=False,
        )

        self.fc_log_sigma = torch.nn.Linear(
            hidden_features + 1, hidden_features, bias=False,
        )

        self.graph = None

    def f(self, t, y):
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        ty = torch.cat([t, y], dim=-1)
        return self.gcn(self.graph, ty).tanh().squeeze(-2) - y

    def g(self, t, y):
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        return torch.nn.functional.softplus(self.fc_log_sigma(torch.cat([t, y], dim=-1)).unsqueeze(-1))
