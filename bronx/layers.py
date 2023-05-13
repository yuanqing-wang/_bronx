from typing import Optional, Callable
import torch
import dgl
from dgl.nn import DotGatConv, GraphConv
import torchsde

# class GraphConv(torch.nn.Module):
#     def __init__(
#             self,
#             in_features: int,
#             out_features: int,
#             **kwargs,
#     ):
#         super().__init__()
#         # self.W = torch.nn.Parameter(torch.randn(in_features, out_features))
#         # torch.nn.init.xavier_uniform_(self.W)

#     def forward(self, graph, x):
#         with graph.local_scope():
#             norm = torch.pow(graph.in_degrees().float().clamp(min=1), -1).unsqueeze(-1)
#             graph.ndata["x"] = x * norm
#             graph.update_all(dgl.function.copy_src(src="x", out="m"), dgl.function.sum(msg="m", out="x"))
#             x = graph.ndata["x"]
#             # x = x @ self.W.tanh()
#             return x

class BronxLayer(torchsde.SDEStratonovich):
    def __init__(self, hidden_features, num_heads=1, gamma=0.0):
        super().__init__(noise_type="diagonal")
        # self.gcn1 = GraphConv(hidden_features, hidden_features, bias=False)
        # self.gcn2 = GraphConv(hidden_features, hidden_features, bias=False)
        self.gcn = GraphConv(hidden_features, hidden_features, bias=False)

        self.fc_mu = torch.nn.Linear(2, hidden_features)
        self.fc_log_sigma = torch.nn.Linear(2, hidden_features)
        self.w = torch.nn.Parameter(torch.zeros(hidden_features, hidden_features))
        torch.nn.init.xavier_uniform_(self.w, gain=0.1)
        torch.nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.1)
        torch.nn.init.xavier_uniform_(self.fc_log_sigma.weight, gain=0.1)

        self.graph = None
        self.graph2 = None
        self.graph3 = None
        self.num_heads = num_heads
        self.gamma = gamma

    def ty(self, t, y):
        # y = torch.nn.functional.normalize(y, dim=-1)
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        ty = torch.cat([t.cos(), t.sin(), y], dim=-1)
        return ty

    def f(self, t, y):
        # ty = self.ty(t, y)
        # y = torch.nn.functional.normalize(y, dim=-1)
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        t = torch.cat([t.cos(), t.sin()], dim=-1)
        w = self.w - self.w.T
        # y1 = y @ w + self.gcn(self.graph, y) - self.gamma * y + self.fc_mu(t)
        y1 = torch.nn.functional.sigmoid(self.gcn(self.graph, y))
        y2 = torch.nn.functional.sigmoid(self.gcn(self.graph2, y))
        y = y @ w + y1 + y2 - self.gamma * y + self.fc_mu(t)
        return y.tanh()

    def g(self, t, y):
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        t = torch.cat([t.cos(), t.sin()], dim=-1)
        return torch.nn.functional.silu(self.fc_log_sigma(t))

    def h(self, t, y):
        return -y