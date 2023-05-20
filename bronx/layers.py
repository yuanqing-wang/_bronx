from typing import Optional, Callable
import torch
import dgl
from dgl.nn import DotGatConv, GraphConv, GATConv
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


class _GraphConv(GraphConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_feats, out_feats = self.weight.shape
        del self.weight
        self.w = torch.nn.Parameter(torch.eye(in_feats))
        self.d = torch.nn.Parameter(torch.zeros(in_feats))

    @property
    def weight(self):
        d = self.d.sigmoid()
        w = torch.mm(self.w * d, self.w.T)
        return w

class BronxLayer(torchsde.SDEStratonovich):
    def __init__(self, hidden_features, num_heads=1, gamma=0.0, gain=0.0):
        super().__init__(noise_type="scalar")
        # self.gcn = GraphConv(hidden_features, hidden_features // 4)
        # self.gcn2 = _GraphConv(hidden_features, hidden_features)
        self.gcn = GraphConv(hidden_features, hidden_features // 2)
        self.fc_mu = torch.nn.Linear(2, hidden_features)
        self.fc_log_sigma = torch.nn.Linear(2, hidden_features)
        self.w = torch.nn.Parameter(torch.zeros(hidden_features, hidden_features))
        torch.nn.init.xavier_uniform_(self.w, gain=gain)
        
        self.graph = None
        self.graph2 = None
        # self.graph3 = None
        self.num_heads = num_heads
        self.gamma = gamma


    def ty(self, t, y):
        # y = torch.nn.functional.normalize(y, dim=-1)
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        ty = torch.cat([t.cos(), t.sin(), y], dim=-1)
        return ty

    def f(self, t, y):
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        t = torch.cat([t.cos(), t.sin()], dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
        mu = self.fc_mu(t).sigmoid()
        w = self.w - self.w.T
        y1 = self.gcn(self.graph, y)# .tanh()
        y2 = self.gcn(self.graph2, y)# .tanh()

        if y1.dim() > 2:
            y1 = y1.flatten(-2, -1)
            y2 = y2.flatten(-2, -1)

        y12 = torch.cat([y1, y2], dim=-1)
        # y12 = torch.nn.functional.silu(y12)
        y = y @ w + y12 - self.gamma * y
        return torch.nn.functional.tanh(y) * mu

    def g(self, t, y):
        t = torch.broadcast_to(t, (*y.shape[:-1], 1))
        t = torch.cat([t.cos(), t.sin()], dim=-1)
        return self.fc_log_sigma(t).unsqueeze(-1)

    def h(self, t, y):
        return -y
        
