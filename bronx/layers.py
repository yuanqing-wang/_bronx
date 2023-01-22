from typing import Optional, Callable
import torch
import dgl
from dgl.nn import GraphConv
import torchsde
from dgl.nn.functional import edge_softmax

class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, num_heads=1):
        super().__init__()
        if out_features is None: out_features = in_features
        if hidden_features is None: hidden_features = out_features
        self.fc_k = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc = torch.nn.Linear(in_features, out_features)
        self.num_heads = num_heads

    def forward(self, g, h):
        g = g.local_var()
        k = self.fc_k(h).reshape(*h.shape[:-1], self.num_heads, -1)
        q = self.fc_q(h).reshape(*h.shape[:-1], self.num_heads, -1)
        g.ndata["k"] = k
        g.ndata["q"] = q
        g.apply_edges(dgl.function.u_dot_v("k", "q", "e"))
        g.edata["a"] = edge_softmax(g, g.edata["e"])
        g.ndata["h"] = h
        g.update_all(dgl.function.u_mul_e("h", "a", "m"), dgl.function.sum("m", "h"))
        h = self.fc(g.ndata["h"]).flatten(-2, -1)
        return h
