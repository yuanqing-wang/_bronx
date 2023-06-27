import torch
import dgl


def test_dimension():
    from bronx.models import BronxModel

    g = dgl.rand_graph(5, 8)
    h0 = torch.zeros(5, 16)
    layer = BronxModel(16, 32, 32, 2)
    h = layer.model(g, h0)
    layer.guide(g, h0)
