import torch
import dgl

def test_dimension():
    from bronx.layers import BronxLayer
    g = dgl.rand_graph(5, 8)
    h0 = torch.zeros(5, 16)
    layer = BronxLayer(16, 32)
    h = layer.model(g, h0)
    layer.guide(g, h0)


    