import torch
import dgl


def test_multihead():
    g = dgl.rand_graph(3, 5)
    h = torch.rand(3, 10)
    e = torch.rand(5, 1)
    from bronx.layers import linear_diffusion

    a = linear_diffusion(g, h, e=e)
    print(a.shape)
