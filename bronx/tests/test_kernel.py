from bronx.kernels import CombinedGraphDiffusion

def test_forward():
    import torch
    import dgl
    g = dgl.rand_graph(5, 8)
    g.ndata["x"] = torch.zeros(5, 2)
    from pyro.contrib.gp.kernels import RBF
    base_kernel = RBF(2)
    kernel = CombinedGraphDiffusion(2, base_kernel=base_kernel)
    k = kernel(g.ndata["x"], iX=g.nodes(), graph=g)

