import torch
import dgl

def test_forward():
    import torch
    import dgl
    g = dgl.rand_graph(5, 8)
    x = torch.zeros(5, 2)
    from bronx.models import BronxModel
    model = BronxModel(
        inducing_points=x,
        inducing_indices=g.nodes(),
        graph=g,
    )
    y = model(x)

