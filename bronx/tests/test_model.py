import torch
import dgl

def test_forward():
    g = dgl.rand_graph(5, 8)
    X = torch.zeros(5, 2)
    y = torch.zeros(5)
    iX = g.nodes()
    Xu = torch.zeros(4, 2)
    iXu = torch.tensor([0, 1, 2, 3])
    from bronx.kernels import CombinedGraphDiffusion
    from pyro.contrib.gp.kernels import RBF
    base_kernel = RBF(2)
    kernel = CombinedGraphDiffusion(2, base_kernel=base_kernel)
    from pyro.contrib.gp.likelihoods import Gaussian
    from bronx.models import GraphVariationalSparseGaussianProcess as GVSGP
    model = GVSGP(
        graph=g,
        X=X,
        y=y,
        iX=iX,
        Xu=Xu,
        iXu=iXu,
        kernel=kernel,
        likelihood=Gaussian(),
        jitter=1e-3,
    )
    model.model()
    model.guide()
    model(X, iX)
    

