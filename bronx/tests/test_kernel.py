
def test_forward():
    from bronx.kernels import CombinedGraphDiffusion
    import gpytorch
    gpytorch.settings.debug._default = False
    gpytorch.settings.lazily_evaluate_kernels._default = False
    import torch
    import dgl
    g = dgl.rand_graph(5, 8)
    x1 = torch.zeros(5, 2)
    x2 = torch.zeros(5, 2)
    ix1 = torch.tensor([0, 1, 2])
    ix2 = torch.tensor([0, 1])
    from gpytorch.kernels import ScaleKernel, RBFKernel
    
    base_kernel = ScaleKernel(RBFKernel(ard_num_dims=2))
    kernel = CombinedGraphDiffusion(base_kernel=base_kernel)
    covar = kernel(x1=x1, x2=x2, graph=g, ix1=ix1, ix2=ix2)
    covar = covar.evaluate()

test_forward()