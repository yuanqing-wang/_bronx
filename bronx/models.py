import torch
import dgl
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel
from .kernels import CombinedGraphDiffusion
from .variational import GraphVariationalStrategy

class BronxModel(ApproximateGP):
    def __init__(
            self,
            inducing_points: torch.Tensor,
            inducing_indices: torch.Tensor,
            graph: dgl.DGLGraph,
            hidden_features: int = 32,
            embedding_features: int = 32,
            learn_inducing_locations: bool = False,
    ):

        batch_shape = torch.Size([hidden_features])
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=batch_shape,
        )
        
        variational_strategy = GraphVariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            graph=graph,
            inducing_indices=inducing_indices,
            learn_inducing_locations=learn_inducing_locations,
        )

        # variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
        #     variational_strategy,
        #     num_tasks=batch_shape[0],
        # )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.LinearMean(embedding_features)
        base_kernel = ScaleKernel(
            LinearKernel(),
        )
        self.covar_module = CombinedGraphDiffusion(base_kernel=base_kernel)
        self.fc = torch.nn.Linear(inducing_points.size(-1), embedding_features, bias=False)
        torch.nn.init.normal_(self.fc.weight, std=0.01)
        
    def forward(self, x, x2=None, **kwargs):
        if x2 is None:
            x2 = x
        x = self.fc(x).tanh()
        x2 = self.fc(x2).tanh()
        ix1 = kwargs.get("ix1", torch.arange(x.size(-2), device=x.device))
        mean = self.mean_module(x)[ix1]
        covar = self.covar_module(x1=x, x2=x2, **kwargs)
        result = gpytorch.distributions.MultivariateNormal(mean, covar)
        return result
    
    def to(self, device):
        self = super().to(device)
        # self.variational_strategy.base_variational_strategy.inducing_indices = self.variational_strategy.base_variational_strategy.inducing_indices.to(device)
        # self.variational_strategy.base_variational_strategy.graph = self.variational_strategy.base_variational_strategy.graph.to(device)

        self.variational_strategy.inducing_points = self.variational_strategy.inducing_points.to(device)
        self.variational_strategy.graph = self.variational_strategy.graph.to(device)
        return self
    