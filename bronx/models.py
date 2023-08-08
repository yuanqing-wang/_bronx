from functools import lru_cache
import torch
import dgl
import gpytorch
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
)
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel, CosineKernel, MaternKernel
from .variational import GraphVariationalStrategy

class ExactBronxModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, x):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size((num_classes,)),
        )
        self.covar_module = ScaleKernel(
            RBFKernel(),
            batch_shape=torch.Size((num_classes,)),
        )
        self.likelihood = likelihood
        self.num_classes = num_classes
        self.register_buffer("x", x)

    def forward(self, x):
        x = self.x[x.squeeze().long()]
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


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

        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        
        # variational_strategy = GraphVariationalStrategy(
        #     self,
        #     inducing_points=inducing_points,
        #     variational_distribution=variational_distribution,
        #     graph=graph,
        #     inducing_indices=inducing_indices,
        #     learn_inducing_locations=learn_inducing_locations,
        # )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())
        # self.fc = torch.nn.Linear(inducing_points.size(-1), embedding_features, bias=False)
        # torch.nn.init.normal_(self.fc.weight, std=1.0)
        
    def forward(self, x):
        # x = self.fc(x)# .tanh()
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        result = gpytorch.distributions.MultivariateNormal(mean, covar)
        return result
    
    # def to(self, device):
    #     self = super().to(device)
    #     self.variational_strategy.inducing_points = self.variational_strategy.inducing_points.to(device)
    #     self.variational_strategy.graph = self.variational_strategy.graph.to(device)
    #     return self
    