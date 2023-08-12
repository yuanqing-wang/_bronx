from functools import lru_cache
from typing import Callable
import torch
import dgl
import gpytorch
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
    NaturalVariationalDistribution,
    MeanFieldVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
)
from gpytorch.kernels import (
    ScaleKernel, RBFKernel, LinearKernel, CosineKernel, MaternKernel,
    PolynomialKernel,
    GridInterpolationKernel, SpectralMixtureKernel, GaussianSymmetrizedKLKernel
)
from .variational import GraphVariationalStrategy
from dgl.nn.functional import edge_softmax

class Rewire(torch.nn.Module):
    def __init__(self, in_features, out_features, t):
        super().__init__()
        self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, out_features, bias=False)
        self.register_buffer("t", torch.tensor(t))

    def forward(self, feat, graph):
        graph = graph.local_var()
        k = self.fc_k(feat)
        q = self.fc_q(feat)
        graph.ndata["k"] = k
        graph.ndata["q"] = q
        graph.apply_edges(dgl.function.u_dot_v("k", "q", "e"))
        e = edge_softmax(graph, graph.edata["e"]).squeeze(-1)
        a = torch.zeros(
            graph.number_of_nodes(),
            graph.number_of_nodes(),
            dtype=torch.float32,
            device=graph.device,
        )
        src, dst = graph.edges()
        a[src, dst] = e
        a = a - torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
        a = a * self.t.exp()
        a = torch.linalg.matrix_exp(a)
        return a

@lru_cache(maxsize=1)
def graph_exp(graph):
    a = torch.zeros(
        graph.number_of_nodes(),
        graph.number_of_nodes(),
        dtype=torch.float32,
        device=graph.device,
    )
    src, dst = graph.edges()
    a[src, dst] = 1
    d = a.sum(-1, keepdims=True).clamp(min=1)
    a = a / d
    a = a - torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    a = torch.linalg.matrix_exp(a)
    return a

class ExactBronxModel(ExactGP):
    def __init__(
            self, train_x, train_y, likelihood, num_classes, 
            features, graph, in_features, hidden_features, t, log_sigma,
            activation,
        ):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size((num_classes,)),
        )

        self.covar_module = ScaleKernel(GaussianSymmetrizedKLKernel())
        self.rewire = Rewire(
            hidden_features, hidden_features, t=t,
        )
        self.likelihood = likelihood
        self.num_classes = num_classes
        self.register_buffer("features", features)
        self.graph = graph
        self.fc = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.norm = torch.nn.LayerNorm(hidden_features)
        self.log_sigma = torch.nn.Parameter(torch.tensor(log_sigma))
        self.activation = activation

    def forward(self, x):
        h = self.fc(self.features)
        h = self.norm(h)
        h = self.activation(h)
        a = self.rewire(h, self.graph)
        mean = self.mean_module(h)
        covar = self.covar_module(h)
        covar = a @ covar @ a.t()
        covar = covar \
        + self.log_sigma.exp() \
        * torch.eye(covar.shape[-1], dtype=covar.dtype, device=covar.device)
        x = x.squeeze().long()
        mean = mean[..., x]
        covar = covar[..., x, :][..., :, x]
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class ApproximateBronxModel(ApproximateGP):
    def __init__(
            self,
            features,
            inducing_points,
            graph: dgl.DGLGraph,
            in_features: int,
            hidden_features: int,
            num_classes: int,
            learn_inducing_locations: bool = False,
            t: float=1.0,
            log_sigma: float=0.0,
            activation: Callable=torch.nn.functional.silu,
    ):

        batch_shape = torch.Size([num_classes])
        variational_distribution = NaturalVariationalDistribution(
            inducing_points.size(-1),
            batch_shape=batch_shape,
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        
        variational_strategy = IndependentMultitaskVariationalStrategy(
            variational_strategy, 
            num_tasks=num_classes,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size((num_classes,)),
        )

        self.covar_module = LinearKernel()
        self.rewire = Rewire(
            hidden_features, hidden_features, t=t,
        )
        self.num_classes = num_classes
        self.register_buffer("features", features)
        self.graph = graph
        self.fc = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.norm = torch.nn.LayerNorm(hidden_features)
        self.register_buffer("log_sigma", torch.tensor(log_sigma))
        self.activation = activation

    def forward(self, x):
        h = self.fc(self.features)
        h = self.norm(h)
        h = self.activation(h)
        a = self.rewire(h, self.graph)
        h = a @ h
        mean = self.mean_module(h)
        covar = self.covar_module(h)
        covar = covar \
        + self.log_sigma.exp() \
        * torch.eye(covar.shape[-1], dtype=covar.dtype, device=covar.device)
        x = x.squeeze().long()
        mean = mean[..., x]
        covar = covar[..., x, :][..., :, x]
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    