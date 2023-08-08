from functools import lru_cache
import torch
import dgl
import gpytorch
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
)
from gpytorch.kernels import (
    ScaleKernel, RBFKernel, LinearKernel, CosineKernel, MaternKernel,
    GridInterpolationKernel,
)
from .variational import GraphVariationalStrategy
from dgl.nn.functional import edge_softmax

class Rewire(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, out_features, bias=False)
        self.t = torch.nn.Parameter(torch.tensor(0.0))

        torch.nn.init.constant_(self.fc_k.weight, 1e-3)
        torch.nn.init.constant_(self.fc_q.weight, 1e-3)

    def forward(self, feat, graph):
        graph = graph.local_var()
        k = self.fc_k(feat)
        q = self.fc_q(feat)
        graph.ndata["k"] = k
        graph.ndata["q"] = q
        graph.apply_edges(dgl.function.u_dot_v("k", "q", "e"))
        e = graph.edata["e"]
        e = e / k.shape[-1] ** 0.5
        e = edge_softmax(graph, e).squeeze(-1)
        a = torch.zeros(
            graph.number_of_nodes(),
            graph.number_of_nodes(),
            dtype=torch.float32,
            device=graph.device,
        )
        src, dst = graph.edges()
        a[src, dst] = e
        a = a - torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
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
            features, graph, in_features, hidden_features,
        ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size((num_classes,)),
        )

        self.covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=hidden_features, 
                batch_shape=torch.Size((num_classes,))
            ),
            batch_shape=torch.Size([num_classes]),
        )

        self.rewire = Rewire(hidden_features, hidden_features)
        self.likelihood = likelihood
        self.num_classes = num_classes
        self.register_buffer("features", features)
        self.graph = graph
        self.fc = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(lower_bound=-1., upper_bound=1.)

    def forward(self, x):
        h = self.fc(self.features).tanh()
        h = self.scale_to_bounds(h)
        a = self.rewire(h, self.graph)
        mean = self.mean_module(h)
        covar = self.covar_module(h)
        covar = a @ covar @ a.T
        covar = covar + 1e-3 * torch.eye(covar.shape[-1], dtype=covar.dtype, device=covar.device)
        x = x.squeeze().long()
        mean = mean[:, x]
        covar = covar[:, x, :][:, :, x]
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
    