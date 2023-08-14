from functools import lru_cache
from typing import Callable
import torch
import dgl
from dgl import function as fn
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
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._g = None
        self._e = None

    @property
    def g(self):
        return self._g

    @property
    def e(self):
        return self._e
    
    @g.setter
    def g(self, g):
        self._g = g

    @e.setter
    def e(self, e):
        self._e = e

    def forward(self, t, x):
        x, k = x[:, :-x.shape[0]], x[:, -x.shape[0]:]
        # x = x - x.mean(dim=-1, keepdim=True)
        # x = torch.nn.functional.normalize(x, dim=-1)
        k = x @ x.t()
        g = self.g
        e = self.e
        g = g.local_var()
        g.edata["e"] = e
        g.ndata["x"] = x
        g.update_all(fn.u_mul_e("x", "e", "m"), fn.sum("m", "x"))
        x = g.ndata["x"]
        x = torch.cat([x, k], dim=1)
        return x

class ODEBlock(torch.nn.Module):
    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc

    def forward(self, g, h, e, t=1.0):
        g = g.local_var()
        self.odefunc.g = g
        self.odefunc.e = e
        t = torch.tensor([0, t], device=h.device, dtype=h.dtype)
        k = odeint(self.odefunc, h, t, method="rk4")[1]
        k = k[:, -k.shape[0]:]
        return k

class LinearDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ode_block = ODEBlock(ODEFunc())

    def forward(self, g, h, e, t=1.0, gamma=-1.0):
        g = g.local_var()
        g.edata["e"] = e
        src, dst = g.edges()
        g.edata["e"][src==dst, ...] = gamma
        k = torch.zeros(
            g.number_of_nodes(), g.number_of_nodes(), 
            device=h.device, dtype=h.dtype
        )
        h = torch.cat([h, k], dim=-1)
        result = self.ode_block(g, h, g.edata["e"], t=t)
        return result

class Rewire(torch.nn.Module):
    def __init__(self, in_features, out_features, t=1.0, gamma=-1.0):
        super().__init__()
        self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, out_features, bias=False)
        self.register_buffer("t", torch.tensor(t))
        self.register_buffer("gamma", torch.tensor(gamma))
        self.linear_diffusion = LinearDiffusion()

    def forward(self, feat, graph):
        graph = graph.local_var()
        k = self.fc_k(feat)
        q = self.fc_q(feat)
        graph.ndata["k"] = k
        graph.ndata["q"] = q
        graph.apply_edges(dgl.function.u_dot_v("k", "q", "e"))
        e = graph.edata["e"]
        e = edge_softmax(graph, e)
        k = self.linear_diffusion(graph, feat, e, t=self.t, gamma=self.gamma)
        return k

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
            features, graph, in_features, hidden_features, t, gamma, log_sigma,
            activation,
        ):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size((num_classes,)),
        )

        self.covar_module = ScaleKernel(GaussianSymmetrizedKLKernel())
        self.rewire = Rewire(
            hidden_features, hidden_features, t=t, gamma=gamma,
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
            gamma: float=-1.0,
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
        mean = self.mean_module(h)
        covar = self.rewire(h, self.graph)
        covar = covar + self.log_sigma.exp() \
        * torch.eye(covar.shape[-1], dtype=covar.dtype, device=covar.device)
        print(covar)
        x = x.squeeze().long()
        mean = mean[..., x]
        covar = covar[..., x, :][..., :, x]
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    