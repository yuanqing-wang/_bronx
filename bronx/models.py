from functools import partial, lru_cache
import math
import torch
import dgl
from dgl.nn.functional import edge_softmax
import pyro
from pyro import poutine
from pyro.contrib.gp.models.vsgp import VariationalSparseGP as VSGP
from pyro.nn.module import pyro_method
from pyro import distributions as dist
from pyro.distributions.util import eye_like


class Rewire(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, feat, graph):
        graph = graph.local_var()
        k = self.fc_k(feat)
        q = self.fc_q(feat)
        graph.ndata["k"] = k
        graph.ndata["q"] = q
        graph.apply_edges(dgl.function.u_dot_v("k", "q", "e"))
        e = graph.edata["e"]
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

@lru_cache(maxsize=1)
def graph_exp_inv(graph):
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
    a = torch.linalg.matrix_exp(-a)
    return a

def graph_conditional(
    X,
    graph,
    iX,
    kernel,
    f_loc,
    f_scale_tril=None,
    Lff=None,
    full_cov=False,
    whiten=False,
    jitter=1e-6,
):
    # N = X.size(0)
    # M = Xnew.size(0)

    if graph is not None:
        A = graph_exp(graph)
    K = kernel(X)
    M = len(iX)
    N = len(X)
    latent_shape = f_loc.shape[:-1]

    if Lff is None:
        Kff = K.contiguous()
        Kff = Kff + torch.eye(Kff.shape[-1], device=Kff.device) * jitter
        Lff = torch.linalg.cholesky(Kff)

    # convert f_loc_shape from latent_shape x N to N x latent_shape
    f_loc = f_loc.permute(-1, *range(len(latent_shape)))
    # convert f_loc to 2D tensor for packing
    f_loc_2D = f_loc.reshape(N, -1)

    # convert f_scale_tril_shape from latent_shape x N x N to N x N x latent_shape
    f_scale_tril = f_scale_tril.permute(-2, -1, *range(len(latent_shape)))
    # convert f_scale_tril to 2D tensor for packing
    f_scale_tril_2D = f_scale_tril.reshape(N, -1)

    if whiten:
        v_2D = f_loc_2D
        W = K
        if f_scale_tril is not None:
            S_2D = f_scale_tril_2D
    else:
        v_2D = torch.linalg.solve_triangular(Lff, f_loc_2D, upper=False)
        W = K
        S_2D = torch.linalg.solve_triangular(Lff, f_scale_tril_2D, upper=False)

    # v_2D = A_inv @ v_2D
    # W = A @ W @ A
    # S_2D = A_inv @ S_2D

    loc_shape = latent_shape + (M,)
    loc = W.matmul(v_2D).t()
    if graph is not None:
        loc = loc @ A
    loc = loc[:, iX]
    loc = loc.reshape(loc_shape)

    W_S_shape = (M,) + f_scale_tril.shape[1:]
    W_S = W.matmul(S_2D)
    if graph is not None:
        W_S = A @ W_S
    W_S = W_S[iX, :]
    W_S = W_S.reshape(W_S_shape)
    # convert W_S_shape from M x N x latent_shape to latent_shape x M x N
    W_S = W_S.permute(list(range(2, W_S.dim())) + [0, 1])

    if full_cov:
        St_Wt = W_S.transpose(-2, -1)
        cov = W_S.matmul(St_Wt)
        return loc, cov

    else:
        var = W_S.pow(2).sum(dim=-1)
        return loc, var

class GraphVariationalSparseGaussianProcess(pyro.nn.PyroModule):
    def __init__(
        self, 
        graph,
        X,
        y,
        iX,
        kernel, 
        likelihood, 
        in_features,
        hidden_features,
        embedding_features,
        jitter=1e-6,
        whiten=False,
    ):
        super().__init__()
        self.graph = graph
        self.num_inducing_points = graph.number_of_nodes()
        self.X = X
        self.y = y
        self.kernel = kernel
        self.likelihood = likelihood
        self.jitter = jitter
        self.whiten = whiten
        self.register_buffer("Xu", X)
        self.register_buffer("iX", iX)

        self.latent_shape = torch.Size((hidden_features,))

        M = self.Xu.size(0)
        u_loc = self.Xu.new_zeros(self.latent_shape + (M,))
        self.u_loc = torch.nn.Parameter(u_loc)

        identity = eye_like(self.Xu, M)
        u_scale_tril = identity.repeat(self.latent_shape + (1, 1))
        self.u_scale_tril = pyro.nn.PyroParam(u_scale_tril, pyro.distributions.constraints.lower_cholesky)

        self.W = torch.nn.Parameter(
            torch.empty(X.shape[-1], embedding_features),
        )
        torch.nn.init.normal_(self.W, std=1.0)

        self.W_out = torch.nn.Parameter(
            torch.empty(embedding_features, y.max() + 1),
        )
        torch.nn.init.normal_(self.W_out, std=1.0)

        self.rewire = Rewire(in_features, embedding_features)


    def set_data(self, X, y):
        self.X = X
        self.y = y

    @pyro_method
    def model(self, iX, y=None):
        X = self.Xu @ self.W
        zero_loc = self.Xu.new_zeros(self.u_loc.shape)
        Luu = None
        if self.whiten:
            graph = self.graph
            identity = eye_like(self.Xu, self.num_inducing_points)
            pyro.sample(
                self._pyro_get_fullname("u"),
                dist.MultivariateNormal(
                        zero_loc, 
                        identity,
                    ).to_event(
                    zero_loc.dim() - 1
                ),
            )
        else:
            graph = None
            Kuu = self.kernel(X).contiguous()
            Kuu = graph_exp(self.graph) @ Kuu @ graph_exp(self.graph).T
            Kuu = Kuu + torch.eye(Kuu.shape[-1], device=Kuu.device) * self.jitter
            Luu = torch.linalg.cholesky(Kuu)
            pyro.sample(
                self._pyro_get_fullname("u"),
                dist.MultivariateNormal(zero_loc, scale_tril=Luu).to_event(
                    zero_loc.dim() - 1
                ),
            )

        f_loc, f_var = graph_conditional(
            X=X,
            graph=graph,
            iX=iX,
            kernel=self.kernel, 
            f_loc=self.u_loc, 
            f_scale_tril=self.u_scale_tril, 
            full_cov=False, 
            jitter=self.jitter, 
            whiten=self.whiten,
            Lff=Luu,
        )

        f = dist.Normal(f_loc, f_var.sqrt())()
        f_swap = f.transpose(-1, -2)
        f_swap = f_swap @ self.W_out
        y_dist = dist.Categorical(logits=f_swap)
        y_dist = y_dist.expand_by(self.y.shape[: -f.dim() + 1]).to_event(self.y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)
    
    @pyro_method
    def guide(self, iX, y=None):
        pyro.sample(
            self._pyro_get_fullname("u"),
            dist.MultivariateNormal(self.u_loc, scale_tril=self.u_scale_tril).to_event(
                self.u_loc.dim() - 1
            ),
        )

    




        


