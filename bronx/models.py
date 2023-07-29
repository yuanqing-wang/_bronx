from functools import partial, lru_cache
import math
import torch
import dgl
import pyro
from pyro import poutine
from pyro.contrib.gp.models.vsgp import VariationalSparseGP as VSGP
from pyro.contrib.gp.util import conditional
from pyro.nn.module import pyro_method
from pyro import distributions as dist

# class GraphVariationalSparseGaussianProcess(VSGP):
#     def __init__(self, graph, X, y, kernel, Xu, likelihood, latent_shape=None):
#         kernel = partial(kernel, graph=graph)
#         super().__init__(
#             X, y, kernel, Xu.float(), likelihood, latent_shape=latent_shape,
#         )
#         del self.Xu
#         self.register_buffer("Xu", Xu)


def graph_conditional(
    Xnew,
    X,
    graph,
    iXnew,
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
    N = len(iX)
    M = len(iXnew)
    latent_shape = f_loc.shape[:-1]

    if Lff is None:
        Kff = kernel(X, graph=graph, iX=iX).contiguous()
        Kff.view(-1)[:: N + 1] += jitter  # add jitter to diagonal
        Lff = torch.linalg.cholesky(Kff)
    Kfs = kernel(X, Xnew, graph=graph, iX=iX, iZ=iXnew)
    # convert f_loc_shape from latent_shape x N to N x latent_shape
    f_loc = f_loc.permute(-1, *range(len(latent_shape)))
    # convert f_loc to 2D tensor for packing
    f_loc_2D = f_loc.reshape(N, -1)
    if f_scale_tril is not None:
        # convert f_scale_tril_shape from latent_shape x N x N to N x N x latent_shape
        f_scale_tril = f_scale_tril.permute(-2, -1, *range(len(latent_shape)))
        # convert f_scale_tril to 2D tensor for packing
        f_scale_tril_2D = f_scale_tril.reshape(N, -1)

    if whiten:
        v_2D = f_loc_2D
        W = torch.linalg.solve_triangular(Lff, Kfs, upper=False).t()
        if f_scale_tril is not None:
            S_2D = f_scale_tril_2D
    else:
        pack = torch.cat((f_loc_2D, Kfs), dim=1)
        if f_scale_tril is not None:
            pack = torch.cat((pack, f_scale_tril_2D), dim=1)

        Lffinv_pack = torch.linalg.solve_triangular(Lff, pack, upper=False)
        # unpack
        v_2D = Lffinv_pack[:, : f_loc_2D.size(1)]
        W = Lffinv_pack[:, f_loc_2D.size(1) : f_loc_2D.size(1) + M].t()
        if f_scale_tril is not None:
            S_2D = Lffinv_pack[:, -f_scale_tril_2D.size(1) :]

    loc_shape = latent_shape + (M,)
    loc = W.matmul(v_2D).t().reshape(loc_shape)

    if full_cov:
        Kss = kernel(Xnew, graph=graph, iX=iXnew)
        Qss = W.matmul(W.t())
        cov = Kss - Qss
    else:
        Kssdiag = kernel(Xnew, graph=graph, iX=iXnew, diag=True)
        Qssdiag = W.pow(2).sum(dim=-1)
        # Theoretically, Kss - Qss is non-negative; but due to numerical
        # computation, that might not be the case in practice.

        var = (Kssdiag - Qssdiag).clamp(min=0)

    if f_scale_tril is not None:
        W_S_shape = (M,) + f_scale_tril.shape[1:]
        W_S = W.matmul(S_2D).reshape(W_S_shape)
        # convert W_S_shape from M x N x latent_shape to latent_shape x M x N
        W_S = W_S.permute(list(range(2, W_S.dim())) + [0, 1])

        if full_cov:
            St_Wt = W_S.transpose(-2, -1)
            K = W_S.matmul(St_Wt)
            cov = cov + K
        else:
            Kdiag = W_S.pow(2).sum(dim=-1)
            var = var + Kdiag
    else:
        if full_cov:
            cov = cov.expand(latent_shape + (M, M))
        else:
            var = var.expand(latent_shape + (M,))

    return (loc, cov) if full_cov else (loc, var)


class GraphVariationalSparseGaussianProcess(VSGP):
    def __init__(
        self, 
        graph,
        X,
        y,
        iX,
        Xu,
        iXu,
        kernel, 
        likelihood, 
        hidden_features,
        latent_shape=None,
        jitter=1e-6,
    ):
        super().__init__(
            X=X,
            y=y,
            kernel=kernel,
            Xu=Xu,
            likelihood=likelihood,
            latent_shape=latent_shape,
            jitter=jitter,
        )
        self.graph = graph
        self.num_inducing_points = len(iXu)
        del self.Xu
        self.register_buffer("Xu", Xu)
        self.register_buffer("iX", iX)
        self.register_buffer("iXu", iXu)
        self.W = torch.nn.Parameter(
            torch.empty(X.shape[-1], hidden_features),
        )
        torch.nn.init.xavier_normal_(self.W)

        self.W_mean = torch.nn.Parameter(
            torch.empty(X.shape[-1], *latent_shape),
        )
        torch.nn.init.xavier_normal_(self.W_mean)

    def set_data(self, X, y):
        self.X = X
        self.y = y

    @pyro_method
    def model(self):
        self.set_mode("model")
        Kuu = self.kernel(
            self.Xu@self.W, 
            self.Xu@self.W, 
            self.graph, 
            self.iXu, 
            self.iXu,
        ).contiguous()
        Kuu.view(-1)[:: self.num_inducing_points + 1] += self.jitter
        Luu = torch.linalg.cholesky(Kuu)
        zero_loc = self.Xu.new_zeros(self.u_loc.shape)
        pyro.sample(
            self._pyro_get_fullname("u"),
            dist.MultivariateNormal(zero_loc, scale_tril=Luu).to_event(
                zero_loc.dim() - 1
            ),
        )
        f_loc, f_var = graph_conditional(
            Xnew=self.X@self.W, 
            X=self.Xu@self.W,
            graph=self.graph,
            iXnew=self.iX,
            iX=self.iXu,
            kernel=self.kernel, 
            f_loc=self.u_loc, 
            f_scale_tril=self.u_scale_tril, 
            full_cov=False, 
            jitter=self.jitter, 
        )
        self.likelihood._load_pyro_samples()
        # f_loc = f_loc + (self.X[self.iX, :] @ self.W_mean).T
        return self.likelihood(f_loc, f_var, self.y)

    def forward(self, X, iX, full_cov=False):
        self._check_Xnew_shape(X)
        self.set_mode("guide")
        loc, cov = graph_conditional(
            Xnew=X@self.W,
            X=self.Xu@self.W,
            graph=self.graph,
            iXnew=iX,
            iX=self.iXu,
            kernel=self.kernel,
            f_loc=self.u_loc,
            f_scale_tril=self.u_scale_tril,
            full_cov=full_cov,
            jitter=self.jitter,
        )
        # loc = loc + (X[iX, :] @ self.W_mean).T
        return loc, cov
        




        


