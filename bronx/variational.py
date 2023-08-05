from cgi import test
from functools import partial, lru_cache
from typing import Optional, Tuple
import math
import torch
from torch import Tensor
import dgl
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    _VariationalDistribution,
    CholeskyVariationalDistribution,
)
from gpytorch.variational import VariationalStrategy
from gpytorch.variational.variational_strategy import (
    pop_from_cache_ignore_args,
    CachingError,
    _linalg_dtype_cholesky,
    SumLinearOperator,
    trace_mode,
    MatmulLinearOperator,
    MultivariateNormal,
    TriangularLinearOperator,
    CholLinearOperator,
    NotPSDError,
    clear_cache_hook,
    RootLinearOperator,
    cached,
)


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

class GraphVariationalStrategy(VariationalStrategy):
    def __init__(
            self,
            model: ApproximateGP,
            inducing_points: Tensor,
            variational_distribution: _VariationalDistribution,
            learn_inducing_locations: bool = False,
            jitter_val: float = 1e-4,
            graph: Optional[dgl.DGLGraph] = None,
            inducing_indices: Optional[Tensor] = None,
    ):
        super().__init__(
            model=model,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
            jitter_val=jitter_val,
        )

        self.graph = graph
        self.register_buffer("inducing_indices", inducing_indices)

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: Optional = None,
        x_indices: Optional[Tensor] = None,
        **kwargs,
    ):
        # Compute full prior distribution
        full_inputs = inducing_points
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean
        induc_induc_covar = full_covar
        induc_data_covar = full_covar
        data_data_covar = full_covar
        induc_induc_covar = full_covar.add_jitter(self.jitter_val)
        induc_data_covar = induc_data_covar.to_dense()

        # Compute full prior distribution
        # full_inputs = torch.cat([inducing_points, x], dim=-2)
        # full_output = self.model.forward(full_inputs, **kwargs)
        # full_covar = full_output.lazy_covariance_matrix

        # # Covariance terms
        # num_induc = inducing_points.size(-2)
        # test_mean = full_output.mean[..., num_induc:]
        # induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)
        # induc_data_covar = full_covar[..., :num_induc, num_induc:].to_dense()
        # data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar) # g-1
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies
            # when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)

        g = graph_exp(self.graph)
        g_inv = graph_exp_inv(self.graph)

        interp_term = L.solve(
            induc_data_covar.to_dense().to(L.dtype) # g-1 g-1
        ).to(induc_data_covar.dtype) #g-1

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (
            interp_term.transpose(-1, -2) @ g
            @ inducing_values.unsqueeze(-1)
        ).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1) # g-1 g-1
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(
                variational_inducing_covar, middle_term
            )


        if trace_mode.on():
            predictive_covar = (
                g @ data_data_covar.add_jitter(self.jitter_val).to_dense() @ g
                + interp_term.transpose(-1, -2)
                @ middle_term.to_dense() 
                @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val) @ g, # g
                MatmulLinearOperator(
                    interp_term.transpose(-1, -2), # g-1
                    middle_term
                    @ interp_term
                ),
            )

        predictive_mean =  predictive_mean[..., x_indices]
        predictive_covar = predictive_covar[..., x_indices, :][..., :, x_indices]
        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(
        self, x: Tensor, prior: bool = False, **kwargs
    ) -> MultivariateNormal:
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = self(
                    self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(
                    prior_function_dist.lazy_covariance_matrix.add_jitter(
                        self.jitter_val
                    )
                )

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = (
                    self._variational_distribution.mean_init_std
                )
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                if isinstance(variational_dist, MultivariateNormal):
                    mean_diff = (
                        (variational_dist.loc - prior_mean)
                        .unsqueeze(-1)
                        .type(_linalg_dtype_cholesky.value())
                    )
                    whitened_mean = (
                        L.solve(mean_diff)
                        .squeeze(-1)
                        .to(variational_dist.loc.dtype)
                    )
                    covar_root = (
                        variational_dist.lazy_covariance_matrix.root_decomposition().root.to_dense()
                    )
                    covar_root = covar_root.type(
                        _linalg_dtype_cholesky.value()
                    )
                    whitened_covar = RootLinearOperator(
                        L.solve(covar_root).to(variational_dist.loc.dtype)
                    )
                    whitened_variational_distribution = (
                        variational_dist.__class__(
                            whitened_mean, whitened_covar
                        )
                    )
                    self._variational_distribution.initialize_variational_distribution(
                        whitened_variational_distribution
                    )

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = (
                    orig_mean_init_std
                )

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, **kwargs)
    


