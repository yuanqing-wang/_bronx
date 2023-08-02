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
        # full_inputs = torch.cat([inducing_points, x], dim=-2)
        # full_output = self.model.forward(full_inputs, **kwargs)
        # full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        # num_induc = inducing_points.size(-2)
        # test_mean = full_output.mean[..., num_induc:]
        # induc_induc_covar = full_covar[
        #     ..., :num_induc, :num_induc
        # ].add_jitter(self.jitter_val)
        # induc_data_covar = full_covar[..., :num_induc, num_induc:].to_dense()
        # data_data_covar = full_covar[..., num_induc:, num_induc:]

        num_induc = inducing_points.size(-2)

        data_result = self.model.forward(x, ix1=x_indices, graph=self.graph)
        test_mean = data_result.mean
        data_data_covar = data_result.lazy_covariance_matrix

        induc_induc_covar = self.model.forward(
            x=inducing_points,
            ix1=self.inducing_indices,
            graph=self.graph,
        ).lazy_covariance_matrix

        induc_data_covar = self.model.forward(
            x=inducing_points,
            x2=x,
            ix1=self.inducing_indices,
            ix2=x_indices,
            graph=self.graph,
        ).lazy_covariance_matrix

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies
            # when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)

        interp_term = L.solve(
            induc_data_covar.to_dense().to(L.dtype)
        ).to(induc_data_covar.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (
            interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)
        ).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(
                variational_inducing_covar, middle_term
            )

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(self.jitter_val).to_dense()
                + interp_term.transpose(-1, -2)
                @ middle_term.to_dense()
                @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(
                    interp_term.transpose(-1, -2), middle_term @ interp_term
                ),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    @cached(name="pseudo_points_memo")
    def pseudo_points(self) -> Tuple[Tensor, Tensor]:
        # TODO: have var_mean, var_cov come from
        # a method of _variational_distribution
        # while having Kmm_root be a root decomposition to enable
        # CIQVariationalDistribution support.

        # retrieve the variational mean, m and covariance matrix, S.
        if not isinstance(
            self._variational_distribution, CholeskyVariationalDistribution
        ):
            raise NotImplementedError(
                "Only CholeskyVariationalDistribution"
                "has pseudo-point support currently, ",
                "but your _variational_distribution is a ",
                self._variational_distribution.__name__,
            )

        var_cov_root = TriangularLinearOperator(
            self._variational_distribution.chol_variational_covar
        )
        var_cov = CholLinearOperator(var_cov_root)
        var_mean = self.variational_distribution.mean
        if var_mean.shape[-1] != 1:
            var_mean = var_mean.unsqueeze(-1)

        # compute R = I - S
        cov_diff = var_cov.add_jitter(-1.0)
        cov_diff = -1.0 * cov_diff

        # K^{1/2}
        Kmm = self.model.covar_module(
            x1=self.inducing_points,
            ix1=self.inducing_indices,
            graph=self.graph,
        )
        Kmm_root = Kmm.cholesky()

        # D_a = (S^{-1} - K^{-1})^{-1} = S + S R^{-1} S
        # note that in the whitened case R = I - S, unwhitened R = K - S
        # we compute (R R^{T})^{-1} R^T S for stability reasons as R is probably not PSD.
        eval_var_cov = var_cov.to_dense()
        eval_rhs = cov_diff.transpose(-1, -2).matmul(eval_var_cov)
        inner_term = cov_diff.matmul(cov_diff.transpose(-1, -2))
        # TODO: flag the jitter here
        inner_solve = inner_term.add_jitter(self.jitter_val).solve(
            eval_rhs, eval_var_cov.transpose(-1, -2)
        )
        inducing_covar = var_cov + inner_solve

        inducing_covar = Kmm_root.matmul(inducing_covar).matmul(
            Kmm_root.transpose(-1, -2)
        )

        # mean term: D_a S^{-1} m
        # unwhitened: (S - S R^{-1} S) S^{-1} m = (I - S R^{-1}) m
        rhs = cov_diff.transpose(-1, -2).matmul(var_mean)
        # TODO: this jitter too
        inner_rhs_mean_solve = inner_term.add_jitter(self.jitter_val).solve(
            rhs
        )
        pseudo_target_mean = Kmm_root.matmul(inner_rhs_mean_solve)

        # ensure inducing covar is psd
        # TODO: make this be an explicit root decomposition
        try:
            pseudo_target_covar = CholLinearOperator(
                inducing_covar.add_jitter(self.jitter_val).cholesky()
            ).to_dense()
        except NotPSDError:
            from linear_operator.operators import DiagLinearOperator

            evals, evecs = torch.linalg.eigh(inducing_covar)
            pseudo_target_covar = (
                evecs.matmul(DiagLinearOperator(evals + self.jitter_val))
                .matmul(evecs.transpose(-1, -2))
                .to_dense()
            )

        return pseudo_target_covar, pseudo_target_mean

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
    


