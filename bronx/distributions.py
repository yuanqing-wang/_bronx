import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

class ProjectedTruncatedExponential(tfd.TransformedDistribution):
    def __init__(self, rate, scale, shift):
        parameters = dict(locals())
        low = jnp.exp(-1.0 / rate)
        uniform = tfd.Uniform(low=low, high=1.0)
        super().__init__(
            distribution=uniform,
            bijector=tfp.bijectors.Chain(
                (
                    tfp.bijectors.Shift(shift),
                    tfp.bijectors.Scale(scale),
                    tfp.bijectors.Scale(-rate),
                    tfp.bijectors.Log(),
                )
            )
        )
        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            rate=tfp.util.ParameterProperties(),
            scale=tfp.util.ParameterProperties(),
            shift=tfp.util.ParameterProperties(),
        )

class SoftBernoulli(tfd.MixtureSameFamily):
    def __init__(self, rate0, rate1, alpha):
        parameters = dict(locals())

        left = (1.0 - alpha) * (rate0 / (1.0 - jnp.exp(-rate0)))
        right = alpha * (rate1 / (1.0 - jnp.exp(-rate1)))

        rate = jnp.stack((rate0, rate1))
        scale = jnp.array((1.0, -1.0)).reshape((2,) + (len(rate.shape) - 1) * (1, ))
        shift = jnp.array((0.0, 1.0)).reshape((2,) + (len(rate.shape) - 1) * (1, ))

        components_distribution = ProjectedTruncatedExponential(
            rate=rate, scale=scale, shift=shift,
        )
        mixture_distribution = tfd.Categorical(
            probs=jnp.stack((left, right)),
        )
        
        super().__init__(
            mixture_distribution=mixture_distribution,
            components_distribution=components_distribution,
            reparameterize=True,
        )
        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            rate0=tfp.util.ParameterProperties(),
            rate1=tfp.util.ParameterProperties(),
            alpha=tfp.util.ParameterProperties(),
        )
