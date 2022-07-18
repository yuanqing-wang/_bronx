from typing import Optional, Callable
import jax
import jax.numpy as jnp
from flax import linen as nn


class GCN(nn.Module):
    features: int
    activation: Optional[Callable] = None

    @nn.compact
    def __call__(self, a, h):
        degrees = a.sum(-1).todense()
        norm = jnp.expand_dims(degrees ** (-0.5), -1)
        h = h * norm
        h = nn.Dense(
            self.features,
            use_bias=False,
            kernel_init=jax.nn.initializers.glorot_uniform()
        )(h)
        h = a @ h
        h = h * norm
        if self.activation is not None:
            h = self.activation(h)
        return h
