from typing import Optional, Callable
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
from .distributions import SoftBernoulli
from .utils import weighted_cross_entropy_with_logits

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

class GraphAutoEncoder(nn.Module):
    hidden_features: int
    out_features: int

    def setup(self):
        self.gcn0 = GCN(self.hidden_features, activation=jax.nn.relu)
        self.gcn1 = GCN(self.out_features)

    def encode(self, a, h):
        h = self.gcn0(a, h)
        z = self.gcn1(a, h)
        return z

    def decode(self, z):
        a_hat = z @ z.transpose()
        return a_hat

    def __call__(self, a, h):
        a_hat = self.decode(self.encode(a, h))
        return a_hat

    def loss(self, a, h):
        pos_weight = float(a.shape[0] * a.shape[0] - a.sum()) / a.sum()
        norm = a.shape[0] * a.shape[0] / (a.shape[0] * a.shape[0] - a.sum()) * 2

        a_hat = self(a, h)
        loss = weighted_cross_entropy_with_logits(
            labels=a.todense(),
            logits=a_hat,
            pos_weight=pos_weight,
        )

        loss = norm * loss
        return loss.mean()


class SoftBernoulliGraphVariationalAutoencoder(nn.Module):
    features: int

    def setup(self):
        self.gcn0 = GCN(self.features)
        self.gcn1 = GCN(self.features)
        self.fc_rate0 = nn.Dense(self.features, use_bias=False)
        self.fc_rate1 = nn.Dense(1, use_bias=False)
        self.fc_alpha = nn.Dense(1, use_bias=False)

    def parameterize(self, a, h):
        h = self.gcn0(a, h)
        h = self.gcn1(a, h)
        rate0, rate1, alpha = self.fc_rate0(h), self.fc_rate1(h), self.fc_alpha(h)
        return rate0, rate1, alpha

    def encode(self, a, h):
        rate0, rate1, alpha = self.parameterize(a, h)
        rate0 = (rate0 @ rate0.transpose()) / rate0.shape[-1]
        rate1 = (jnp.expand_dims(rate1, 0) + jnp.expand_dims(rate1, 1)).squeeze(-1)
        alpha = (jnp.expand_dims(rate0, 0) + jnp.expand_dims(rate0, 1)).squeeze(-1)
        q_z = SoftBernoulli(rate0, rate1, alpha)
        return q_z

    def decode(self, q_z):
        a_hat = q_z.sample()
        return a_hat

    def loss(self, a, h, key):
        rate0, rate1, alpha = self.parameterize(a, h)

        idxs = a.indices
        src_real, dst_real = idxs[:, 0], idxs[:, 1]
        rate0_real = jnp.exp((rate0[src_real] + rate0[dst_real]).sum(-1))
        rate1_real = jnp.exp((rate1[src_real] + rate1[dst_real]).sum(-1))
        alpha_real = jax.nn.sigmoid((alpha[src_real] * alpha[dst_real]).sum(-1) / alpha.shape[-1] ** 0.5)
        q_z_real = SoftBernoulli(rate0_real, rate1_real, alpha_real)

        key_src, key_dst = jax.random.split(key)
        src_fake = jax.random.randint(key_src, src_real.shape, 0, len(src_real))
        dst_fake = jax.random.randint(key_dst, dst_real.shape, 0, len(dst_real))
        rate0_fake = jnp.exp((rate0[src_fake] + rate1[dst_fake]).sum(-1))
        rate1_fake = jnp.exp((rate1[src_fake] + rate1[dst_fake]).sum(-1))
        alpha_fake = jax.nn.sigmoid((alpha[src_real] * alpha[dst_fake]).sum(-1) / alpha.shape[-1])
        q_z_fake = SoftBernoulli(rate0_fake, rate0_fake, alpha_fake)

        mll = q_z_real.log_prob(1.0) + q_z_fake.log_prob(0.0)
        return -mll.mean()
