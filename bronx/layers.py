import math
import jax
from jax import numpy as jnp
from jax.experimental import sparse
import numpyro
import numpyro.distributions as dist


def approximate_matrix_exp(a, k:int=4):
    result = a
    for i in range(1, k):
        # a = a @ a
        a = sparse.bcoo_dot_general(
            a, a,
            dimension_numbers=(
                ((0,), (1,)), 
                ((), ()),
            ),
        )
        result = result + a / math.factorial(i)
    return result

def linear_diffusion(src, dst, e, h, gamma=0.0):
    n = h.shape[-2]
    idxs = jnp.stack([src, dst], axis=-1)
    e = jnp.repeat(e, 2, axis=-1)
    idxs = jnp.concatenate([idxs, jnp.flip(idxs, -2)], axis=-2)
    idxs = jnp.concatenate([idxs, jnp.arange(n)[..., None].repeat(2, -1)], axis=-2)
    e = jnp.concatenate([e, jnp.ones(n) * gamma], axis=-1)
    a = sparse.BCOO((e, idxs), shape=(n, n))
    # a = a.at[..., jnp.arange(n), jnp.arange(n)].set(gamma)
    a = a / a.sum(-1, keepdims=True).todense()
    print(a)
    a = approximate_matrix_exp(a)
    return a @ h

def bronx_layer(src, dst, h, gamma, scale, in_features, out_features, idx):
    n = h.shape[-2]
    with numpyro.plate(f"edges{idx}", len(src)):
        with numpyro.handlers.scale(scale=scale):
            e = numpyro.sample(
                f"e{idx}",
                numpyro.distributions.LogNormal(
                    jnp.zeros(n), jnp.ones(n),
                )
            )
    h = linear_diffusion(src, dst, e, h, gamma=gamma)
    return h

def bronx_layer_guide(src, dst, h, gamma, scale, in_features, out_features, idx):
    n = h.shape[-2]
    w_k = numpyro.param(
        f"w_k{idx}",
        lambda key: jax.random.normal(
                key=jax.random.split(key, 3)[0], 
                shape=(in_features, out_features)),
    )

    w_mu = numpyro.param(
        f"w_mu{idx}",
        lambda key: jax.random.normal(
                key=jax.random.split(key, 3)[0], 
                shape=(in_features, out_features)),
    )

    w_log_sigma = numpyro.param(
        f"w_log_sigma{idx}",
        lambda key: jax.random.normal(
                key=jax.random.split(key, 3)[1], 
                shape=(in_features, out_features)),
    )

    h = jax.nn.standardize(h)
    k, mu, log_sigma = h @ w_k, h @ w_mu, h @ w_log_sigma
    mu = (k[..., src, :] * mu[..., dst, :]).sum(-1)
    log_sigma = (k[..., src, :] * log_sigma[..., dst, :]).sum(-1)

    with numpyro.plate(f"edges{idx}", len(src)):
        with numpyro.handlers.scale(scale=scale):
            e = numpyro.sample(
                f"e{idx}",
                numpyro.distributions.LogNormal(
                    mu, jnp.exp(log_sigma),
                )
            )
    
    h = linear_diffusion(src, dst, e, h, gamma=gamma)
    return h

# class BronxLayer(pyro.nn.PyroModule):
#     def __init__(
#             self, 
#             in_features, out_features, activation=torch.nn.SiLU(), 
#             dropout=0.0, idx=0, num_heads=4, gamma=0.0, edge_drop=0.0,
#         ):
#         super().__init__()
#         self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)
#         self.fc_mu = torch.nn.Linear(in_features, out_features, bias=False)
#         self.fc_log_sigma = torch.nn.Linear(in_features, out_features, bias=False)
#         self.activation = activation
#         self.idx = idx
#         self.out_features = out_features
#         self.num_heads = num_heads
#         self.dropout = torch.nn.Dropout(dropout)
#         self.linear_diffusion = LinearDiffusion(
#             dropout=edge_drop, gamma=gamma,
#         )

#     def guide(self, g, h):
#         h = h - h.mean(-1, keepdims=True)
#         h = torch.nn.functional.normalize(h, dim=-1)
#         k, mu, log_sigma = self.fc_k(h), self.fc_mu(h), self.fc_log_sigma(h)
     
#         src, dst = g.edges()
#         mu = (k[..., src, :] * mu[..., dst, :]).sum(-1, keepdims=True)
#         log_sigma = (k[..., src, :] * log_sigma[..., dst, :]).sum(-1, keepdims=True)

#         with pyro.plate(f"edges{self.idx}", g.number_of_edges(), device=g.device):
#             with pyro.poutine.scale(None, float(g.ndata["train_mask"].sum() / (2 * g.number_of_edges()))):
#                 e = pyro.sample(
#                         f"e{self.idx}", 
#                         pyro.distributions.LogNormal(
#                         mu, log_sigma.exp(),
#                     ).to_event(1)
#                 )

#         return e

#     def mp(self, g, h, e):
#         # e = edge_softmax(g, e).squeeze(-1)
#         h = self.linear_diffusion(g, h, e.squeeze(-1))
#         return h

#     def forward(self, g, h):
#         with pyro.plate(f"edges{self.idx}", g.number_of_edges(), device=g.device):
#             with pyro.poutine.scale(None, float(g.ndata["train_mask"].sum() / (2 * g.number_of_edges()))):
#                 e = pyro.sample(
#                         f"e{self.idx}", 
#                         pyro.distributions.LogNormal(
#                             torch.zeros(g.number_of_edges(), 1, device=g.device),
#                             torch.ones(g.number_of_edges(), 1, device=g.device),
#                     ).to_event(1)
#                 )

#         h = self.mp(g, h, e)
#         h = self.dropout(h)
#         return h

