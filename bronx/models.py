from collections import OrderedDict
import torch
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from .layers import bronx_layer, bronx_layer_guide
import functools

def bronx_model(
    src, dst, h, y, mask, gamma, scale, in_features, out_features, hidden_features, 
    depth, activation, 
):
    
    w_in = numpyro.param(
        "w_in",
        lambda key: jax.random.normal(
            key=jax.random.split(key, 2)[0],
            shape=(in_features, hidden_features),
        ),
    )

    w_out = numpyro.param(
        "w_out",
        lambda key: jax.random.normal(
            key=jax.random.split(key, 2)[1],
            shape=(hidden_features, out_features),
        ),
    )

    for idx in range(depth):
        h = bronx_layer(
            src, dst, h, gamma, scale, in_features, out_features, idx,
        )
        h = activation(h)


    if mask is not None:
        h = h[..., mask, :]
        if y is not None:
            y = y[..., mask, :]

    if y is not None:
        with numpyro.plate("data", y.shape[0], device=h.device):
            numpyro.sample(
                "y", 
                numpyro.distributions.OneHotCategorical(h), 
                obs=y,
            )

    return h

def bronx_guide(
    src, dst, h, y, mask, gamma, scale, in_features, out_features, hidden_features, 
    depth, activation,
):
    w_in = numpyro.param(
        "w_in",
        lambda key: jax.random.normal(
            key=jax.random.split(key, 2)[0],
            shape=(in_features, hidden_features),
        ),
    )

    w_out = numpyro.param(
        "w_out",
        lambda key: jax.random.normal(
            key=jax.random.split(key, 2)[1],
            shape=(hidden_features, out_features),
        ),
    )

    for idx in range(depth):
        h = bronx_layer_guide(
            src, dst, h, gamma, scale, in_features, out_features, idx,
        )
        h = activation(h)

    return h

# class BronxModel(pyro.nn.PyroModule):
#     def __init__(
#             self, in_features, hidden_features, out_features, 
#             embedding_features=None,
#             activation=torch.nn.SiLU(), gamma=0.0,
#             depth=2,
#             dropout=0.0,
#             edge_drop=0.0,
#             num_heads=4,
#         ):
#         super().__init__()
#         if embedding_features is None:
#             embedding_features = hidden_features
#         self.fc_in = torch.nn.Linear(in_features, hidden_features)
#         self.fc_out = torch.nn.Linear(hidden_features, out_features)
#         self.activation = activation
#         self.gamma = gamma
#         self.depth = depth

#         for idx in range(depth):
#             setattr(
#                 self, 
#                 f"layer{idx}", 
#                 BronxLayer(
#                     hidden_features, 
#                     embedding_features, 
#                     activation=activation, 
#                     idx=idx,
#                     dropout=dropout,
#                     edge_drop=edge_drop,
#                     num_heads=num_heads,
#                     gamma=gamma,
#                 )
#             )

#     def forward(self, g, h, y=None, mask=None):
#         h = self.fc_in(h)
#         h = self.activation(h)

#         for idx in range(self.depth):
#             h = getattr(self, f"layer{idx}")(g, h)
#             h = self.activation(h)

#         h = self.fc_out(h)
#         h = h.softmax(-1)
        
#         if mask is not None:
#             h = h[..., mask, :]
#             if y is not None:
#                 y = y[..., mask, :]

#         if y is not None:
#             with pyro.plate("data", y.shape[0], device=h.device):
#                 pyro.sample(
#                     "y", 
#                     pyro.distributions.OneHotCategorical(h), 
#                     obs=y,
#                 )

#         return h

#     def guide(self, g, h, y=None, mask=None):
#         h = self.fc_in(h)
#         h = self.activation(h)

#         for idx in range(self.depth):
#             e = getattr(self, f"layer{idx}").guide(g, h)
#             h = getattr(self, f"layer{idx}").mp(g, h, e)
#             h = self.activation(h)

#         return h