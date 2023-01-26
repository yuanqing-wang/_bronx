import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

DEPTH = 2
WIDTH = 16

def layer(a, h, w):
    degrees = a.sum(-1).todense()
    norm = jnp.expand_dims(degrees ** (-0.5), -1)
    h = h * norm
    h = h @ w
    h = a @ h
    h = h * norm
    h = jax.nn.silu(h)
    return h


def model(a, h, y=None, mask=None, depth=DEPTH, width=WIDTH):
    for idx in range(depth - 1):
        w = numpyro.sample(
            "W%s" % idx,
            dist.Normal(
                jnp.zeros((h.shape[-1], width)),
                jnp.ones((h.shape[-1], width)),
            ),
        )
        h = layer(a, h, w)

    idx = depth - 1
    w = numpyro.sample(
        "W%s" % idx,
        dist.Normal(
            jnp.zeros((h.shape[-1], label.max() + 1)),
            jnp.ones((h.shape[-1], label.max() + 1)),
        ),
    )
    h = layer(a, h, w)
    h = jax.nn.softmax(h, -1)

    if mask is not None:
        h = h[..., mask, :]

        if y is not None:
            y = y[..., mask]

    if y is not None:
        h = numpyro.sample(
            "obs",
            dist.Categorical(probs=h),
            obs=y,
        )
    else:
        h = numpyro.sample(
            "obs",
            dist.Delta(h),
        )
    return h


def run():
    from dgl.data import CoraGraphDataset
    g = CoraGraphDataset()[0]
    a = sparse.BCOO.fromdense(g.adj().coalesce().to_dense().numpy())
    h = g.ndata["feat"].numpy()
    label = g.ndata["label"].numpy()
    train_mask, val_mask, test_mask = (
        g.ndata["train_mask"].numpy(), 
        g.ndata["val_mask"].numpy(), 
        g.ndata["test_mask"].numpy()
    )

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=10,
        num_samples=10,
    )

    mcmc.run(jax.random.PRNGKey(2666), a, h, label, train_mask)
    samples = mcmc.get_samples()
    model = handlers.substitute(
        handlers.seed(model, jax.random.PRNGKey(2666)), 
        samples,
    )
    model_trace = handlers.trace(model).get_trace(a, h, y=None, mask=val_mask)
    y_hat = model_trace["obs"]["value"].mean(0).argmax(-1) 
    y = label[val_mask]
    accuracy = ((y_hat - y) == 0).sum() / y.shape[0]
    print(accuracy)


if __name__ == "__main__":
    run()
