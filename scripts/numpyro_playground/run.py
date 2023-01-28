from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

def layer(a, h, w, wk, wq):
    k = h @ wk
    q = h @ wq
    src, dst = a.index
    k_src, q_dst = k[src], q[dst]
    a_hat = (k_src * q_dst).sum(-1)
    a_hat = sparse.BCOO((a_hat, a.indices), shape=a.shape)
    h = a_hat @ h
    h = h @ w
    return h


def model(a, h, y=None, mask=None, depth=0, width=0, n_classes=0):
    for idx in range(depth - 1):
        wk = numpyro.sample(
            "WK%s" % idx,
            dist.Normal(
                jnp.zeros((h.shape[-1], width)),
                jnp.ones((h.shape[-1], width)),
            ),
        )

        wq = numpyro.sample(
            "WK%s" % idx,
            dist.Normal(
                jnp.zeros((h.shape[-1], width)),
                jnp.ones((h.shape[-1], width)),
            ),
        )

        w = numpyro.sample(
            "W%s" % idx,
            dist.Normal(
                jnp.zeros((h.shape[-1], width)),
                jnp.ones((h.shape[-1], width)),
            ),
        )
        h = layer(a, h, w, wk, wq)
        h = jax.nn.silu(h)

    idx = depth - 1

    wk = numpyro.sample(
        "WK%s" % idx,
        dist.Normal(
            jnp.zeros((h.shape[-1], n_classes)),
            jnp.ones((h.shape[-1], n_classes)),
        ),
    )

    wq = numpyro.sample(
        "WK%s" % idx,
        dist.Normal(
            jnp.zeros((h.shape[-1], width)),
            jnp.ones((h.shape[-1], width)),
        ),
    )

    w = numpyro.sample(
        "W%s" % idx,
        dist.Normal(
            jnp.zeros((h.shape[-1], n_classes)),
            jnp.ones((h.shape[-1], n_classes)),
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


def run(args):
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

    global model
    model = partial(model, n_classes=label.max()+1, width=args.width, depth=args.depth)
    kernel = NUTS(model, init_strategy=numpyro.infer.init_to_feasible())
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=10000,
        thinning=10,

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
    accuracy = (((y_hat - y) == 0).sum() / y.shape[0]).item()

    handle = open(args.out, "a")
    import json
    args = vars(args)
    out = args.pop("out")
    args["accuracy"] = accuracy
    result = json.dumps(args) + "\n"
    handle.write(result)
    handle.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out", type=str, default="out.txt")
    args = parser.parse_args()
    run(args)
