from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
import dgl

def segment_softmax(logits: jnp.ndarray,
                    segment_ids: jnp.ndarray,
                    num_segments: Optional[int] = None,
                    indices_are_sorted: bool = False,
                    unique_indices: bool = False):
  """Computes a segment-wise softmax.
  For a given tree of logits that can be divded into segments, computes a
  softmax over the segments.
    logits = jnp.ndarray([1.0, 2.0, 3.0, 1.0, 2.0])
    segment_ids = jnp.ndarray([0, 0, 0, 1, 1])
    segment_softmax(logits, segments)
    >> DeviceArray([0.09003057, 0.24472848, 0.66524094, 0.26894142, 0.7310586],
    >> dtype=float32)
  Args:
    logits: an array of logits to be segment softmaxed.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be maxed over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since ``num_segments`` determines the size of
      the output, a static value must be provided to use ``segment_sum`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted
    unique_indices: whether ``segment_ids`` is known to be free of duplicates
  Returns:
    The segment softmax-ed ``logits``.
  """
  # First, subtract the segment max for numerical stability
  maxs = jax.ops.segment_max(logits, segment_ids, num_segments, indices_are_sorted,
                     unique_indices)
  logits = logits - maxs[segment_ids]
  # Then take the exp
  logits = jnp.exp(logits)
  # Then calculate the normalizers
  normalizers = jax.ops.segment_sum(logits, segment_ids, num_segments,
                            indices_are_sorted, unique_indices)
  normalizers = normalizers[segment_ids]
  softmax = logits / normalizers
  return softmax


def matrix_power_series(a, max_power):
    result = []
    eye = a._eye(*a.shape, k=0)
    a = a - eye
    for idx in range(max_power):
        result.append(a + eye)
        a = a @ a
    return result

# @jax.jit
def sum_matrix_power_series(a, c):
    target_shape = c.shape + a.shape[1:]
    broadcast_dimension = [len(c.shape) - 1, len(c.shape), len(c.shape) + 1]
    a = sparse.bcoo_broadcast_in_dim(
        a,
        shape=target_shape,
        broadcast_dimensions=broadcast_dimension,
    )
    c = c[..., jnp.newaxis, jnp.newaxis]
    a = a * c
    a = a.sum(-3)
    return a
    


def gat(a, h, w, wk, wq):
    k = h @ wk
    q = h @ wq
    src, dst = a.indices[:, 0], a.indices[:, 1]
    k_src, q_dst = k[src], q[dst]
    a_hat = (k_src * q_dst).sum(-1)
    a_hat = segment_softmax(a_hat, dst, num_segments=a.nse)
    a_hat = sparse.BCOO((a_hat, a.indices), shape=a.shape)
    h = a_hat @ h
    h = h @ w
    return h

def gcn(a, h, w):
     degrees = a.sum(-1).todense()
     norm = jnp.expand_dims(degrees ** (-0.5), -1)
     h = h * norm
     h = h @ w
     h = a @ h
     h = h * norm
     return h

def model(a, h, y=None, mask=None, depth=0, width=0, n_classes=0, layer=gcn):
    c = numpyro.sample("c", dist.LogNormal(jnp.ones(3), jnp.ones(3)))
    a = sum_matrix_power_series(a, c)

    for idx in range(depth - 1):
        w = numpyro.sample(
            "W%s" % idx,
            dist.Normal(
                jnp.zeros((h.shape[-1], width)),
                jnp.ones((h.shape[-1], width)),
            ),
        )
        h = layer(a, h, w)
        h = jax.nn.silu(h)

    idx = depth - 1
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
    # a = sparse.BCOO.fromdense(g.adj().coalesce().to_dense().numpy())

    a = [dgl.khop_adj(g, k) for k in range(3)]
    a = [sparse.BCOO.fromdense(_a.numpy()) for _a in a]
    a = sparse.sparsify(jnp.stack)(a)

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
        num_warmup=2000,
        num_samples=2000,
        thinning=50,
    )

    mcmc.run(jax.random.PRNGKey(2666), a, h, label, train_mask)
    samples = mcmc.get_samples()
    print(samples)
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
