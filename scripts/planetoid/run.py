from functools import partial
import numpy as onp
import jax
from jax import numpy as jnp
import numpyro
import dgl
dgl.use_libxsmm(False)
from bronx.models import bronx_model, bronx_guide

def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    G = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    SRC, DST = G.edges()
    SRC, DST = jnp.array(SRC), jnp.array(DST)
    SRC, DST = SRC[SRC < DST], DST[SRC < DST]
    H = jnp.array(G.ndata["feat"])
    Y = jnp.array(G.ndata["label"])
    Y = jax.nn.one_hot(Y, Y.max() + 1)
    TRAIN_MASK = jnp.array(G.ndata["train_mask"])
    VAL_MASK = jnp.array(G.ndata["val_mask"])
    TEST_MASK = jnp.array(G.ndata["test_mask"])
    SCALE = TRAIN_MASK.sum() / len(SRC)

    model = partial(
        bronx_model,
        gamma=args.gamma,
        scale=SCALE,
        in_features=H.shape[-1],
        out_features=Y.shape[-1],
        hidden_features=args.hidden_features,
        depth=args.depth,
        activation=jax.nn.silu,
    )

    guide = partial(
        bronx_guide,
        gamma=args.gamma,
        scale=SCALE,
        in_features=H.shape[-1],
        out_features=Y.shape[-1],
        hidden_features=args.hidden_features,
        depth=args.depth,
        activation=jax.nn.silu,
    )

    optimizer = numpyro.optim.Adam(1e-2)
    svi = numpyro.infer.SVI(
        model, guide, optimizer, loss=numpyro.infer.TraceMeanField_ELBO()
    )
    result = svi.run(jax.random.PRNGKey(2666), 1000, SRC, DST, H, Y, TRAIN_MASK)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--embedding_features", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--edge_drop", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_particles", type=int, default=16)
    args = parser.parse_args()
    print(args)
    run(args)
