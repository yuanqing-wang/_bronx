from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
import galax
import optax
from bronx.layers import GCN
from bronx.utils import EarlyStopping


class Model(nn.Module):
    features: int
    deterministic: bool = False

    def setup(self):
        self.dropout0 = nn.Dropout(0.5, deterministic=self.deterministic)
        self.dropout1 = nn.Dropout(0.5, deterministic=self.deterministic)
        self.gcn0 = GCN(16, activation=jax.nn.relu)
        self.gcn1 = GCN(self.features)

    def __call__(self, a, h):
        h = self.dropout0(h)
        h = self.gcn0(a, h)
        h = self.dropout1(h)
        h = self.gcn1(a, h)
        return h

def run(args):
    from galax.data.datasets.nodes.planetoid import cora, citeseer, pubmed
    g = locals()[args.data.lower()]()
    g = g.add_self_loop()
    a = g.adj()
    h = g.ndata['h']
    features = g.ndata['label'].max() + 1
    y_ref = jax.nn.one_hot(g.ndata['label'], features)

    model = Model(features, deterministic=False)
    model_eval = Model(features, deterministic=True)

    key = jax.random.PRNGKey(2666)
    key, key_dropout = jax.random.split(key)

    params = model.init({"params": key, "dropout": key_dropout}, a, h)

    from flax.core import FrozenDict
    mask = FrozenDict(
        {"params":
            {
                "gcn0": True,
                "gcn1": False,
            },
        },
    )

    optimizer = optax.chain(
        optax.additive_weight_decay(5e-4, mask=mask),
        optax.adam(1e-2),
    )

    from flax.training.train_state import TrainState
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    def loss(params, key):
        y = model.apply(params, a, h, rngs={"dropout": key})
        return optax.softmax_cross_entropy(
            y[g.ndata["train_mask"]],
            y_ref[g.ndata["train_mask"]],
        ).mean()

    # @jax.jit
    def step(state, key):
        key, new_key = jax.random.split(key)
        grad_fn = jax.grad(partial(loss, key=new_key))
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, key

    @jax.jit
    def eval(state):
        params = state.params
        y = model_eval.apply(params, a, h)
        accuracy_vl = (y_ref[g.ndata['val_mask']].argmax(-1) ==
                y[g.ndata['val_mask']].argmax(-1)).sum() /\
                g.ndata['val_mask'].sum()
        loss_vl = optax.softmax_cross_entropy(
            y[g.ndata['val_mask']],
            y_ref[g.ndata['val_mask']],
        ).mean()
        return accuracy_vl, loss_vl

    @jax.jit
    def test(state):
        params = state.params
        y = model_eval.apply(params, a, h)
        accuracy_te = (y_ref[g.ndata['test_mask']].argmax(-1) ==
            y[g.ndata['test_mask']].argmax(-1)).sum() /\
            g.ndata['test_mask'].sum()
        loss_te = optax.softmax_cross_entropy(
            y[g.ndata['test_mask']],
            y_ref[g.ndata['test_mask']],
        ).mean()
        return accuracy_te, loss_te

    from galax.nn.utils import EarlyStopping
    early_stopping = EarlyStopping(10)

    import tqdm
    for _ in tqdm.tqdm(range(1000)):
        state, key = step(state, key)
        accuracy_vl, loss_vl = eval(state)
        if early_stopping((-accuracy_vl, loss_vl), state.params):
            state = state.replace(params=early_stopping.params)
            break

    accuracy_te, _ = test(state)
    print(accuracy_te)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    args = parser.parse_args()
    run(args)
