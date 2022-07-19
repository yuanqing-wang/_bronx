from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
import galax
import optax
from bronx.layers import GraphAutoEncoder
from bronx.utils import EarlyStopping
from sklearn.metrics import average_precision_score

def run():
    from galax.data.datasets.nodes.planetoid import cora, citeseer, pubmed
    # g = locals()[args.data.lower()]()
    g = cora()
    g = g.add_self_loop()
    a = g.adj()
    h = g.ndata['h']

    model = GraphAutoEncoder(32, 16)
    key = jax.random.PRNGKey(2666)

    params = model.init(key, a, h, method=model.loss)
    optimizer = optax.adam(1e-3)
    from flax.training.train_state import TrainState
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    def loss(params, key):
        loss = model.apply(params, a, h, method=model.loss)
        return loss

    # @jax.jit
    def step(state, key):
        key, new_key = jax.random.split(key)
        grad_fn = jax.grad(partial(loss, key=new_key))
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, key

    def eval(state):
        a_hat = model.apply(state.params, a, h)
        a_hat = jax.nn.sigmoid(a_hat)
        ap = average_precision_score(a.todense(), a_hat)
        return ap


    import tqdm
    for _ in tqdm.tqdm(range(1000)):
        state, key = step(state, key)
        ap = eval(state)
        print(ap)


if __name__ == "__main__":
    run()
