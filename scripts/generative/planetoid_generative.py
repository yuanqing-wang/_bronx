from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
import galax
import optax
from bronx.layers import SoftBernoulliGraphVariationalAutoencoder
from bronx.utils import EarlyStopping

def run():
    from galax.data.datasets.nodes.planetoid import cora, citeseer, pubmed
    # g = locals()[args.data.lower()]()
    g = cora()
    g = g.add_self_loop()
    a = g.adj()
    h = g.ndata['h']

    model = SoftBernoulliGraphVariationalAutoencoder(16)
    key = jax.random.PRNGKey(2666)
    key, key_fake = jax.random.split(key)

    params = model.init(key, a, h, key_fake, method=model.loss)
    optimizer = optax.adam(1e-3)
    from flax.training.train_state import TrainState
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )


    def loss(params, key):
        loss = model.apply(params, a, h, key, method=model.loss)
        return loss

    # @jax.jit
    def step(state, key):
        key, new_key = jax.random.split(key)
        grad_fn = jax.grad(partial(loss, key=new_key))
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, key

    def eval(state, key):
        rate0, rate1, alpha = model.apply(
            state.params, a, h, method=model.parameterize,
        )
        idxs = a.indices
        src_real, dst_real = idxs[:, 0], idxs[:, 1]
        rate0_real = jnp.exp((rate0[src_real] + rate0[dst_real]).sum(-1))
        rate1_real = jnp.exp((rate1[src_real] + rate1[dst_real]).sum(-1))
        alpha_real = jax.nn.sigmoid((alpha[src_real] * alpha[dst_real]).sum(-1) / alpha.shape[-1] ** 0.5)
        # q_z_real = SoftBernoulli(rate0_real, rate1_real, alpha_real)

        key_src, key_dst = jax.random.split(key)
        src_fake = jax.random.randint(key_src, src_real.shape, 0, len(src_real))
        dst_fake = jax.random.randint(key_dst, dst_real.shape, 0, len(dst_real))
        rate0_fake = jnp.exp((rate0[src_fake] + rate1[dst_fake]).sum(-1))
        rate1_fake = jnp.exp((rate1[src_fake] + rate1[dst_fake]).sum(-1))
        alpha_fake = jax.nn.sigmoid((alpha[src_real] * alpha[dst_fake]).sum(-1) / alpha.shape[-1] ** 0.5)
        # q_z_fake = SoftBernoulli(rate0_fake, rate0_fake, alpha_fake)

        return alpha_fake.mean(), alpha_real.mean()


    import tqdm
    for _ in tqdm.tqdm(range(1000)):
        state, key = step(state, key)
        alpha_real, alpha_fake = eval(state, key)
        print(alpha_real, alpha_fake)





if __name__ == "__main__":
    run()
