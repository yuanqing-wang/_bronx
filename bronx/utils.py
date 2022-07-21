import jax
import jax.numpy as jnp

class EarlyStopping(object):
    """Early stopping.

    Parameters
    ----------
    patience : int = 10
        Patience for early stopping.

    """

    best_losses = None
    params = None
    counter = 0

    def __init__(self, patience: int = 10):
        self.patience = patience

    def __call__(self, losses, params):
        if self.best_losses is None:
            self.best_losses = losses
            self.counter = 0

        elif any(
            loss <= best_loss
            for loss, best_loss in zip(losses, self.best_losses)
        ):
            if all(
                loss <= best_loss
                for loss, best_loss in zip(losses, self.best_losses)
            ):
                self.params = params
            self.best_losses = [
                min(loss, best_loss)
                for loss, best_loss in zip(losses, self.best_losses)
            ]
            self.counter = 0

        else:
            self.counter += 1
            if self.counter == self.patience:
                return True

        return False

def weighted_cross_entropy_with_logits(labels, logits, pos_weight=1.0):
    log_weight = 1 + (pos_weight - 1) * labels
    loss = (1 - labels) * logits + log_weight * (jnp.log1p(jnp.exp(-jnp.abs(logits))) +
                      jax.nn.relu(-logits))
    return loss
