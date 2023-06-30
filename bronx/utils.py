import torch


class EarlyStopping(object):
    best_losses = None
    best_state = None
    counter = 0

    def __init__(self, patience=10):
        self.patience = patience

    def __call__(self, losses, model):
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
                import copy

                self.best_state = copy.deepcopy(model.state_dict())
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


def personalized_page_rank(alpha, k):
    result = torch.tensor([alpha * (1 - alpha) ** idx for idx in range(k)])
    result = torch.nn.functional.normalize(result, p=1.0, dim=0)
    return result
