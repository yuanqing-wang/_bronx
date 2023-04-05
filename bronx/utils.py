import torch

def get_candidates(g, k):
    import dgl
    import torch
    candidates = [g] + [dgl.khop_graph(g, k) for k in range(2, k+1)]
    print(candidates)
    g_ref = candidates[-1]
    src_ref, dst_ref = g_ref.edges()
    for g in candidates:
        src, dst = g.edges()
        idxs = torch.logical_and(
            src.unsqueeze(-1) == src_ref.unsqueeze(0),
            dst.unsqueeze(-1) == dst_ref.unsqueeze(0)
        )
        idxs = (1 * idxs).argmax(-1)
        g.edata["idxs"] = idxs
    return candidates        

def combine(candidates, coefficients):
    candidates = [g.local_var() for g in candidates]
    g_ref = candidates[-1]
    g_ref.edata["e"] = torch.ones(g_ref.num_edges(), device=g_ref.device) * coefficients[-1]
    for candidate, coefficient in zip(candidates[:-1], coefficients[:-1]):
        g_ref.edata["e"] = g_ref.edata["e"].scatter_add(
            dim=0,
            index=candidate.edata["idxs"].long(),
            src=torch.ones(g_ref.num_edges(), device=g_ref.device) * coefficient
        )
    return g_ref

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

        elif any(loss <= best_loss for loss, best_loss in zip(losses, self.best_losses)):
            if all(loss <= best_loss for loss, best_loss in zip(losses, self.best_losses)):
                import copy
                self.best_state = copy.deepcopy(model.state_dict())
            self.best_losses = [min(loss, best_loss) for loss, best_loss in zip(losses, self.best_losses)]
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
