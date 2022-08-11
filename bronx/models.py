import torch
from .layers import Attention

class Bronx(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        depth: int,
        num_heads: int,
        activation = torch.nn.ELU(),
    ):
        super().__init__()

        layers = []
        layers.append(Attention(in_features, hidden_features))
        layers += [Attention(hidden_features, hidden_features) for _ in range(depth-2)]
        layers.append(Attention(hidden_features, out_features))
        self.layers = torch.nn.ModuleList(layers)
        self.depth = depth

    def forward(self, h, return_adj=False):
        adjs = []
        for layer in self.layers:
            h, a = layer(h)
            adjs.append(a)
        if return_adj:
            a = torch.stack(adjs).mean(0)
            return h, a
        else:
            return h

    @staticmethod
    def reconstruction_loss(a_hat, a):
        a_hat = a_hat * a.sum(-1)
        a_hat = a_hat.fill_diagonal_(1.0)
        pos_weight = (a.shape[0] * a.shape[0] - a.sum()) / a.sum()
        nll = torch.nn.functional.binary_cross_entropy_with_logits(
            input=a_hat,
            target=a,
            pos_weight=pos_weight,
        )
        return nll
