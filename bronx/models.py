import torch
from .layers import BronxLayer
import torchsde
from torchsde import BrownianInterval

class BronxModel(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_out = torch.nn.Linear(hidden_features, out_features, bias=False)
        self.sde = BronxLayer(hidden_features)

    def forward(self, g, h):
        self.sde.graph = g
        h = self.fc_in(h)
        t = torch.tensor([0.0, 1.0], device=h.device)
        h = torchsde.sdeint(
            self.sde, 
            h, 
            t,
            bm=BrownianInterval(
                t0=t[0],
                t1=t[-1],
                size=(h.shape[0], 1),
                device=h.device,
                cache_size=None,
                pool_size=4,
            )

        )[-1]
        h = self.fc_out(h)
        return h
