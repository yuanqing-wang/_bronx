import torch
from .layers import BronxLayer
import torchsde
from torchsde import BrownianInterval

class BronxModel(torch.nn.Module):
    def __init__(
            self, 
            in_features, hidden_features, out_features, num_heads=1, 
            dropout0=0.0, dropout1=0.0, gamma=0.0, gain=0.0,
        ):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_out = torch.nn.Linear(hidden_features, out_features, bias=False)
        self.sde = BronxLayer(hidden_features, gamma=gamma, gain=gain)
        self.dropout0 = torch.nn.Dropout(dropout0)
        self.dropout1 = torch.nn.Dropout(dropout1)

    def forward(self, g, h):
        # self.sde.graph = g 
        h = self.dropout0(h)
        h = self.fc_in(h)# .tanh()
        t = torch.tensor([0.0, 1.0], device=h.device, dtype=h.dtype)
        h = torchsde.sdeint(
            self.sde, 
            h, 
            t,
            # bm=BrownianInterval(
            #     t0=t[0],
            #     t1=t[-1],
            #     size=(h.shape[0], 1),
            #     device=h.device,
            #     cache_size=None,
            #     pool_size=4,
            # ),
            dt=0.05,
            logqp=self.training,
        )

        if self.training:
            h, kl = h
        else:
            kl = 0.0

        h = h[-1]
        # h = torch.nn.functional.silu(h)
        h = self.dropout1(h)
        h = self.fc_out(h)
        return h, kl
