from collections import OrderedDict
import torch
import pyro
from pyro import poutine
from .layers import BronxLayer

class BronxModel(torch.nn.Module):
    def __init__(
            self, in_features, hidden_features, out_features, 
            activation=torch.nn.Tanh(),
            depth=2, gamma=1.0, dropout=0.5,
        ):
        super().__init__()
        # self.fc_in = InLayer(in_features, hidden_features)
        self.fc_out = torch.nn.Linear(hidden_features, out_features, bias=False)
        self.activation = activation
        self.depth = depth
        self.layer0 = BronxLayer(in_features, hidden_features, gamma=gamma, idx=0)
        self.dropout = torch.nn.Dropout(dropout)
        for idx in range(1, depth):
            setattr(
                self, 
                f"layer{idx}", 
                BronxLayer(
                    hidden_features, 
                    hidden_features,
                    idx=idx,
                    gamma=gamma,
                )
            )

    def forward(self, g, h, y=None, mask=None):
        # h = self.fc_in(g, h)
        # h = self.activation(h)
        for idx in range(self.depth):
            h = getattr(self, f"layer{idx}")(g, h)
            h = self.activation(h)

        h = self.fc_out(h)
        h = h.softmax(-1)
        
        if mask is not None:
            h = h[..., mask, :]
            if y is not None:
                y = y[..., mask, :]

        if y is not None:
            with pyro.plate("data", h.shape[0], device=h.device):
                pyro.sample(
                    "y", 
                    pyro.distributions.OneHotCategorical(h).to_event(1), 
                    obs=y,
                )

        return h

    def guide(self, g, h, y=None, mask=None):
        # h = self.fc_in(g, h)
        # h = self.activation(h)
        for idx in range(self.depth):
            h = getattr(self, f"layer{idx}").guide(g, h)
            h = self.activation(h)
        return h