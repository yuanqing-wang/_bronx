from audioop import bias
from collections import OrderedDict
import torch
import pyro
from pyro import poutine
from .layers import BronxLayer, Linear

import functools

class LinearDiffusionModel(torch.nn.Module):
    def __init__(
            self, in_features, hidden_features, out_features,
            activation=torch.nn.SiLU(), gamma=0.0,
        ):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features, hidden_features)
        # self.fc_out = torch.nn.Linear(hidden_features, out_features)
        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features),
        )
        
        self.activation = activation
        self.gamma = gamma
        self.linear_diffusion = LinearDiffusion()


    def forward(self, g, h):
        h = self.fc_in(h)
        h = self.activation(h)
        h = self.linear_diffusion(g, h, gamma=self.gamma)
        h = self.fc_out(h)
        return h
    
class BronxModel(pyro.nn.PyroModule):
    def __init__(
            self, in_features, hidden_features, out_features, 
            embedding_features=None,
            activation=torch.nn.SiLU(),
            depth=2,
            num_heads=4,
            num_factors=2,
        ):
        super().__init__()
        if embedding_features is None:
            embedding_features = hidden_features
        self.fc_in = torch.nn.Linear(in_features, hidden_features, bias=False)
        # self.fc_in = Linear(in_features, hidden_features)
        self.fc_out = torch.nn.Linear(hidden_features, out_features, bias=False)

        self.activation = activation
        self.depth = depth
        self.dropout = torch.nn.Dropout(0.5)

        for idx in range(depth):
            setattr(
                self, 
                f"layer{idx}", 
                BronxLayer(
                    hidden_features, 
                    embedding_features, 
                    activation=activation, 
                    idx=idx,
                    num_heads=num_heads,
                )
            )

    def forward(self, g, h, y=None, mask=None):
        h = self.fc_in(h)
        h = self.activation(h)
        h = self.dropout(h)

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
            with pyro.plate("data", y.shape[0], device=h.device):
                pyro.sample(
                    "y", 
                    pyro.distributions.OneHotCategorical(h), 
                    obs=y,
                )

        return h

    def guide(self, g, h, y=None, mask=None):
        h = self.fc_in(h)
        h = self.activation(h)

        for idx in range(self.depth):
            h = getattr(self, f"layer{idx}").guide(g, h)
            h = self.activation(h)

        return h
        