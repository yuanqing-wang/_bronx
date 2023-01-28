import torch
import pyro
from pyro import poutine
from .layers import BronxLayer

class BronxModel(pyro.nn.PyroModule):
    def __init__(self, in_features, hidden_features, out_features, depth, num_heads=1):
        super().__init__()
        self.fc_in = pyro.nn.PyroModule[torch.nn.Linear](in_features, hidden_features, bias=False)
        self.fc_out = pyro.nn.PyroModule[torch.nn.Linear](hidden_features, out_features, bias=False)
        self.depth = depth
        for idx in range(depth):
            setattr(
                self,
                f"layer{idx}",
                BronxLayer(hidden_features, hidden_features, index=idx)
            )

    def model(self, g, h, y=None, mask=None):
        g = g.local_var()
        h = self.fc_in(h)
        for idx in range(self.depth):
            layer = getattr(self, f"layer{idx}")
            h = layer.model(g, h)
        h = self.fc_out(h).softmax(-1)
        if mask is not None:
            h = h[mask]

            if y is not None:
                y = y[mask]

        if y is not None: # training
            h = pyro.sample(
                "obs",
                pyro.distributions.OneHotCategorical(h).to_event(1),
                obs=y,
            )
        return h
        
    def guide(self, g, h, y=None, mask=None):
        g = g.local_var()
        h = self.fc_in(h)
        for idx in range(self.depth):
            layer = getattr(self, f"layer{idx}")
            e = layer.guide(g, h)
            h = layer.mp(g, h, e)

        

