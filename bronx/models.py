import torch
import pyro
from pyro import poutine
from .layers import BronxLayer

class BronxModel(pyro.nn.PyroModule):
    def __init__(self, in_features, hidden_features, out_features, depth, num_heads=1):
        super().__init__()
        self.fc_in = pyro.nn.PyroModule[torch.nn.Linear](in_features, hidden_features, bias=False)
        self.fc_out = pyro.nn.PyroModule[torch.nn.Linear](hidden_features, out_features, bias=False)
        layers = []
        for idx in range(depth):
            layers.append(
                BronxLayer(hidden_features, hidden_features)
            )
        self.layers = layers

    def model(self, g, h, y=None):
        g = g.local_var()
        h = self.fc_in(h)
        for layer in self.layers:
            h = layer.model(g, h)
        h = self.fc_out(h).softmax(-1)
        h = pyro.sample(
            "obs",
            pyro.distributions.OneHotCategorical(h),
            obs=y,
        )
        return h
        
    def guide(self, g, h):
        g = g.local_var()
        h = self.fc_in(h)
        for layer in self.layers:
            e = layer.guide(g, h)
            h = poutine.condition(layer.model, {"e": e})(g, h)

        

