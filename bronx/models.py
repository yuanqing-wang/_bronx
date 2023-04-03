from collections import OrderedDict
import torch
import pyro
from pyro import poutine
from .layers import BronxLayer

import functools

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class BronxModel(pyro.nn.PyroModule):
    def __init__(self, in_features, hidden_features, out_features, depth, num_heads=1, activation
=torch.nn.SiLU()):
        super().__init__()
        self.fc_in = pyro.nn.PyroModule[torch.nn.Linear](in_features, hidden_features, bias=False)
        self.fc_out = pyro.nn.PyroModule[torch.nn.Linear](hidden_features, out_features, bias=False)
        self.depth = depth
        self.activation = activation
        for idx in range(depth):
            _in_features = _out_features = hidden_features
            # if idx == 0: _in_features = in_features
            # if idx + 1 == self.depth: _out_features = out_features
            setattr(
                self,
                f"layer{idx}",
                BronxLayer(_in_features, _out_features, index=idx)
            )

        # self.prior()
    
    def prior(self):
        # specify the prior distribution for the parameters
        parameter_shapes = OrderedDict()

        for name, parameter in self.named_parameters():
            if "weight" in name:
                parameter_shapes[name] = parameter.shape

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        for name, shape in parameter_shapes.items():
                rsetattr(
                    self,
                    name,
                    pyro.nn.PyroSample(
                        pyro.distributions.Normal(
                            parameter.new_zeros(shape, device=device),
                            parameter.new_ones(shape, device=device),
                        ).to_event(len(shape)),
                    )
                )



    def model(self, g, h, y=None, mask=None):
        g = g.local_var()
        h = self.fc_in(h)
        for idx in range(self.depth):
            layer = getattr(self, f"layer{idx}")
            h = layer.model(g, h)
            h = self.activation(h)
        h = self.fc_out(h)# .softmax(-1)
        h = h.softmax(-1)
        if mask is not None:
            h = h[mask]

            if y is not None:
                y = y[mask]

        if y is not None: # training
            with pyro.plate("_obs", h.shape[0]):
                h = pyro.sample(
                    "obs",
                    pyro.distributions.OneHotCategorical(h).to_event(1),
                    obs=y,
                )
        return h
        
    forward = model

    def guide(self, g, h, y=None, mask=None):
        g = g.local_var()
        h = self.fc_in(h)
        for idx in range(self.depth):
            layer = getattr(self, f"layer{idx}")
            e = layer.guide(g, h)
            h = layer.mp(g, h, e)
            h = self.activation(h)

    # def combined_guide(self, g, h, y=None, mask=None):
    #     edge_guide = poutine.block(self.guide, expose=lambda x: x.startswith("e"))
    #     weight_guide = poutine.block(self.guide, hide=lambda x: x.startswith("e"))
    #     combined_guide = pyro.infer.autoguide.guides.AutoGuideList(self)
    #     combined_guide.append(edge_guide)
    #     combined_guide.append(weight_guide)
    #     return combined_guide

class BaselineModel(BronxModel):
    def forward(self, g, h, mask=None):
        g = g.local_var()
        h = self.fc_in(h)
        for idx in range(self.depth):
            layer = getattr(self, f"layer{idx}")
            h = layer.mp(g, h, e=None)
            h = self.activation(h)
        h = self.fc_out(h)# .softmax(-1)
        h = h.softmax(-1)
        if mask is not None:
            h = h[mask]
        return h
