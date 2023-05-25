from collections import OrderedDict
import torch
import pyro
from pyro import poutine
from .layers import LinearDiffusion, BronxLayer

import functools

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class LinearDiffusionModel(torch.nn.Module):
    def __init__(
            self, in_features, hidden_features, out_features,
            activation=torch.nn.SiLU(), gamma=0.0,
        ):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features, hidden_features)
        self.fc_out = torch.nn.Linear(hidden_features, out_features)
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
            activation=torch.nn.SiLU(), gamma=0.0,
            depth=2,
            dropout=0.0,
            edge_drop=0.0,
            num_heads=4,
        ):
        super().__init__()
        if embedding_features is None:
            embedding_features = hidden_features
        self.fc_in = pyro.nn.PyroModule[torch.nn.Linear](in_features, hidden_features)
        self.fc_out = pyro.nn.PyroModule[torch.nn.Linear](hidden_features, out_features)
        self.activation = activation
        self.gamma = gamma
        self.depth = depth

        for idx in range(depth):
            setattr(
                self, 
                f"layer{idx}", 
                BronxLayer(
                    hidden_features, 
                    embedding_features, 
                    activation=activation, 
                    idx=idx,
                    dropout=dropout,
                    edge_drop=edge_drop,
                    num_heads=num_heads,
                )
            )

        self.prior()

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

    def forward(self, g, h, y=None, mask=None):
        h = self.fc_in(h)
        h = self.activation(h)

        for idx in range(self.depth):
            h = getattr(self, f"layer{idx}")(g, h)
            h = self.activation(h)

        h = self.fc_out(h)
        h = h.softmax(-1)
        if mask is not None:
            h = h[mask]
            if y is not None:
                y = y[mask]

        if y is not None:
            with pyro.plate("data", h.shape[0], device=h.device):
                pyro.sample(
                    "y", 
                    pyro.distributions.OneHotCategorical(h).to_event(1), 
                    obs=y,
                )

        return h

    def guide(self, g, h, y=None, mask=None):
        h = self.fc_in(h)
        h = self.activation(h)

        for idx in range(self.depth):
            e = getattr(self, f"layer{idx}").guide(g, h)
            h = getattr(self, f"layer{idx}").mp(g, h, e)
            h = self.activation(h)

        return h

    def combined_guide(self, g, h, y=None, mask=None):
        edge_guide = poutine.block(self.guide, expose=lambda x: x.startswith("e"))
        weight_guide = poutine.block(self.guide, hide=lambda x: x.startswith("e"))
        combined_guide = pyro.infer.autoguide.guides.AutoGuideList(self)
        combined_guide.append(edge_guide)
        combined_guide.append(weight_guide)
        return combined_guide