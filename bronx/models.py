import math
import torch
import dgl
import pyro
from pyro import poutine
from .layers import BronxLayer, NodeRecover, EdgeRecover, NeighborhoodRecover
from dgl.nn.pytorch import GraphConv

class BronxModel(pyro.nn.PyroModule):
    def __init__(
            self, in_features, hidden_features, out_features, 
            embedding_features=None,
            activation=torch.nn.SiLU(),
            depth=1,
            readout_depth=1,
            num_heads=4,
            sigma_factor=1.0,
            kl_scale=1.0,
            t=1.0,
            alpha=0.1,
            adjoint=False,
            physique=False,
            gamma=1.0,
            dropout_in=0.0,
            dropout_out=0.0,
            edge_recover_scale=1.0,
        ):
        super().__init__()
        if embedding_features is None:
            embedding_features = hidden_features
        self.fc_in = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_out = torch.nn.Linear(hidden_features, out_features, bias=False)

        fc_out = []
        for idx in range(readout_depth-1):
            fc_out.append(activation)
            fc_out.append(
                torch.nn.Linear(hidden_features, hidden_features, bias=False)
            )
        fc_out.append(activation)
        fc_out.append(
            torch.nn.Linear(hidden_features, out_features, bias=False)
        )
        self.fc_out = torch.nn.Sequential(*fc_out)

        self.alpha = alpha
        self.log_alpha = torch.nn.Parameter(
            torch.ones(hidden_features) * math.log(alpha)
        )
        self.activation = activation
        self.depth = depth

        for idx in range(depth):
            layer = BronxLayer(
                hidden_features,
                embedding_features,
                activation=activation,
                idx=idx,
                num_heads=num_heads,
                sigma_factor=sigma_factor,
                kl_scale=kl_scale,
                t=t/depth,
                adjoint=adjoint,
                physique=physique,
                gamma=gamma,
            )
            
            if idx > 0:
                layer.fc_mu = self.layer0.fc_mu
                layer.fc_log_sigma = self.layer0.fc_log_sigma

            setattr(
                self,
                f"layer{idx}",
                layer,
            )

        self.edge_recover = EdgeRecover(
            hidden_features,
            hidden_features,
            scale=edge_recover_scale,
        )
        self.dropout_in = torch.nn.Dropout(dropout_in)
        self.dropout_out = torch.nn.Dropout(dropout_out)

    def guide(self, g, h, *args, **kwargs):
        g = g.local_var()
        h = self.fc_in(h)        
        h = self.dropout_in(h)
        for idx in range(self.depth):
            h = getattr(self, f"layer{idx}").guide(g, h)
        h = self.dropout_out(h)
        return h

    def forward(self, g, h, *args, **kwargs):
        g = g.local_var()
        h0 = h
        h = self.fc_in(h)
        h = self.dropout_in(h)
        for idx in range(self.depth):
            h = getattr(self, f"layer{idx}")(g, h)
        h = self.dropout_out(h) 
        self.edge_recover(g, h)
        h = self.fc_out(h)
        return h

class NodeClassificationBronxModel(BronxModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, g, h, y=None, mask=None):
        h = super().forward(g, h, )
        # h = h.softmax(-1)
        if mask is not None:
            h = h[..., mask, :]
            if y is not None:
                y = y[..., mask, :]

        if y is not None:
            with pyro.plate("data", y.shape[0], device=h.device):
                pyro.sample(
                    "y",
                    pyro.distributions.OneHotCategorical(logits=h),
                    obs=y,
                )

        return h


class GraphRegressionBronxModel(BronxModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc_mu = torch.nn.Linear(
            kwargs["hidden_features"], kwargs["out_features"], bias=False,
        )
        self.fc_log_sigma = torch.nn.Linear(
            kwargs["hidden_features"], kwargs["out_features"], bias=False,
        )


    def forward(self, g, h, y=None):
        g = g.local_var()

        h = self.fc_in(h)
        h = self.activation(h)

        for idx in range(self.depth):
            h = getattr(self, f"layer{idx}")(g, h)
            h = self.activation(h)

        parallel = h.dim() == 3
        if parallel:
            h = h.swapaxes(0, 1)
            
        g.ndata["h"] = h
        h = dgl.sum_nodes(g, "h")

        if parallel:
            h = h.swapaxes(0, 1)

        if y is not None:
            with pyro.plate("data", y.shape[0], device=h.device):
                pyro.sample(
                    "y",
                    pyro.distributions.Normal(
                        self.fc_mu(h), 
                        self.fc_log_sigma(h).exp()
                    ).to_event(1),
                    obs=y,
                )
        return h

        
