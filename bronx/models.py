import math
import torch
import dgl
import pyro
from pyro import poutine
from .layers import BronxLayer, NodeRecover, EdgeRecover
from dgl.nn.pytorch import GraphConv

class BronxModel(pyro.nn.PyroModule):
    def __init__(
            self, in_features, hidden_features, out_features, 
            embedding_features=None,
            activation=torch.nn.SiLU(),
            depth=2,
            num_heads=4,
            sigma_factor=1.0,
            kl_scale=1.0,
            t=1.0,
            gamma=1.0,
            edge_recover_scale=1e-5,
            alpha=0.1,
        ):
        super().__init__()
        if embedding_features is None:
            embedding_features = hidden_features
        self.fc_in = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_out = torch.nn.Linear(depth * hidden_features, out_features, bias=False)
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

        # self.node_recover = NodeRecover(
        #     hidden_features, in_features, scale=node_recover_scale,
        # )
        self.edge_recover = EdgeRecover(
            hidden_features, embedding_features, scale=edge_recover_scale,
        )

    def guide(self, g, h, *args, **kwargs):
        g = g.local_var()
        h = self.fc_in(h)

        with pyro.plate("nodes", g.number_of_nodes(), device=h.device):
            epsilon = pyro.sample(
                "epsilon_in",
                pyro.distributions.Normal(
                    torch.ones(g.number_of_nodes(), h.shape[-1], device=h.device), 
                    torch.ones(g.number_of_nodes(), h.shape[-1], device=h.device) * self.log_alpha.exp(),
                ).to_event(1)
            )

        h = h * epsilon
        
        for idx in range(self.depth):
            h = getattr(self, f"layer{idx}").guide(g, h)

        return h

    def forward(self, g, h, *args, **kwargs):
        g = g.local_var()
        h0 = h
        h = self.fc_in(h)

        with pyro.plate("nodes", g.number_of_nodes(), device=h.device):
            epsilon = pyro.sample(
                "epsilon_in",
                pyro.distributions.Normal(
                    torch.ones(g.number_of_nodes(), h.shape[-1], device=h.device), 
                    self.alpha * torch.ones(g.number_of_nodes(), h.shape[-1], device=h.device),
                ).to_event(1)
            )

        h = h * epsilon
        
        hs = []
        for idx in range(self.depth):
            h = getattr(self, f"layer{idx}")(g, h)
            hs.append(h)
        
        self.edge_recover(g, self.activation(h))

        h = torch.cat(hs, dim=-1)
        h = self.activation(h)
        h = self.fc_out(h)
        return h

class NodeClassificationBronxModel(BronxModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, g, h, y=None, mask=None):
        h = super().forward(g, h, )
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

        
