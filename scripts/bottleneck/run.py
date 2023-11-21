import dgl
import torch
import pyro
from bronx.models import BronxModel

class NodeClassificationBronxModel(BronxModel):
    def __init__(self, *args, **kwargs):
        temperature = kwargs.pop("consistency_temperature", 1.0)
        factor = kwargs.pop("consistency_factor", 1.0)
        super().__init__(*args, **kwargs)
        
    def forward(self, g, h, y=None, mask=None):
        h = super().forward(g, h, )
        h = h.softmax(-1)

        if mask is not None:
            h = h[..., mask, :]

        if y is not None:
            with pyro.plate(
                "data", y.shape[0], 
                device=h.device, 
            ):
                pyro.sample(
                    "y",
                    pyro.distributions.OneHotCategorical(probs=h),
                    obs=y,
                )

        return h 

def run(args):
    graphs, labels = dgl.load_graphs(args.data)
    y = labels["y"]
    y = torch.nn.functional.one_hot(y).float()
    g = dgl.batch(graphs)
    g = dgl.remove_self_loop(g)
    g = dgl.add_reverse_edges(g)
    g.ndata["h0"] = torch.nn.functional.one_hot(g.ndata["h"][:, 0])
    g.ndata["h1"] = torch.nn.functional.one_hot(g.ndata["h"][:, 1])
    g.ndata["h"] = torch.cat([g.ndata["h0"], g.ndata["h1"]], dim=-1).float()

    model = NodeClassificationBronxModel(
        in_features=g.ndata["h"].shape[-1],
        out_features=y.shape[-1],
        hidden_features=args.hidden_features,
        embedding_features=args.embedding_features,
        num_heads=args.num_heads,
        t=args.t,
        activation=getattr(torch.nn, args.activation)(),
        sigma_factor=1e-10,
        kl_scale=1e-10,
        depth=1,
    )

    if torch.cuda.is_available():
        model = model.to("cuda:0")
        g = g.to("cuda:0")
        y = y.to("cuda:0")

    optimizer = pyro.optim.Adam(
        {
            "lr": args.learning_rate, 
            "weight_decay": args.weight_decay
        },
    )

    svi = pyro.infer.SVI(
        model,
        model.guide,
        optimizer,
        loss=pyro.infer.TraceMeanField_ELBO(
            num_particles=args.num_particles, 
            vectorize_particles=True,
        ),
    )

    for epoch in range(args.epochs):
        model.train()
        loss = svi.step(g, g.ndata["h"], y=y, mask=g.ndata["mask"])

        model.eval()
        predictive = pyro.infer.Predictive(
            model,
            guide=model.guide,
            num_samples=1,
            parallel=True,
            return_sites=["_RETURN"],
        )


        with torch.no_grad():
            y_pred = predictive(g, g.ndata["h"], mask=g.ndata["mask"])
            y_pred = y_pred["_RETURN"].mean(0)
            y_pred = y_pred.argmax(-1)
            y_true = y.argmax(-1)
            accuracy = (y_pred == y_true).float().mean()
            print(f"Epoch: {epoch}, Accuracy: {accuracy}, Loss: {loss}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--hidden_features", type=int, default=128)
    parser.add_argument("--embedding_features", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--activation", type=str, default="ELU")
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--num_particles", type=int, default=1)

    args = parser.parse_args()
    run(args)