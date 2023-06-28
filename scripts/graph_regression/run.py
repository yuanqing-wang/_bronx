import numpy as np
import torch
import pyro
from pyro import poutine
import dgl
dgl.use_libxsmm(False)
from bronx.models import GraphRegressionBronxModel

def run(args):
    pyro.clear_param_store()
    from dgl.data import ZINCDataset
    from dgl.dataloading import GraphDataLoader
    data_train = ZINCDataset(mode="train")
    data_val = ZINCDataset(mode="valid")
    data_test = ZINCDataset(mode="test")
    n_types = data_train.num_atom_types
    data_train = GraphDataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    data_val = GraphDataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    data_test = GraphDataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    g, y = next(iter(data_train))

    model = GraphRegressionBronxModel(
        in_features=n_types,
        out_features=1,
        hidden_features=args.hidden_features,
        embedding_features=args.embedding_features,
        depth=args.depth,
        num_heads=args.num_heads,
    )

    if torch.cuda.is_available():
        model = model.to("cuda:0")

    optimizer = (
        torch.optim.Adam
    )  # ({"lr": args.learning_rate, "weight_decay": args.weight_decay})
    scheduler = pyro.optim.ReduceLROnPlateau(
        {
            "optimizer": optimizer,
            "optim_args": {
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            "patience": args.patience,
            "factor": args.factor,
            "mode": "max",
        }
    )

    svi = pyro.infer.SVI(
        model,
        model.guide,
        scheduler,
        loss=pyro.infer.TraceMeanField_ELBO(
            num_particles=args.num_particles, vectorize_particles=True
        ),
    )

    for idx in range(5000):
        for g, y in data_train:
            if torch.cuda.is_available():
                g = g.to("cuda:0")
                y = y.to("cuda:0")
            model.train()
            loss = svi.step(
                g, 
                torch.nn.functional.one_hot(g.ndata["feat"], n_types).float(), 
                y,
            )
            print(loss)
            model.eval()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--embedding_features", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_particles", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    print(args)
    run(args)
