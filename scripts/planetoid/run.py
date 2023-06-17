import numpy as np
import torch
import pyro
from pyro import poutine
import dgl
dgl.use_libxsmm(False)
from bronx.models import LinearDiffusionModel, BronxModel

def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.remove_self_loop(g)
    src, dst = g.edges()
    eids = torch.where(src > dst)[0]
    g = dgl.remove_edges(g, eids)
    g.ndata["label"] = torch.nn.functional.one_hot(g.ndata["label"])

    model = BronxModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].shape[-1],
        hidden_features=args.hidden_features,
        embedding_features=args.embedding_features,
        dropout=args.dropout,
        depth=args.depth,
        edge_drop=args.edge_drop,
        num_heads=args.num_heads,
    )

    if torch.cuda.is_available():
        # a = a.cuda()
        model = model.cuda()
        g = g.to("cuda:0")

    optimizer = torch.optim.Adam # ({"lr": args.learning_rate, "weight_decay": args.weight_decay})
    scheduler = pyro.optim.ReduceLROnPlateau(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": args.learning_rate, "weight_decay": args.weight_decay},
            "patience": args.patience,
            "factor": args.factor,
            "mode": "max",
        }
    )

    svi = pyro.infer.SVI(
        model, model.guide, scheduler, 
        loss=pyro.infer.TraceMeanField_ELBO(num_particles=args.num_particles, vectorize_particles=True)
    )

    accuracy_vl = []
    accuracy_te = []

    for idx in range(500):
        model.train()
        loss = svi.step(g, g.ndata["feat"], g.ndata["label"], g.ndata["train_mask"])
        model.eval()

        with torch.no_grad():
            predictive = pyro.infer.Predictive(
                model, guide=model.guide, num_samples=args.num_samples, parallel=True,
                return_sites=["_RETURN"],
            )
            
            y_hat = predictive(g, g.ndata["feat"], mask=g.ndata["val_mask"])["_RETURN"].mean(0)
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy = float((y_hat.argmax(-1) == y.argmax(-1)).sum()) / len(y_hat)
            accuracy_vl.append(accuracy)
            scheduler.step(accuracy)
            print(accuracy, loss)

            y_hat = predictive(g, g.ndata["feat"], mask=g.ndata["test_mask"])["_RETURN"].mean(0)
            y = g.ndata["label"][g.ndata["test_mask"]]
            accuracy = float((y_hat.argmax(-1) == y.argmax(-1)).sum()) / len(y_hat)
            accuracy_te.append(accuracy)

    accuracy_vl = np.array(accuracy_vl)
    accuracy_te = np.array(accuracy_te)

    print(accuracy_vl.max(), accuracy_te[accuracy_vl.argmax()])

    import pandas as pd
    df = vars(args)
    df["accuracy_vl"] = accuracy_vl.max()
    df["accuracy_te"] = accuracy_te[accuracy_vl.argmax()]
    df = pd.DataFrame.from_dict([df])
    import os
    header = not os.path.exists("performance.csv")
    df.to_csv("performance.csv", mode="a", header=header)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--embedding_features", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--edge_drop", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_particles", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=1)
    args = parser.parse_args()
    print(args)
    run(args)
