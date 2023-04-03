import numpy as np
import torch
import pyro
from pyro import poutine
import dgl
dgl.use_libxsmm(False)
from bronx.models import BronxModel
from bronx.layers import BronxLayer
from bronx.utils import personalized_page_rank


def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g.ndata["label"] = torch.nn.functional.one_hot(g.ndata["label"])

    model = BronxModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].shape[-1],
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_heads=args.num_heads,
        # scale=float(g.ndata["train_mask"].sum() / g.number_of_nodes()),
        # scale=args.scale,
        # bayesian_weights=False,
    )

    if torch.cuda.is_available():
        # a = a.cuda()
        model = model.cuda()
        g = g.to("cuda:0")

    # guide = pyro.infer.autoguide.AutoGuideList(model)    
    # guide.add(pyro.infer.autoguide.AutoNormal(poutine.block(model, hide_fn=lambda x: "weight" not in x["name"])))
    # guide.add(poutine.block(model.guide, expose_fn=lambda x: x["name"].startswith("e")))

    guide = model.guide

    optimizer = pyro.optim.Adam({"lr": args.learning_rate, "weight_decay": args.weight_decay})
    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO(num_particles=4))
    accuracy_vl = []
    accuracy_te = []

    import tqdm
    for idx in range(3000):
        loss = svi.step(g, g.ndata["feat"], g.ndata["label"], g.ndata["train_mask"])

        if idx % 10 != 0:
            continue

        with torch.no_grad():
            # predictive = pyro.infer.Predictive(
            #     model, guide=guide, num_samples=4, 
            #     return_sites=["_RETURN"], parallel=False,
            # )
            
            y_hat = torch.stack([
                poutine.replay(
                    model, 
                    poutine.trace(guide).get_trace(g, g.ndata["feat"], mask=g.ndata["val_mask"]),
                )(g, g.ndata["feat"], mask=g.ndata["val_mask"])
                for _ in range(args.n_samples)
            ]).mean(0)

            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy = float((y_hat.argmax(-1) == y.argmax(-1)).sum()) / len(y_hat)
            accuracy_vl.append(accuracy)
            print(accuracy, flush=True)

            y_hat = torch.stack([
                poutine.replay(
                    model, 
                    poutine.trace(guide).get_trace(g, g.ndata["feat"], mask=g.ndata["test_mask"]),
                )(g, g.ndata["feat"], mask=g.ndata["test_mask"])
                for _ in range(args.n_samples)
            ]).mean(0)

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
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    print(args)
    run(args)
