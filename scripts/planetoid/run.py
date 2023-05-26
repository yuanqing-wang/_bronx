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
    # g = dgl.add_self_loop(g)
    g.ndata["label"] = torch.nn.functional.one_hot(g.ndata["label"])

    model = BronxModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].shape[-1],
        hidden_features=args.hidden_features,
        embedding_features=args.embedding_features,
        gamma=args.gamma,
        dropout=args.dropout,
        depth=args.depth,
        edge_drop=args.edge_drop,
        num_heads=args.num_heads,
    )

    if torch.cuda.is_available():
        # a = a.cuda()
        model = model.cuda()
        g = g.to("cuda:0")


    def is_weight(x):

        result = "weight" in x["name"]
        print(x["name"], result)
        return result

    # is_weight = lambda x: "weight" in x["name"]
    guide = pyro.infer.autoguide.AutoGuideList(model)   
    # guide.add(poutine.block(pyro.infer.autoguide.AutoNormal(model), expose_fn=is_weight))
    # guide.add(poutine.block(pyro.infer.autoguide.guides.AutoCallable(model, model.guide), hide_fn=is_weight))
    guide.add(model.guide)


    optimizer = pyro.optim.Adam({"lr": args.learning_rate, "weight_decay": args.weight_decay})
    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO(num_particles=1))

    accuracy_vl = []
    accuracy_te = []

    for idx in range(50):
        model.train()
        loss = svi.step(g, g.ndata["feat"], g.ndata["label"], g.ndata["train_mask"])
        print(loss)
        model.eval()

        # with torch.no_grad():
        #     y_hat = torch.stack([
        #         poutine.replay(
        #             model, 
        #             poutine.trace(guide).get_trace(g, g.ndata["feat"], mask=g.ndata["val_mask"]),
        #         )(g, g.ndata["feat"], mask=g.ndata["val_mask"])
        #         for _ in range(args.num_samples)
        #     ]).mean(0)

        #     y = g.ndata["label"][g.ndata["val_mask"]]
        #     accuracy = float((y_hat.argmax(-1) == y.argmax(-1)).sum()) / len(y_hat)
        #     accuracy_vl.append(accuracy)
        #     print(accuracy, flush=True)

        #     y_hat = torch.stack([
        #         poutine.replay(
        #             model, 
        #             poutine.trace(guide).get_trace(g, g.ndata["feat"], mask=g.ndata["test_mask"]),
        #         )(g, g.ndata["feat"], mask=g.ndata["test_mask"])
        #         for _ in range(args.num_samples)
        #     ]).mean(0)

        #     y = g.ndata["label"][g.ndata["test_mask"]]
        #     accuracy = float((y_hat.argmax(-1) == y.argmax(-1)).sum()) / len(y_hat)
        #     accuracy_te.append(accuracy)

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
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--embedding_features", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--edge_drop", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()
    print(args)
    run(args)
