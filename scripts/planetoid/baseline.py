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
        gamma=args.gamma,
    )

    if torch.cuda.is_available():
        # a = a.cuda()
        model = model.cuda()
        g = g.to("cuda:0")

    optimizer = torch.optim.Adam(
        model.parameters(), args.learning_rate, weight_decay=args.weight_decay,
    )

    accuracy_vl = []
    accuracy_te = []

    for idx in range(100):
        model.train()
        optimizer.zero_grad()
        y_hat = model(g, g.ndata["feat"])
        model.eval()
        y_hat_tr = y_hat[g.ndata["train_mask"]]
        y_hat_vl = y_hat[g.ndata["val_mask"]]
        y_hat_te = y_hat[g.ndata["test_mask"]]
        y_tr = g.ndata["label"][g.ndata["train_mask"]]
        y_vl = g.ndata["label"][g.ndata["val_mask"]]
        y_te = g.ndata["label"][g.ndata["test_mask"]]
        loss = torch.nn.CrossEntropyLoss()(y_hat_tr, y_tr.float())
        loss.backward()
        optimizer.step()

        accuracy = float((y_hat_vl.argmax(-1) == y_vl.argmax(-1)).sum()) / len(y_hat_vl)
        accuracy_vl.append(accuracy)
        print(accuracy)

        accuracy = float((y_hat_te.argmax(-1) == y_te.argmax(-1)).sum()) / len(y_hat_te)
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
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.0)
    args = parser.parse_args()
    print(args)
    run(args)
