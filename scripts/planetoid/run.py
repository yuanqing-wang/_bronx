import numpy as np
import torch
import dgl
from bronx.models import BronxModel

def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.remove_self_loop(g)
    a = g.adj().to_dense()

    model = BronxModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].max() + 1,
        hidden_features=args.hidden_features,
        depth=args.depth,
        residual=args.residual,
        semantic_weight=args.semantic_weight,
        num_heads=args.num_heads,
        a_h_drop=args.a_h_dropout,
        a_x_drop=args.a_x_dropout,
        fc_drop=args.fc_dropout,
        epsilon=args.epsilon,
    )

    if torch.cuda.is_available():
        a = a.cuda()
        model = model.cuda()
        g = g.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    accuracy_vl = []
    accuracy_te = []

    # import tqdm
    for _ in range(1000):
        model.train()
        optimizer.zero_grad()
        y_hat = model(g.ndata['feat'], a)[g.ndata['train_mask']]
        y = g.ndata['label'][g.ndata['train_mask']]
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        loss.backward()
        optimizer.step()
        model.eval()

        with torch.no_grad():
            y_hat = model(g.ndata["feat"], a)[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy = float((y_hat.argmax(-1) == y).sum()) / len(y_hat)
            accuracy_vl.append(accuracy)

            y_hat = model(g.ndata["feat"], a)[g.ndata["test_mask"]]
            y = g.ndata["label"][g.ndata["test_mask"]]
            accuracy = float((y_hat.argmax(-1) == y).sum()) / len(y_hat)
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
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--residual", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--semantic_weight", type=float, default=-1.0)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--a_h_dropout", type=float, default=0.0)
    parser.add_argument("--a_x_dropout", type=float, default=0.0)
    parser.add_argument("--fc_dropout", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()
    print(args)
    run(args)
