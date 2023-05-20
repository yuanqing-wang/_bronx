#!/bin/bash
import numpy as np
import torch
import dgl
from bronx.models import BronxModel
from bronx.utils import personalized_page_rank

def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.remove_self_loop(g)
    # g.ndata["feat"] = torch.cat([g.ndata["feat"], dgl.laplacian_pe(g, 10)], dim=-1)
    # g = dgl.add_self_loop(g)

    model = BronxModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].max() + 1,
        hidden_features=args.hidden_features,
        num_heads=args.num_heads,
        dropout0=args.dropout0,
        dropout1=args.dropout1,
        gamma=args.gamma,
        gain=args.gain,
    )

    if torch.cuda.is_available():
        model = model.cuda()
        g = g.to("cuda:0")

    model.sde.graph = g
    model.sde.graph2 = dgl.khop_graph(g, 2)
    # model.sde.graph3 = dgl.khop_graph(g, 3)
    # model.sde.graph4 = dgl.khop_graph(g, 4)
    # model.sde.graph3 = dgl.khop_graph(g, 3)
    # model.sde.graph4 = dgl.khop_graph(g, 4)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.patience, factor=args.factor)
    accuracy_vl = []
    accuracy_te = []

    # import tqdm
    for idx_epoch in range(100):
        model.train()
        optimizer.zero_grad()
        y_hat, kl = model(g, g.ndata['feat'])
        y_hat = y_hat[g.ndata['train_mask']]
        kl = kl.squeeze(-2)[g.ndata['train_mask']]
        y = g.ndata['label'][g.ndata['train_mask']]
        kl = kl.mean()
        loss = torch.nn.CrossEntropyLoss()(y_hat, y) + kl
        loss.backward()
        optimizer.step()
        # scheduler.step()
        model.eval()

        with torch.no_grad():
            _y_hat = torch.stack([model(g, g.ndata["feat"])[0] for _ in range(4)], 0).mean(0)
            y_hat = _y_hat[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy = float((y_hat.argmax(-1) == y).sum()) / len(y_hat)
            print(accuracy, kl.item(), flush=True)
            accuracy_vl.append(accuracy)
            scheduler.step(accuracy)

            y_hat = _y_hat[g.ndata["test_mask"]]
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
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_heads", type=float, default=4)
    parser.add_argument("--dropout0", type=float, default=0.6)
    parser.add_argument("--dropout1", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--factor", type=float, default=0.4)
    parser.add_argument("--gain", type=float, default=0.1)
    args = parser.parse_args()
    print(args)
    run(args)
