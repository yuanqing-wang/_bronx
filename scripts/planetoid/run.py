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
    )

    if torch.cuda.is_available():
        a = a.cuda()
        model = model.cuda()
        g = g.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    best_accuracy = 0.0

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
            best_accuracy = max(best_accuracy, accuracy)

    print(best_accuracy)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--residual", type=bool, default=True)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    args = parser.parse_args()
    print(args)
    run(args)
