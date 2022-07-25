import torch
import dgl
from bronx.layers import Bronx
from bronx.utils import EarlyStopping


def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.add_self_loop(g)
    a = g.adj()

    model = Bronx(
        g.ndata['feat'].shape[-1],
        args.hidden_features,
        g.ndata['label'].shape[-1],
        neighborhood_size=args.neighborhood_size,
    )

    if torch.cuda.is_available():
        a = a.cuda()
        model = model.cuda()
        g = g.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=0.001)
    early_stopping = EarlyStopping(args.early_stopping)

    # import tqdm
    for _ in range(1000):
        model.train()
        optimizer.zero_grad()
        y_hat = model(a, g.ndata['feat'], n_samples=args.n_samples)[g.ndata['train_mask']]
        y = g.ndata['label'][g.ndata['train_mask']]
        loss = torch.nn.CrossEntropyLoss()(y_hat, y) + model.loss_vae(a, g.ndata['feat'])
        loss.backward()
        optimizer.step()
        model.eval()

        with torch.no_grad():
            y_hat = model(a, g.ndata["feat"], n_samples=args.n_samples)[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy = float((y_hat.argmax(-1) == y).sum()) / len(y_hat)
            if early_stopping([loss, -accuracy], model) is True:
                model.load_state_dict(early_stopping.best_state)
                break

    model.eval()
    with torch.no_grad():
        y_hat = model(a, g.ndata["feat"])[g.ndata["test_mask"]]
        y = g.ndata["label"][g.ndata["test_mask"]]
        accuracy = float((y_hat.argmax(-1) == y).sum()) / len(y_hat)
        print(accuracy)
    print(args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    parser.add_argument("--hidden_features", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--neighborhood_size", type=int, default=3)
    args = parser.parse_args()
    run(args)
