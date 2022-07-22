import torch
import dgl
from bronx.layers import Bronx
from bronx.utils import EarlyStopping


def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.add_self_loop(g)
    a = g.adj()

    model = Bronx(g.ndata['feat'].shape[-1], 16, g.ndata['label'].shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    early_stopping = EarlyStopping(10)

    import tqdm
    for _ in tqdm.tqdm(range(500)):
        model.train()
        optimizer.zero_grad()
        y_hat = model(a, g.ndata['feat'])[g.ndata['train_mask']]
        y = g.ndata['label'][g.ndata['train_mask']]
        loss = torch.nn.CrossEntropyLoss()(y_hat, y) + model.loss_vae(a, g.ndata['feat'])
        loss.backward()
        optimizer.step()
        model.eval()

        with torch.no_grad():
            y_hat = model(a, g.ndata["feat"])[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy = float((y_hat.argmax(-1) == y).sum()) / len(y_hat)
            print(accuracy)
            if early_stopping([loss, -accuracy], model) is True:
                model.load_state_dict(early_stopping.best_state)
                break

    model.eval()
    with torch.no_grad():
        y_hat = model(a, g.ndata["feat"])[g.ndata["test_mask"]]
        y = g.ndata["label"][g.ndata["test_mask"]]
        accuracy = float((y_hat.argmax(-1) == y).sum()) / len(y_hat)
        print(accuracy)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    args = parser.parse_args()
    run(args)
