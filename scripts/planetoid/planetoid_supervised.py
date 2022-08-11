import torch
import dgl
from bronx.models import Bronx
from bronx.utils import EarlyStopping

def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.remove_self_loop(g)
    h = g.ndata["feat"]
    h_rw = dgl.random_walk_pe(g, 32)
    h = torch.cat([h, h_rw], -1)
    g.ndata["feat"] = h

    g = dgl.add_self_loop(g)
    a = g.adj().to_dense()

    model = Bronx(
        in_features=g.ndata['feat'].shape[-1],
        out_features=7,
        hidden_features=16,
        depth=2,
        num_heads=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    early_stopping = EarlyStopping(10)

    import tqdm
    for _ in range(500):
        model.train()
        optimizer.zero_grad()
        y_hat, a_hat = model(g.ndata['feat'], return_adj=True)
        y_hat = y_hat[g.ndata['train_mask']]
        y = g.ndata['label'][g.ndata['train_mask']]
        loss_nll = torch.nn.CrossEntropyLoss()(y_hat, y)
        loss_rst = model.reconstruction_loss(a_hat, a)
        loss = loss_nll + 0.01 * loss_rst
        loss.backward()
        optimizer.step()
        model.eval()

        with torch.no_grad():
            y_hat = model(g.ndata["feat"])[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]

            # y_hat = model(g.ndata["feat"])[g.ndata["train_mask"]]
            # y = g.ndata["label"][g.ndata["train_mask"]]

            accuracy = float((y_hat.argmax(-1) == y).sum()) / len(y_hat)
            print(accuracy)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            # if early_stopping([loss, -accuracy], model) is True:
            #     model.load_state_dict(early_stopping.best_state)
            #     break

    # model.eval()
    # with torch.no_grad():
    #     y_hat = model(g.ndata["feat"])[g.ndata["test_mask"]]
    #     y = g.ndata["label"][g.ndata["test_mask"]]
    #     accuracy = float((y_hat.argmax(-1) == y).sum()) / len(y_hat)
    #     print(accuracy)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    args = parser.parse_args()
    run(args)
