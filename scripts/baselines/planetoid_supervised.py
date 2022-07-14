import torch
import dgl
from bronx.layers import GCN
from bronx.utils import EarlyStopping

class Model(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer0 = GCN(in_features, 16, activation=torch.nn.ReLU())
        self.layer1 = GCN(16, 7)
        self.dropout0 = torch.nn.Dropout(0.5)
        self.dropout1 = torch.nn.Dropout(0.5)

    def forward(self, a, h):
        h = self.dropout0(h)
        h = self.layer0(a, h)
        h = self.dropout1(h)
        h = self.layer1(a, h)
        return h

def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.add_self_loop(g)
    a = g.adj()

    model = Model(g.ndata['feat'].shape[-1])
    optimizer = torch.optim.Adam(
        [
            {"params": model.layer0.parameters(), "lr": 1e-2, "weight_decay": 5e-4},
            {"params": model.layer1.parameters(), "lr": 1e-2, "weight_decay": 0.0},
        ]
    )
    early_stopping = EarlyStopping(10)

    import tqdm
    for _ in tqdm.tqdm(range(500)):
        model.train()
        optimizer.zero_grad()
        y_hat = model(a, g.ndata['feat'])[g.ndata['train_mask']]
        y = g.ndata['label'][g.ndata['train_mask']]
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        loss.backward()
        optimizer.step()
        model.eval()

        with torch.no_grad():
            y_hat = model(a, g.ndata["feat"])[g.ndata["val_mask"]]
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    args = parser.parse_args()
    run(args)
