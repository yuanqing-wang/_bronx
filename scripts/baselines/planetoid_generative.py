import torch
import dgl
from brnx.layers import VGAE
from brnx.utils import EarlyStopping

def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.add_self_loop(g)
    a = g.adj()
    model = VGAE(g.ndata['feat'].shape[-1], 32, 16)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    h = g.ndata['feat']

    import tqdm
    for _ in range(1000):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(a, h)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            p_a = model(a, h)
            a_hat = p_a.sample()
            accuracy = (a_hat == a.to_dense()).sum() / a_hat.numel()
            print(accuracy)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    args = parser.parse_args()
    run(args)
