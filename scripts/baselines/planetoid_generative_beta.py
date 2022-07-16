import torch
import dgl
from sklearn.metrics import average_precision_score
from bronx.layers import BVGAE
from bronx.utils import EarlyStopping

def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.add_self_loop(g)
    a = g.adj()
    h = g.ndata['feat']
    model = BVGAE(g.ndata['feat'].shape[-1], 32, 16)

    if torch.cuda.is_available():
        a = a.cuda()
        h = h.cuda()
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), 1e-2)

    import tqdm
    for _ in range(10000):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(a, h)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            a_hat = (model(a, h).sample() > 0.5) * 1
            accuracy = (a_hat.flatten() == a.to_dense().flatten()).sum() / a_hat.numel()

            ap = average_precision_score(a_hat.flatten().cpu(), a.to_dense().flatten().cpu())
            print(ap, accuracy.item())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    args = parser.parse_args()
    run(args)
