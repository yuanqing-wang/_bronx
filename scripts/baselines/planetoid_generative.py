import torch
import dgl
from sklearn.metrics import average_precision_score
from bronx.layers import GAE
from bronx.utils import EarlyStopping

def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.add_self_loop(g)
    a = g.adj()
    h = g.ndata['feat']
    h = 1.0 * (h > 0.0)
    model = GAE(g.ndata['feat'].shape[-1], 32, 16)

    if torch.cuda.is_available():
        a = a.cuda()
        h = h.cuda()
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), 1e-2)

    import tqdm
    for idx_range in range(10000):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(a, h)
        loss.backward()
        optimizer.step()

        if idx_range % 100 == 0:
            with torch.no_grad():
                model.eval()
                a_hat = model(a, h)
                # a_hat = 1.0 * (a_hat > 0.5)
                # accuracy = (a_hat.flatten() == a.to_dense().flatten()).sum() / a_hat.numel()
                # print(accuracy)

                ap = average_precision_score(a.to_dense().flatten().cpu(), a_hat.flatten().cpu())
                print(ap)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    args = parser.parse_args()
    run(args)
