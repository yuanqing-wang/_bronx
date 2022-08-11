import numpy as np
import scipy.sparse as sp
import torch
import dgl
from sklearn.metrics import average_precision_score
from bronx.layers import Attention
from bronx.utils import EarlyStopping
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from sklearn.metrics import roc_auc_score, average_precision_score

def run(args):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.remove_self_loop(g)
    degrees = g.in_degrees()
    h = g.ndata['feat']
    h = 1.0 * (h > 0.0)
    h_rw = dgl.random_walk_pe(g, 16)
    h = torch.cat([h, h_rw], -1)
    g.ndata["feat"] = h

    g = dgl.add_self_loop(g)
    a = g.adj()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_in = torch.nn.Linear(h.shape[-1], 64, bias=False)
            self.attn0 = Attention(64, 8)
            self.attn1 = Attention(64, 8)

        def forward(self, h):
            h = self.embedding_in(h)
            h, a_hat = self.attn0(h)
            h, a_hat = self.attn1(h)
            return a_hat

    a_ref = g.adj(scipy_fmt="coo")
    a_ref = a_ref - sp.dia_matrix((a_ref.diagonal()[np.newaxis, :], [0]), shape=a_ref.shape)
    a_ref.eliminate_zeros()
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, train_edges_false = mask_test_edges(a_ref)
    a = preprocess_graph(adj_train).to_dense()

    model = Model()
    if torch.cuda.is_available():
        a = a.cuda()
        h = h.cuda()
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), 1e-2, weight_decay=1e-10)

    import tqdm
    for idx_range in range(200):
        model.train()
        optimizer.zero_grad()
        a_hat = (model(h) * degrees).fill_diagonal_(1.0)
        pos_weight = (a.shape[0] * a.shape[0] - a.sum()) / a.sum()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=a_hat,
            target=a,
            pos_weight=pos_weight,
        )
        print(a_hat)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            a_hat = (model(h) * degrees).fill_diagonal_(1.0)

        roc_score, ap_score = get_roc_score(a_hat, a_ref, train_edges, train_edges_false)
        # roc_score, ap_score = get_roc_score(a_hat, a_ref, test_edges, test_edges_false)
        print(roc_score, ap_score)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    args = parser.parse_args()
    run(args)
