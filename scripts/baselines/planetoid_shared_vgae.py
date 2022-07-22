import numpy as np
import scipy.sparse as sp
import torch
import dgl
from sklearn.metrics import average_precision_score
from bronx.layers import SharedVariationalGraphAutoEncoder as VGAE
from bronx.utils import EarlyStopping
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score

def run(args):

    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.add_self_loop(g)
    a = g.adj()
    h = g.ndata['feat']
    h = 1.0 * (h > 0.0)
    model = VGAE(g.ndata['feat'].shape[-1], 16, 16)

    a_ref = g.adj(scipy_fmt="coo")
    a_ref = a_ref - sp.dia_matrix((a_ref.diagonal()[np.newaxis, :], [0]), shape=a_ref.shape)
    a_ref.eliminate_zeros()
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(a_ref)
    a = preprocess_graph(adj_train)

    if torch.cuda.is_available():
        a = a.cuda()
        h = h.cuda()
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), 1e-2)

    import tqdm
    for idx_range in range(200):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(a, h)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _h = model.encode(a, h).rsample()
    roc_score, ap_score = get_roc_score(_h, a_ref, test_edges, test_edges_false)
    print(roc_score, ap_score)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cora")
    args = parser.parse_args()
    run(args)
