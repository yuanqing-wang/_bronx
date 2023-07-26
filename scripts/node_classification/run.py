import numpy as np
import torch
import pyro
from pyro import poutine
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)
from pyro.contrib.gp.likelihoods.multi_class import MultiClass
from bronx.kernels import GraphLinearDiffusion
from bronx.models import GraphVariationalSparseGaussianProcess
from pyro.contrib.gp.models.vsgp import VariationalSparseGP
from pyro.contrib.gp.kernels.dot_product import Linear

def run(args):
    pyro.clear_param_store()
    torch.cuda.empty_cache()
    from dgl.data import (
        CoraGraphDataset,
        CiteseerGraphDataset,
        PubmedGraphDataset,
        CoauthorCSDataset,
        CoauthorPhysicsDataset,
        AmazonCoBuyComputerDataset,
        AmazonCoBuyPhotoDataset,
    )

    g = locals()[args.data](verbose=False)[0]
    g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)
    src, dst = g.edges()
    eids = torch.where(src > dst)[0]
    g = dgl.remove_edges(g, eids)

    if "train_mask" not in g.ndata:
        g.ndata["train_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        g.ndata["val_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        g.ndata["test_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)

        train_idxs = torch.tensor([], dtype=torch.int32)
        val_idxs = torch.tensor([], dtype=torch.int32)
        test_idxs = torch.tensor([], dtype=torch.int32)

        n_classes = g.ndata["label"].shape[-1]
        for idx_class in range(n_classes):
            idxs = torch.where(g.ndata["label"][:, idx_class] == 1)[0]
            assert len(idxs) > 50
            idxs = idxs[torch.randperm(len(idxs))]
            _train_idxs = idxs[:20]
            _val_idxs = idxs[20:50]
            _test_idxs = idxs[50:]
            train_idxs = torch.cat([train_idxs, _train_idxs])
            val_idxs = torch.cat([val_idxs, _val_idxs])
            test_idxs = torch.cat([test_idxs, _test_idxs])

        g.ndata["train_mask"][train_idxs] = True
        g.ndata["val_mask"][val_idxs] = True
        g.ndata["test_mask"][test_idxs] = True

    if torch.cuda.is_available():
        g = g.to("cuda:0")

    kernel = GraphLinearDiffusion(1)
    Xu = torch.arange(g.number_of_nodes(), dtype=torch.int32)
    likelihood = MultiClass(num_classes=g.ndata["label"].max()+1)
    model = GraphVariationalSparseGaussianProcess(
        g, 
        torch.where(g.ndata["train_mask"])[0], 
        g.ndata["label"][g.ndata["train_mask"]], 
        kernel, 
        Xu,
        likelihood,
        latent_shape=(g.ndata["label"].max()+1,),
    )

    # kernel = Linear(g.ndata["feat"].shape[-1])
    # Xu = g.ndata["feat"]
    # likelihood = MultiClass(num_classes=g.ndata["label"].max()+1)
    # model = VariationalSparseGP(
    #     X=g.ndata["feat"],
    #     y=g.ndata["label"],
    #     kernel=kernel,
    #     Xu=Xu,
    #     likelihood=likelihood,
    #     latent_shape=(g.ndata["label"].max()+1,),
    # )

    if torch.cuda.is_available():
        # a = a.cuda()
        model = model.cuda()

    optimizer = getattr(pyro.optim, args.optimizer)(
        {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    )

    # optimizer = SWA(
    #     {
    #         "base": getattr(torch.optim, args.optimizer),
    #         "base_args": {"lr": args.learning_rate, "weight_decay": args.weight_decay},
    #         "swa_args": {
    #             "swa_start": args.swa_start, 
    #             "swa_freq": args.swa_freq, 
    #             "swa_lr": args.swa_lr,
    #         },
    #     }
    # )

    svi = pyro.infer.SVI(
        model.model,
        model.guide,
        optimizer,
        loss=pyro.infer.TraceMeanField_ELBO(
            num_particles=args.num_particles, vectorize_particles=True
        ),
    )

    accuracies = []
    accuracies_te = []
    for idx in range(args.n_epochs):
        model.train()
        loss = svi.step()
        print(loss, flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--embedding_features", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--sigma_factor", type=float, default=10.0)
    parser.add_argument("--t", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=-1.0)
    parser.add_argument("--optimizer", type=str, default="RMSprop")
    parser.add_argument("--edge_recover_scale", type=float, default=1e-2)
    parser.add_argument("--node_recover_scale", type=float, default=1e-2)
    parser.add_argument("--kl_scale", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--swa_start", type=int, default=20)
    parser.add_argument("--swa_freq", type=int, default=10)
    parser.add_argument("--swa_lr", type=float, default=1e-2)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--test", type=int, default=1)
    args = parser.parse_args()
    run(args)
