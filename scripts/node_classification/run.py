import numpy as np
import torch
import pyro
from pyro import poutine
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)
from bronx.models import NodeClassificationBronxModel
from ray.air import session
import warnings
warnings.filterwarnings("ignore")
from bronx.optim import SWA, swap_swa_sgd

def get_graph(data):
    from dgl.data import (
        CoraGraphDataset,
        CiteseerGraphDataset,
        PubmedGraphDataset,
        CoauthorCSDataset,
        CoauthorPhysicsDataset,
        AmazonCoBuyComputerDataset,
        AmazonCoBuyPhotoDataset,
    )

    g = locals()[data](verbose=False)[0]
    g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)
    # src, dst = g.edges()
    # eids = torch.where(src > dst)[0]
    # g = dgl.remove_edges(g, eids)
    g.ndata["label"] = torch.nn.functional.one_hot(g.ndata["label"])

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
    return g

def run(args):
    pyro.clear_param_store()
    torch.cuda.empty_cache()
    if args.seed > 0:
        torch.manual_seed(args.seed)

    g = get_graph(args.data)

    if args.k > 0:
        h_pe = dgl.random_walk_pe(g, k=args.k)
        g.ndata["feat"] = torch.cat([g.ndata["feat"], h_pe], dim=-1)

    model = NodeClassificationBronxModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].shape[-1],
        hidden_features=args.hidden_features,
        embedding_features=args.embedding_features,
        depth=args.depth,
        readout_depth=args.readout_depth,
        num_heads=args.num_heads,
        sigma_factor=args.sigma_factor,
        kl_scale=args.kl_scale,
        t=args.t,
        adjoint=bool(args.adjoint),
        activation=getattr(torch.nn, args.activation)(),
        physique=args.physique,
        gamma=args.gamma,
        dropout_in=args.dropout_in,
        dropout_out=args.dropout_out,
        norm=bool(args.norm),
    )
 
    if torch.cuda.is_available():
        # a = a.cuda()
        model = model.cuda()
        g = g.to("cuda:0")

    optimizer = getattr(torch.optim, args.optimizer)
    scheduler = pyro.optim.ReduceLROnPlateau(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": args.learning_rate, "weight_decay": args.weight_decay},
            "factor": args.factor,
            "patience": args.patience,
            "mode": "max"
        }
    )

    svi = pyro.infer.SVI(
        model,
        model.guide,
        scheduler,
        loss=pyro.infer.TraceMeanField_ELBO(
            num_particles=args.num_particles, vectorize_particles=True
        ),
    )

    for idx in range(args.n_epochs):
        model.train()
        loss = svi.step(
            g, g.ndata["feat"], y=g.ndata["label"], mask=g.ndata["train_mask"]
        )

        model.eval()
        with torch.no_grad():
            predictive = pyro.infer.Predictive(
                model,
                guide=model.guide,
                num_samples=args.num_samples,
                parallel=True,
                return_sites=["_RETURN"],
            )

            y_hat = predictive(g, g.ndata["feat"], mask=g.ndata["val_mask"])[
                "_RETURN"
            ].mean(0)
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy_vl = float((y_hat.argmax(-1) == y.argmax(-1)).sum()) / len(
                y_hat
            )
        scheduler.step(accuracy_vl)

        lr = next(iter(scheduler.get_state().values()))["optimizer"][
            "param_groups"
        ][0]["lr"]

        if lr <= 1e-6:
            break
            
    y_hat = predictive(g, g.ndata["feat"], mask=g.ndata["test_mask"])[
        "_RETURN"
    ].mean(0)
    y = g.ndata["label"][g.ndata["test_mask"]]
    accuracy_te = float((y_hat.argmax(-1) == y.argmax(-1)).sum()) / len(
        y_hat
    )

    if len(args.checkpoint) > 1:
        torch.save(model, args.checkpoint)

    print("ACCURACY,%.6f,%.6f" % (accuracy_vl, accuracy_te), flush=True)
    return accuracy_vl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    parser.add_argument("--hidden_features", type=int, default=25)
    parser.add_argument("--embedding_features", type=int, default=20)
    parser.add_argument("--activation", type=str, default="SiLU")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=5)
    parser.add_argument("--sigma_factor", type=float, default=10.0)
    parser.add_argument("--t", type=float, default=5.0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--kl_scale", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--adjoint", type=int, default=0)
    parser.add_argument("--physique", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--readout_depth", type=int, default=1)
    parser.add_argument("--dropout_in", type=float, default=0.0)
    parser.add_argument("--dropout_out", type=float, default=0.0)
    parser.add_argument("--norm", type=int, default=0)
    parser.add_argument("--subsample_size", type=int, default=100)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()
    run(args)
