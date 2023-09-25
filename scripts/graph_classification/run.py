import random
from re import split
import numpy as np
import torch
import pyro
from pyro import poutine
import dgl
dgl.use_libxsmm(False)
from bronx.models import GraphClassificationBronxModel
from bronx.optim import SWA, swap_swa_sgd
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from torch.utils.data import DataLoader


def run(args):
    pyro.clear_param_store()
    dataset = DglGraphPropPredDataset(name=args.data)
    split_idx = dataset.get_idx_split()
    batch_size = args.batch_size
    if batch_size == -1:
        batch_size = len(split_idx["train"])
    data_train = DataLoader(
        dataset[split_idx["train"]], 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_dgl,
    )
    data_valid = DataLoader(
        dataset[split_idx["valid"]], 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_dgl,
    )
    data_test = DataLoader(
        dataset[split_idx["test"]], 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_dgl,
    )
    g, y = next(iter(data_train))

    model = GraphClassificationBronxModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=1,
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
        model = model.to("cuda:0")

    optimizer = SWA(
        {
            "base": getattr(torch.optim, args.optimizer),
            "base_args": {
                "lr": args.learning_rate / batch_size, 
                "weight_decay": args.weight_decay
            },
            "swa_args": {
                "swa_start": args.swa_start, 
                "swa_freq": args.swa_freq, 
                "swa_lr": args.swa_lr,
            },
        }
    )

    svi = pyro.infer.SVI(
        model,
        model.guide,
        optimizer,
        loss=pyro.infer.TraceMeanField_ELBO(
            num_particles=args.num_particles, vectorize_particles=True
        ),
    )

    import tqdm
    for idx in tqdm.tqdm(range(args.n_epochs)):
        for g, y in data_train:
            if torch.cuda.is_available():
                g = g.to("cuda:0")
                y = y.to("cuda:0")
            model.train()
            loss = svi.step(g, g.ndata["feat"].float(), y.float())
    

        model.eval()
        swap_swa_sgd(svi.optim)
        ys = []
        ys_hat = []
        with torch.no_grad():
            for g, y in data_valid:
                ys.append(y)
                if torch.cuda.is_available():
                    g = g.to("cuda:0")
                    y = y.to("cuda:0")
                predictive = pyro.infer.Predictive(
                    model,
                    guide=model.guide,
                    num_samples=args.num_samples,
                    parallel=True,
                    return_sites=["_RETURN"],
                )
                y_hat = predictive(g, g.ndata["feat"].float())["_RETURN"].mean(0)
                ys_hat.append(y_hat.cpu())
        ys = torch.cat(ys, 0)
        ys_hat = torch.cat(ys_hat, 0)

        from ogb.graphproppred import Evaluator
        evaluator = Evaluator(name=args.data)
        results = evaluator.eval({"y_true": ys, "y_pred": ys_hat})
        rocauc = results["rocauc"]
        print("ROCAUC: %.6f" % rocauc, flush=True)
    return rocauc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ogbg-molhiv")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_features", type=int, default=25)
    parser.add_argument("--embedding_features", type=int, default=20)
    parser.add_argument("--activation", type=str, default="SiLU")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=5)
    parser.add_argument("--sigma_factor", type=float, default=2.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--kl_scale", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--adjoint", type=int, default=0)
    parser.add_argument("--physique", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--readout_depth", type=int, default=1)
    parser.add_argument("--swa_start", type=int, default=20)
    parser.add_argument("--swa_freq", type=int, default=10)
    parser.add_argument("--swa_lr", type=float, default=1e-2)
    parser.add_argument("--dropout_in", type=float, default=0.0)
    parser.add_argument("--dropout_out", type=float, default=0.0)
    parser.add_argument("--norm", type=int, default=1)
    parser.add_argument("--subsample_size", type=int, default=100)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=2666)
    args = parser.parse_args()
    run(args)
