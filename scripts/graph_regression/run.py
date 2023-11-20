import random
from re import split
import numpy as np
import torch
import pyro
from pyro import poutine
import dgl
dgl.use_libxsmm(False)
from bronx.models import GraphRegressionBronxModel
from bronx.optim import SWA, swap_swa_sgd

def run(args):
    pyro.clear_param_store()
    from dgllife.data import (
        ESOL,
        FreeSolv,
        Lipophilicity,
    )
    from dgllife.utils import (
        CanonicalAtomFeaturizer,
        CanonicalBondFeaturizer,
    )
    data = locals()[args.data](
        node_featurizer=CanonicalAtomFeaturizer("h0"),
        edge_featurizer=CanonicalBondFeaturizer("e0"),
    )
    from dgllife.utils import RandomSplitter
    splitter = RandomSplitter()
    data_train, data_valid, data_test = splitter.train_val_test_split(
        data, frac_train=0.8, frac_val=0.1, frac_test=0.1, 
        random_state=args.seed,
    )

    _, g, y = next(iter(dgl.dataloading.GraphDataLoader(
        data_train, batch_size=len(data_train),
    )))

    model = GraphRegressionBronxModel(
        in_features=g.ndata["h0"].shape[-1],
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
        y_mean=y.mean(),
        y_std=y.std(),
    )

    if torch.cuda.is_available():
        model = model.to("cuda:0")

    batch_size = args.batch_size if args.batch_size > 0 else len(data_train)

    data_train = dgl.dataloading.GraphDataLoader(
        data_train, batch_size=batch_size, shuffle=True, drop_last=True
    )

    data_valid = dgl.dataloading.GraphDataLoader(
        data_valid, batch_size=len(data_valid),
    )

    scheduler = pyro.optim.ReduceLROnPlateau(
        {
            "optimizer": getattr(torch.optim, args.optimizer),
            "optim_args": {
                "lr": args.learning_rate, 
                "weight_decay": args.weight_decay
            },
            "factor": args.lr_factor,
            "patience": args.patience,
            "mode": "min",
        },
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
        for _, g, y in data_train:
            if torch.cuda.is_available():
                g = g.to("cuda:0")
                y = y.to("cuda:0")
            model.train()
            loss = svi.step(g, g.ndata["h0"], y)
    
        _, g, y = next(iter(data_valid))
        if torch.cuda.is_available():
            g = g.to("cuda:0")
            y = y.to("cuda:0")

        model.eval()
        with torch.no_grad():

            predictive = pyro.infer.Predictive(
                model,
                guide=model.guide,
                num_samples=args.num_samples,
                parallel=True,
                return_sites=["_RETURN"],
            )

            y_hat = predictive(g, g.ndata["h0"])["_RETURN"].mean(0)
            rmse = float(((y_hat - y) ** 2).mean() ** 0.5)
        scheduler.step(rmse)
        print(rmse)

    return rmse

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ESOL")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--hidden_features", type=int, default=100)
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
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--adjoint", type=int, default=0)
    parser.add_argument("--physique", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--readout_depth", type=int, default=1)
    parser.add_argument("--dropout_in", type=float, default=0.0)
    parser.add_argument("--dropout_out", type=float, default=0.0)
    parser.add_argument("--norm", type=int, default=1)
    parser.add_argument("--subsample_size", type=int, default=100)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=2666)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    run(args)
