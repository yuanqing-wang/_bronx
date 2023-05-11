from types import SimpleNamespace
import numpy as np
import torch
import dgl
from ray import tune, air
from ray.tune.trainable import session
from ray.tune.search.optuna import OptunaSearch
from bronx.models import BronxModel

def objective(args):
    args = SimpleNamespace(**args)
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
    g = locals()[f"{args.data.capitalize()}GraphDataset"]()[0]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    model = BronxModel(
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].max() + 1,
        hidden_features=args.hidden_features,
        dropout0=args.dropout0,
        dropout1=args.dropout1,
        gamma=args.gamma,
    )

    if torch.cuda.is_available():
        model = model.cuda()
        g = g.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    while True:
        model.train()
        optimizer.zero_grad()
        y_hat, kl = model(g, g.ndata['feat'])
        y_hat = y_hat[g.ndata['train_mask']]
        kl = kl.squeeze(-2)[g.ndata['train_mask']]
        y = g.ndata['label'][g.ndata['train_mask']]
        loss = torch.nn.CrossEntropyLoss()(y_hat, y) + kl.mean()
        loss.backward()
        optimizer.step()
        model.eval()

        with torch.no_grad():
            y_hat, _ = model(g, g.ndata["feat"])
            y_hat = y_hat[g.ndata["val_mask"]]
            y = g.ndata["label"][g.ndata["val_mask"]]
            accuracy = float((y_hat.argmax(-1) == y).sum()) / len(y_hat)
            session.report({"accuracy": accuracy})

def run():
    param_space = {
        "data": tune.choice(["cora"]),
        "hidden_features": tune.qlograndint(16, 512, 1),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "dropout0": tune.uniform(0.0, 0.8),
        "dropout1": tune.uniform(0.0, 0.8),
        "gamma": tune.uniform(0.0, 1.0),
    }

    tune_config = tune.TuneConfig(
        metric="accuracy",
        mode="max",
        search_alg=OptunaSearch(),
    )

    run_config=air.RunConfig(stop={"training_iteration": 50})

    tuner = tune.Tuner(
        objective,
        tune_config=tune_config,
        run_config=run_config,
        param_space=param_space,
    )

    tuner.fit()


if __name__ == "__main__":
    run()
