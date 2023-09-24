from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.trainable import session
from ray.tune.search.optuna import OptunaSearch
import os

if "head_node" in os.environ:
    ray.init(address=os.environ["head_node"] + ":" + os.environ["port"])
else:
    import torch
    n_gpus = torch.cuda.device_count()
    ray.init(num_cpus=n_gpus, num_gpus=n_gpus)

def multiply_by_heads(args):
    args["embedding_features"] = (
        args["embedding_features"] * args["num_heads"]
    )
    args["hidden_features"] = args["hidden_features"] * args["num_heads"]
    return args

def objective(args):
    args = multiply_by_heads(args)
    args = SimpleNamespace(**args)
    accuracy_vl, accuracy_te = run(args)
    session.report({"accuracy": accuracy_vl, "accuracy_te": accuracy_te})

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S")
    param_space = {
        "data": args.data,
        "hidden_features": tune.randint(1, 8),
        "embedding_features": tune.randint(2, 8),
        "num_heads": tune.randint(4, 32),
        "depth": 1, # tune.randint(1, 4),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-10, 1e-2),
        "num_samples": 4,
        "num_particles": 4,
        "sigma_factor": tune.uniform(1.0, 15.0),
        "t": tune.uniform(1.0, 15.0),
        "optimizer": "Adam", # tune.choice(["RMSprop", "Adam", "AdamW", "Adamax", "SGD", "Adagrad"]),
        "activation": "ELU", # tune.choice(["Tanh", "SiLU", "ELU", "Sigmoid", "ReLU"]),
        "adjoint": 1, # tune.choice([0, 1]),
        "physique": 1,
        "norm": 0, # tune.choice([0, 1]),
        "gamma": 1.0, # tune.uniform(0.0, 1.0),
        "readout_depth": 1, # tune.randint(1, 4),
        "kl_scale": tune.loguniform(1e-5, 1e-2),
        "dropout_in": tune.uniform(0.0, 1.0),
        "dropout_out": tune.uniform(0.0, 1.0),
        "consistency_factor": tune.loguniform(1e-2, 1.0),
        "consistency_temperature": tune.uniform(0.0, 1.0),
        "n_epochs": 50,
        "swa_lr": tune.loguniform(1e-5, 1e-1),
        "node_prior": 1, # tune.choice([0, 1]),
        "edge_recover": 0.0, # tune.loguniform(1e-5, 1e-1),
        "seed": 2666,
        "split_index": 0,
        "k": 0,
        "patience": 10,
        "checkpoint": "",
    }

    tune_config = tune.TuneConfig(
        metric="_metric/accuracy",
        mode="max",
        search_alg=OptunaSearch(),
        num_samples=10000,
    )

    run_config = air.RunConfig(
        name=name,
        storage_path=args.data,
        verbose=0,
    )

    tuner = tune.Tuner(
        tune.with_resources(objective, {"cpu": 1, "gpu": 1}),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CornellDataset")
    args = parser.parse_args()
    experiment(args)
