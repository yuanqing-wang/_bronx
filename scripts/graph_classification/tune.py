from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.trainable import session
from ray.tune.search.optuna import OptunaSearch

def objective(args):
    args["embedding_features"] = (
        args["embedding_features"] * args["num_heads"]
    )
    args["hidden_features"] = args["hidden_features"] * args["num_heads"]
    args = SimpleNamespace(**args)
    rocauc_vl, rocauc_te = run(args)
    session.report({"rocauc": rocauc_vl, "rocauc_te": rocauc_te})

def experiment(args):
    ray.init(num_gpus=1, num_cpus=1)
    name = datetime.now().strftime("%m%d%Y%H%M%S")
    print(name)

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
        "swa_start": tune.randint(10, 30),
        "swa_freq": tune.randint(5, 10),
        "swa_lr": tune.loguniform(1e-5, 1e-1),
        "node_prior": 1, # tune.choice([0, 1]),
        "edge_recover": 0.0, # tune.loguniform(1e-5, 1e-1),
        "seed": 2666,
        "k": 0,
        "checkpoint": "",
        "batch_size": 1024,
    }

    tune_config = tune.TuneConfig(
        metric="_metric/rocauc",
        mode="max",
        search_alg=OptunaSearch(),
        num_samples=1000,
    )

    run_config = air.RunConfig(
        verbose=0,
        name=name,
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
    parser.add_argument("--data", type=str, default="ogbg-molhiv")
    args = parser.parse_args()
    experiment(args)
