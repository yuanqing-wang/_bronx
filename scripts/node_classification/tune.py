from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.trainable import session
from ray.tune.search.ax import AxSearch
from ray.tune.schedulers import ASHAScheduler
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
    run(args)

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S")
    param_space = {
        "data": tune.choice([args.data]),
        "hidden_features": tune.randint(4, 16),
        "embedding_features": tune.randint(4, 16),
        "num_heads": tune.randint(4, 16),
        "depth": tune.choice([1]),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-10, 1e-2),
        "num_samples": tune.choice([8]),
        "num_particles": tune.choice([8]),
        "sigma_factor": tune.uniform(0.5, 10.0),
        "t": tune.uniform(0.5, 10.0),
        "optimizer": tune.choice(["RMSprop", "Adam", "AdamW"]),
        "activation": tune.choice(["Tanh", "SiLU", "ELU", "Sigmoid", "ReLU"]),
        "adjoint": tune.choice([0, 1]),
        "physique": tune.choice([0, 1]),
        "consistency_factor": tune.loguniform(1e-5, 1e-1),
        "consistency_temperature": tune.uniform(0.0, 1.0),
        "kl_scale": tune.loguniform(1e-10, 1e-2),
        "n_epochs": tune.choice([50]),
        "gamma": tune.uniform(0.0, 1.0),
        "readout_depth": tune.randint(1, 3),
        "test": tune.choice([0]),
    }

    tune_config = tune.TuneConfig(
        metric="accuracy",
        mode="max",
        search_alg=AxSearch(),
        num_samples=1000,
    )

    run_config = air.RunConfig(
        name=args.data,
        storage_path=args.data,
        stop={"time_total_s": 200, "training_iteration": 50},
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=1),
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
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    args = parser.parse_args()
    experiment(args)
