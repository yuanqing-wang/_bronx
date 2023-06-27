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
    session.report({"accuracy": run(args)})

def experiment(args):
    ray.init(num_gpus=1, num_cpus=1)
    name = datetime.now().strftime("%m%d%Y%H%M%S")
    print(name)

    param_space = {
        "data": tune.choice([args.data]),
        "hidden_features": tune.randint(8, 16),
        "embedding_features": tune.randint(8, 16),
        "num_heads": tune.randint(4, 16),
        "depth": tune.randint(1, 6),
        "learning_rate": tune.loguniform(1e-4, 5e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "patience": tune.randint(5, 10),
        "factor": tune.uniform(0.5, 1.0),
        "num_samples": tune.choice([16]),
        "num_particles": tune.choice([16]),
    }

    tune_config = tune.TuneConfig(
        metric="_metric/accuracy",
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
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    args = parser.parse_args()
    experiment(args)
