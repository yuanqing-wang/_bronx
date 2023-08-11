from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.trainable import session
from ray.tune.search.ax import AxSearch

def objective(args):
    args = SimpleNamespace(**args)
    accuracy = run(args)
    session.report({"accuracy": accuracy})

def experiment(args):
    ray.init(num_gpus=1, num_cpus=1)
    name = datetime.now().strftime("%m%d%Y%H%M%S")
    print(name) 

    param_space = {
        "data": tune.choice([args.data]),
        "hidden_features": tune.randint(16, 128),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-10, 1e-3),
        "activation": tune.choice(["sigmoid", "tanh", "elu", "silu"]),
        "log_sigma": tune.uniform(-5.0, 0.0),
        "t": tune.uniform(1.0, 5.0),
        "optimizer": tune.choice(["RMSprop", "Adam", "AdamW"]),
        "n_epochs": tune.randint(50, 200),
        "test": tune.choice([0]),
    }

    tune_config = tune.TuneConfig(
        metric="_metric/accuracy",
        mode="max",
        search_alg=AxSearch(),
        num_samples=1000,
    )

    run_config = air.RunConfig(
        verbose=0,
        name=name,
        storage_path=args.data,
        stop={"time_total_s": 100},
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
