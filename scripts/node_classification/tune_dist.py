from calendar import c
from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.trainable import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import os
ray.init(num_cpus=os.cpu_count())
LSF_COMMAND = "bsub -q gpuqueue -gpu " +\
"\"num=1:j_exclusive=yes\" -R \"rusage[mem=5] span[ptile=1]\" -W 0:10 -Is "

PYTHON_COMMAND =\
"python /data/chodera/wangyq/bronx/scripts/node_classification/run.py"

def args_to_command(args):
    command = LSF_COMMAND + "\""
    command += PYTHON_COMMAND
    for key, value in args.items():
        command += f" --{key} {value}"
    command += "\""
    return command

def lsf_submit(command):
    import subprocess
    print("--------------------")
    print("Submitting command:")
    print(command)
    print("--------------------")
    output = subprocess.getoutput(command)
    return output

def parse_output(output):
    line = output.split("\n")[-1]
    if "ACCURACY" not in line:
        print(output, flush=True)
        return 0.0
    return float(line.split(" ")[-1])

def multiply_by_heads(args):
    args["embedding_features"] = (
        args["embedding_features"] * args["num_heads"]
    )
    args["hidden_features"] = args["hidden_features"] * args["num_heads"]
    return args

def objective(args):
    args = multiply_by_heads(args)
    checkpoint = os.path.join(os.getcwd(), "model.pt")
    args["checkpoint"] = checkpoint
    command = args_to_command(args)
    output = lsf_submit(command)
    accuracy = parse_output(output)
    session.report({"accuracy": accuracy})

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S")
    param_space = {
        "data": args.data,
        "hidden_features": tune.randint(4, 16),
        "embedding_features": tune.randint(4, 16),
        "num_heads": tune.randint(4, 32),
        "depth": tune.randint(1, 5),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-10, 1e-2),
        "num_samples": 32,
        "num_particles": 8,
        "sigma_factor": tune.uniform(0.5, 10.0),
        "t": tune.uniform(0.5, 10.0),
        "optimizer": tune.choice(["RMSprop", "Adam", "AdamW"]),
        "activation": tune.choice(["Tanh", "SiLU", "ELU", "Sigmoid", "ReLU"]),
        "adjoint": 0,
        "physique": 0,
        "gamma": tune.uniform(0.0, 1.0),
        "readout_depth": 1,
        "kl_scale": tune.loguniform(1e-10, 1e-2),
        "swa_start": tune.randint(1, 50),
        "swa_freq": tune.randint(1, 20),
        "swa_lr": tune.loguniform(1e-5, 1e-1),
        "dropout_in": tune.uniform(0, 1),
        "dropout_out": tune.uniform(0, 1),
        "n_epochs": 200,
        "seed": 2666,
    }

    tune_config = tune.TuneConfig(
        metric="_metric/accuracy",
        mode="max",
        search_alg=ConcurrencyLimiter(OptunaSearch(), args.concurrent),
        num_samples=5000,
    )

    run_config = air.RunConfig(
        name=name,
        storage_path=args.data,
    )

    tuner = tune.Tuner(
        objective,
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    parser.add_argument("--concurrent", type=int, default=50)
    args = parser.parse_args()
    experiment(args)
