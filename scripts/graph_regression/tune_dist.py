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
"\"num=1:j_exclusive=yes\" -R \"rusage[mem=5] span[ptile=1]\" -W 0:20 -Is "

PYTHON_COMMAND =\
"python /data/chodera/wangyq/bronx/scripts/graph_regression/run.py"

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
    if "RMSE" not in line:
        print(output, flush=True)
        return 99.9, 99.9
    _, accuracy_vl, accuracy_te = line.split(",")
    return float(accuracy_vl), float(accuracy_te)
    
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
    rmse, rmse_te = parse_output(output)
    session.report({"accuracy": rmse, "accuracy_te": rmse_te})

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
        "sigma_factor": tune.uniform(5.0, 15.0),
        "t": tune.uniform(5.0, 15.0),
        "optimizer": "Adam", # tune.choice(["RMSprop", "Adam", "AdamW", "Adamax", "SGD", "Adagrad"]),
        "activation": "ELU", # tune.choice(["Tanh", "SiLU", "ELU", "Sigmoid", "ReLU"]),
        "adjoint": 1, # tune.choice([0, 1]),
        "physique": 1,
        "norm": 0, # tune.choice([0, 1]),
        "gamma": tune.uniform(0.5, 1.0),
        "readout_depth": 1, # tune.randint(1, 4),
        "kl_scale": tune.loguniform(1e-10, 1e-2),
        "dropout_in": tune.uniform(0.0, 1.0),
        "dropout_out": tune.uniform(0.0, 1.0),
        "n_epochs": 500,
        "swa_start": 100,
        "swa_freq": tune.randint(5, 10),
        "swa_lr": tune.loguniform(1e-5, 1e-1),
        "seed": 2666,
        "k": 0,
    }

    tune_config = tune.TuneConfig(
        metric="_metric/rmse",
        mode="min",
        search_alg=ConcurrencyLimiter(OptunaSearch(), args.concurrent),
        num_samples=100,
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
    parser.add_argument("--data", type=str, default="ESOL")
    parser.add_argument("--concurrent", type=int, default=10)
    args = parser.parse_args()
    experiment(args)
