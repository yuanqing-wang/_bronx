from types import SimpleNamespace
import ray
from ray.tune import ExperimentAnalysis
from run import run as run_experiment
from tune import multiply_by_heads

def run(path):
    analysis = ExperimentAnalysis(path)
    trial = analysis.get_best_trial("_metric/accuracy", "max", "all")
    accuracy = trial.metric_analysis["_metric/accuracy"]["max"]
    print(accuracy)
    config = trial.config
    print(config)
    config = multiply_by_heads(config)
    config["test"] = 1
    config["num_samples"] = 16
    config["num_particles"] = 16
    config = SimpleNamespace(**config)  
    accuracy, accuracy_te = run_experiment(config)
    print(accuracy, accuracy_te)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
