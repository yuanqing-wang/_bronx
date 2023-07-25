from types import SimpleNamespace
import ray
from ray.tune import ExperimentAnalysis
from run import run as run_experiment
from tune import multiply_by_heads

def run(path, n_repeat=3):
    from check import check
    config = check(path)
    config["test"] = 1
    config = SimpleNamespace(**config)  

    accuracies = []
    accuracies_te = []

    for _ in range(n_repeat):
        accuracy, accuracy_te = run_experiment(config)
        accuracies.append(accuracy)
        accuracies_te.append(accuracy_te)

    accuracy = sum(accuracies) / len(accuracies)
    accuracy_te = sum(accuracies_te) / len(accuracies_te)
    print(accuracy, accuracy_te)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
