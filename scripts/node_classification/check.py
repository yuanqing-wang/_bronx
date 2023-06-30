import ray
from ray.tune import ExperimentAnalysis

def run(path):
    analysis = ExperimentAnalysis(path)
    df = analysis.dataframe()
    trial = analysis.get_best_trial("_metric/accuracy", "max", "all")
    accuracy = trial.metric_analysis["_metric/accuracy"]["max"]
    print(accuracy)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
