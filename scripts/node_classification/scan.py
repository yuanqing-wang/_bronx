import ray
from ray.tune import ExperimentAnalysis

def run(path):
    analysis = ExperimentAnalysis(path)
    df = analysis.dataframe()
    df.sort_values("_metric/accuracy", ascending=False, inplace=True)
    print(df)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
