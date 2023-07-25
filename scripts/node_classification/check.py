import os
import glob
import json
import ray
from ray.tune import ExperimentAnalysis

def check(path):
    results = []
    result_paths = glob.glob(path + "/*/*/result.json")
    for result_path in result_paths:
        try:
            with open(result_path, "r") as f:
                result_str = f.read()
                result = json.loads(result_str)
            results.append(result)
        except:
            pass

    results = sorted(results, key=lambda x: x["_metric"]["accuracy"], reverse=True)

    print(results[0]["_metric"]["accuracy"])
    print(results[0]["config"])

    return results[0]["config"]

if __name__ == "__main__":
    import sys
    check(sys.argv[1])
