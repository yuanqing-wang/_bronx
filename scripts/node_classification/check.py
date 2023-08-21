import os
import glob
import json
import pandas as pd

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
    df = pd.DataFrame([result["config"] for result in results])
    df["accuracy"] = [result["_metric"]["accuracy"] for result in results]
    df.to_csv("results.csv")

    print(results[0]["_metric"]["accuracy"])
    print(results[0]["config"])

    return results[0]["config"]

if __name__ == "__main__":
    import sys
    check(sys.argv[1])
