import os
import glob
import json
import pandas as pd
import torch
import pyro
import dgl

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
    df["accuracy_te"] = [result["_metric"]["accuracy_te"] for result in results]
    df.to_csv("results.csv")

    print(results[0]["_metric"]["accuracy"], results[0]["_metric"]["accuracy_te"])
    print(results[0]["config"], flush=True)

    from run import get_graph
    g = get_graph(results[0]["config"]["data"])


    if torch.cuda.is_available():
        model = torch.load(results[0]["config"]["checkpoint"])
        g = g.to("cuda:0")
    else:
        model = torch.load(results[0]["config"]["checkpoint"], map_location="cpu")
    model.eval()


    with torch.no_grad():
        predictive = pyro.infer.Predictive(
            model,
            guide=model.guide,
            num_samples=64,
            parallel=True,
            return_sites=["_RETURN"],
        )

        y_hat = predictive(g, g.ndata["feat"], mask=g.ndata["val_mask"])[
            "_RETURN"
        ]
        
        y_hat = y_hat.softmax(-1).mean(0)
        y = g.ndata["label"][g.ndata["val_mask"]]
        accuracy_vl = float((y_hat.argmax(-1) == y.argmax(-1)).sum()) / len(
            y_hat
        )

        y_hat = predictive(g, g.ndata["feat"], mask=g.ndata["test_mask"])[
            "_RETURN"
        ]
        
        y_hat = y_hat.softmax(-1).mean(0)
        y = g.ndata["label"][g.ndata["test_mask"]]
        accuracy_te = float((y_hat.argmax(-1) == y.argmax(-1)).sum()) / len(
            y_hat
        )
        print(accuracy_vl, accuracy_te)

if __name__ == "__main__":
    import sys
    check(sys.argv[1])
