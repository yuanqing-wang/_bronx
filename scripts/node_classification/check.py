import os
import glob
import json
from re import S
import pandas as pd
import torch
import pyro
import dgl
from types import SimpleNamespace

def check(args):
    results = []
    result_paths = glob.glob(args.path + "/*/*/result.json")
    for result_path in result_paths:
        try:
            with open(result_path, "r") as f:
                result_str = f.read()
                result = json.loads(result_str)
            results.append(result)
        except:
            pass

    results = sorted(results, key=lambda x: x["_metric"]["accuracy"], reverse=True)

    print(results[0]["_metric"]["accuracy"], results[0]["_metric"]["accuracy_te"])
    print(results[0]["config"], flush=True)

    if len(args.report) > 1:
        df = pd.DataFrame([result["config"] for result in results])
        df["accuracy"] = [result["_metric"]["accuracy"] for result in results]
        df["accuracy_te"] = [result["_metric"]["accuracy_te"] for result in results]
        df.to_csv(args.report)

    from run import get_graph
    g = get_graph(results[0]["config"]["data"])


    if args.rerun:
        from run import run
        config = results[0]["config"]
        config["split_index"] = -1
        config["lr_factor"] = 0.5
        config["patience"] = 10
        config = SimpleNamespace(**config)
        accuracy_vl, accuracy_te = run(config)

    if args.reevaluate:
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--report", type=str, default="")
    parser.add_argument("--rerun", type=int, default=0)
    args = parser.parse_args()
    check(args)
