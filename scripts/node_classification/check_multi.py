import os
import glob
import json
import pandas as pd
import torch
import pyro
import dgl

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

    from run import get_graph
    g = get_graph(results[0]["config"]["data"])
    y = g.ndata["label"].argmax(-1)

    ys_hat = []
    with torch.no_grad():
        for idx in range(args.first):
            if torch.cuda.is_available():
                model = torch.load(results[idx]["config"]["checkpoint"])
                g = g.to("cuda:0")
            else:
                model = torch.load(results[idx]["config"]["checkpoint"], map_location="cpu")
            model.eval()

            predictive = pyro.infer.Predictive(
                model,
                guide=model.guide,
                num_samples=args.num_samples,
                parallel=True,
                return_sites=["_RETURN"],
            )

            y_hat = predictive(g, g.ndata["feat"])["_RETURN"]
            ys_hat.append(y_hat)

    g = g.to("cpu")
    y_hat = torch.cat(ys_hat, dim=0).mean(0).argmax(-1).cpu()

    print(y_hat.shape, y.shape)
    print(y_hat)
    print(y)

    accuracy_vl = (y_hat[g.ndata["val_mask"]] == y[g.ndata["val_mask"]]).float().mean()
    accuracy_te = (y_hat[g.ndata["test_mask"]] == y[g.ndata["test_mask"]]).float().mean()
    print(accuracy_vl, accuracy_te)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".")
    parser.add_argument("--first", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=32)
    args = parser.parse_args()
    check(args)
