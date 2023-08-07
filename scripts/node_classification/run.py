import numpy as np
import torch
import pyro
from pyro import poutine
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)
from pyro.contrib.gp.likelihoods.multi_class import MultiClass
from bronx.models import GraphVariationalSparseGaussianProcess as GVSGP
from pyro.contrib.gp.models.vsgp import VariationalSparseGP

def run(args):
    pyro.clear_param_store()
    torch.cuda.empty_cache()
    from dgl.data import (
        CoraGraphDataset,
        CiteseerGraphDataset,
        PubmedGraphDataset,
        CoauthorCSDataset,
        CoauthorPhysicsDataset,
        AmazonCoBuyComputerDataset,
        AmazonCoBuyPhotoDataset,
    )

    g = locals()[args.data](verbose=False)[0]

    g.ndata["feat"] = g.ndata["feat"] - g.ndata["feat"].mean(dim=-1, keepdim=True)
    g.ndata["feat"] = torch.nn.functional.normalize(g.ndata["feat"], dim=-1)


    if "train_mask" not in g.ndata:
        g.ndata["train_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        g.ndata["val_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
        g.ndata["test_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool)

        train_idxs = torch.tensor([], dtype=torch.int32)
        val_idxs = torch.tensor([], dtype=torch.int32)
        test_idxs = torch.tensor([], dtype=torch.int32)

        n_classes = g.ndata["label"].shape[-1]
        for idx_class in range(n_classes):
            idxs = torch.where(g.ndata["label"][:, idx_class] == 1)[0]
            assert len(idxs) > 50
            idxs = idxs[torch.randperm(len(idxs))]
            _train_idxs = idxs[:20]
            _val_idxs = idxs[20:50]
            _test_idxs = idxs[50:]
            train_idxs = torch.cat([train_idxs, _train_idxs])
            val_idxs = torch.cat([val_idxs, _val_idxs])
            test_idxs = torch.cat([test_idxs, _test_idxs])

        g.ndata["train_mask"][train_idxs] = True
        g.ndata["val_mask"][val_idxs] = True
        g.ndata["test_mask"][test_idxs] = True

    if torch.cuda.is_available():
        g = g.to("cuda:0")

    likelihood = MultiClass(num_classes=g.ndata["label"].max()+1)
    from pyro.contrib.gp.kernels import Matern32
    kernel = Matern32(args.hidden_features)

    model = GVSGP(
        graph=g,
        X=g.ndata["feat"],
        y=g.ndata["label"][g.ndata["train_mask"]],
        iX=torch.where(g.ndata["train_mask"])[0],
        in_features=g.ndata["feat"].shape[-1],
        hidden_features=args.hidden_features,
        kernel=kernel,
        jitter=1e-5,
        latent_shape=(g.ndata["label"].max()+1,),
        likelihood=likelihood,
    )

    if torch.cuda.is_available():
        # a = a.cuda()
        model = model.cuda()

    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss


    for idx in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model.model, model.guide)
        loss.backward()
        model.eval()

        with torch.no_grad():
            y_hat, _ = model(torch.where(g.ndata["val_mask"])[0])
            y_hat = y_hat.argmax(dim=0)
            accuracy = (y_hat == g.ndata["label"][g.ndata["val_mask"]]).float().mean()
            print(accuracy)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--embedding_features", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--optimizer", type=str, default="RMSprop")
    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--test", type=int, default=1)
    args = parser.parse_args()
    run(args)
