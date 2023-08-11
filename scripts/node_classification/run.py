import numpy as np
import torch
import gpytorch
# gpytorch.settings.debug._default = False
gpytorch.settings.lazily_evaluate_kernels._default = False
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
dgl.use_libxsmm(False)
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior

def run(args):
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
    g = dgl.remove_self_loop(g)
    # g = dgl.add_self_loop(g)

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

    from bronx.models import BronxModel, ExactBronxModel
    if torch.cuda.is_available():
        g = g.to("cuda:0")

    likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
        targets=g.ndata["label"][g.ndata["train_mask"]],
        learn_additional_noise=True,
    )

    model = ExactBronxModel(
        train_x=torch.where(g.ndata["train_mask"])[0],
        train_y=likelihood.transformed_targets,
        likelihood=likelihood,
        num_classes=likelihood.num_classes,
        features=g.ndata["feat"],
        graph=g,
        in_features=g.ndata["feat"].shape[-1],
        hidden_features=args.hidden_features,
    )


    if torch.cuda.is_available():
        model = model.to("cuda:0")
        likelihood = likelihood.cuda()

    model.covar_module.base_kernel.register_prior(
        "lengthscale_prior", 
        UniformPrior(
            torch.tensor(0.01, device=g.device), 
            torch.tensor(0.5, device=g.device),
        ), 
        "lengthscale"
    )
    model.covar_module.register_prior(
        "outputscale_prior", 
        UniformPrior(
            torch.tensor(1.0, device=g.device), 
            torch.tensor(2.0, device=g.device), 
        ), 
        "outputscale"
    )
    likelihood.register_prior(
        "noise_prior", 
        UniformPrior(
            torch.tensor(0.01, device=g.device),
            torch.tensor(0.5, device=g.device),
        ), 
        "noise"
    )

    def pyro_model(x, y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled_model = model.pyro_sample_from_prior()
            output = sampled_model.likelihood(sampled_model(x))
            pyro.sample("obs", output, obs=y)
        return y

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    nuts_kernel = NUTS(pyro_model)
    mcmc_run = MCMC(nuts_kernel, num_samples=200, warmup_steps=50)
    x = torch.where(g.ndata["train_mask"])[0]
    y = likelihood.transformed_targets

    mcmc_run.run(x, y)

    # optimizer = getattr(
    #     torch.optim, args.optimizer
    # )(
    #     list(model.hyperparameters()) + list(likelihood.parameters()), 
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay,
    # )

    # for idx in range(args.n_epochs):
    #     model.train()
    #     likelihood.train()
    #     optimizer.zero_grad()
    #     output = model(torch.where(g.ndata["train_mask"])[0])
    #     loss = -mll(output, target=likelihood.transformed_targets)
    #     loss = loss.sum()
    #     loss.backward()
    #     optimizer.step()

    #     with torch.no_grad():
    #         model.eval()
    #         likelihood.eval()
    #         y_hat = model(torch.where(g.ndata["val_mask"])[0]).loc
    #         y = g.ndata["label"][g.ndata["val_mask"]]
    #         accuracy = (y_hat.argmax(dim=0) == y).float().mean()
    #         print(accuracy)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    parser.add_argument("--hidden_features", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--test", type=int, default=1)
    args = parser.parse_args()
    run(args)
