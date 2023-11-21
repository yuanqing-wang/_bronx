import dgl
import torch
import pyro
from bronx.baseline import DropEdge

def run(args):
    graphs, labels = dgl.load_graphs(args.data)
    y = labels["y"]
    g = dgl.batch(graphs)
    g.ndata["h0"] = torch.nn.functional.one_hot(g.ndata["h"][:, 0])
    g.ndata["h1"] = torch.nn.functional.one_hot(g.ndata["h"][:, 1])
    g.ndata["h"] = torch.cat([g.ndata["h0"], g.ndata["h1"]], dim=-1).float()

    model = DropEdge(
        in_features=g.ndata["h"].shape[-1],
        hidden_features=args.hidden_features,
        out_features=y.max().item()+1,
        p=0.0,
        activation=torch.nn.ELU(),
        depth=3,
    )

    if torch.cuda.is_available():
        model = model.to("cuda:0")
        g = g.to("cuda:0")
        y = y.to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        h = model(g, g.ndata["h"])
        h = h[g.ndata["mask"]]
        loss = torch.nn.CrossEntropyLoss()(h, y)
        loss.backward()
        optimizer.step()
        accuracy = (h.argmax(dim=-1) == y).float().mean()
        print(accuracy)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--hidden_features", type=int, default=32)
    parser.add_argument("--embedding_features", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--activation", type=str, default="ELU")
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--epochs", type=int, default=500000)
    parser.add_argument("--num_particles", type=int, default=1)

    args = parser.parse_args()
    run(args)