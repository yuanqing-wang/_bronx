import pytest

def test_candidates():
    from bronx.utils import candidates
    import dgl
    g = dgl.rand_graph(5, 10)
    candidates = candidates(g, 3)

def test_combine():
    from bronx.utils import candidates, combine
    import dgl
    g = dgl.rand_graph(5, 10)
    candidates = candidates(g, 3)
    g = combine(candidates, [1, 1, 1])
    print(g)


