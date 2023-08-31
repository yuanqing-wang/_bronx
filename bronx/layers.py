import math
from turtle import forward
from typing import Optional, Callable
from functools import partial
import torch
import pyro
from pyro import poutine
import dgl

# dgl.use_libxsmm(False)
from dgl.nn import GraphConv
from dgl import function as fn
from dgl.nn.functional import edge_softmax



class BronxLayer(pyro.nn.PyroModule):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, g, x):
        raise NotImplementedError



                

