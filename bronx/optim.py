import pyro
from torchcontrib.optim import SWA as _SWA


def swa_constructor(param, base, base_args, swa_args):
    base = base(param, **base_args)
    optimizer = _SWA(base, **swa_args)
    return optimizer


def SWA(args):
    return pyro.optim.PyroOptim(
        swa_constructor,
        args,
    )


def swap_swa_sgd(optimizer):
    for key, value in optimizer.optim_objs.items():
        value.swap_swa_sgd()
