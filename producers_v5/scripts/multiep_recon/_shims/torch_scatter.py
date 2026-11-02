"""Minimal pure-torch stand-in for the two torch_scatter functions used by Profactor/continuous-remeshing
(work/third_party/continuous-remeshing @2e55642). torch_scatter is not installed in the system python."""
import torch


def scatter_max(src, index, dim=0, out=None):
    assert dim == 0 and out is not None
    out.scatter_reduce_(0, index, src, reduce='amax', include_self=True)
    return out, None


def scatter_mean(src, index, dim=0, out=None):
    assert dim == 0 and out is not None
    idx = index.expand_as(src) if index.shape != src.shape else index
    s = torch.zeros_like(out).scatter_add_(0, idx, src)
    c = torch.zeros_like(out).scatter_add_(0, idx, torch.ones_like(src))
    out.copy_(s / c.clamp_min(1))
    return out
