"""Various mesh utilities"""
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets


def count_parameters(model, print_result=True):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if print_result:
        if num > 1e6:
            print("The model has {:.1f}M parameters".format(num/1000000))
        elif num > 1000:
            print("The model has {:.1f}k parameters".format(num/1000))
        return
    return num


def NDCnormalize(vertices, return_scale=False):
    """normalization in half unit ball"""
    vM = vertices.max(0)
    vm = vertices.min(0)
    scale = np.sqrt(((vM - vm) ** 2).sum(-1))
    mean = (vM + vm) / 2.0
    nverts = (vertices - mean) / scale
    if return_scale:
        return nverts, mean, scale
    return nverts


def plotSlice(sdf_array, vmax):
    def helper(xhi, slice, vmax,  cmap='seismic'):
        plt.imshow(xhi[slice], origin='lower',
                   cmap=cmap, vmin=-vmax, vmax=vmax)
    slider = ipywidgets.IntSlider(
        min=0, max=sdf_array.shape[0]-1, step=1, value=sdf_array.shape[0]//2)
    return ipywidgets.interact(lambda s: helper(sdf_array, s, vmax), s=slider)
