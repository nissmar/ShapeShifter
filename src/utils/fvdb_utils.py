import fvdb
import fvdb.nn as fvnn
import trimesh
import igl
from meshplot import plot
import torch
import numpy as np


def mesh_to_grid(v, f, grid_size, device='cuda'):
    voxel_sizes = 2/grid_size
    origins = [voxel_sizes / 2.] * 3
    if isinstance(v, np.ndarray):
        v = fvdb.JaggedTensor(torch.tensor(v, device=device))
        f = fvdb.JaggedTensor(torch.tensor(f, device=device))
    return fvdb.sparse_grid_from_mesh(v, f, voxel_sizes=voxel_sizes, origins=origins)


def grid_to_VDB(grid: fvdb.GridBatch, torch_func=torch.zeros, additional_feat=[], dtype=torch.float32):
    size = list(grid.jidx.shape)
    size += additional_feat
    tensor_feature = torch_func(size, dtype=dtype, device=grid.device)
    return fvdb.nn.VDBTensor(grid, grid.jagged_like(tensor_feature))


def grid_to_mesh(grid, output_edges=False, colors=None):
    '''colors: num_voxelsx3 np.array'''
    cube_mesh = trimesh.creation.box()
    cube_v = torch.tensor(cube_mesh.vertices, device=grid.device)[
        None, :]*grid.voxel_sizes
    cube_f = torch.tensor(cube_mesh.faces, device=grid.device)[None, :]
    if output_edges:
        cube_f = torch.tensor([[
            [0, 1], [0, 4], [0, 2], [1, 3], [1, 5], [4, 5],
            [5, 7], [4, 6], [7, 6], [2, 6], [2, 3], [3, 7]]],
            device=grid.device)
    points = grid.grid_to_world(grid.ijk.float()).jdata

    cube_v = cube_v.expand(len(points), *cube_v.shape[1:]).clone()
    cube_f = cube_f.expand(len(points), *cube_f.shape[1:]).clone()
    num = torch.arange(len(points), device=grid.device)
    cube_f += num[:, None, None]*cube_v.shape[1]
    cube_v += points[:, None, :]
    cube_v = cube_v.view(-1, cube_v.shape[-1]).cpu().detach().numpy()
    cube_f = cube_f.view(-1, cube_f.shape[-1]).cpu().detach().numpy()

    if colors is not None:
        colors = np.repeat(colors, len(cube_mesh.vertices), 0)
        return cube_v, cube_f, colors
    return cube_v, cube_f


def show_grid(grid, with_edges=True):
    mp = plot(*grid_to_mesh(grid))
    if with_edges:
        mp.add_edges(*grid_to_mesh(grid, True))
    return mp


def vdb_marching_cubes(out: fvnn.VDBTensor):
    nv, nf, _ = out.grid.marching_cubes(out.feature)
    return nv.jdata.cpu().detach().numpy(), nf.jdata.cpu().detach().numpy()


def show_vdb_marching_cubes(out: fvnn.VDBTensor):
    return plot(*vdb_marching_cubes(out))


def trilinear_upsample(small_tensor: fvnn.VDBTensor, large_grid: fvdb.GridBatch):
    new_centers = large_grid.grid_to_world(large_grid.ijk.float())
    new_features = small_tensor.grid.sample_trilinear(
        new_centers, small_tensor.feature)
    return fvnn.VDBTensor(large_grid, new_features)


def compute_sdf(grid: fvdb.GridBatch, v: np.array, f: np.array) -> fvnn.VDBTensor:
    points = grid.grid_to_world(grid.ijk.float()).jdata.cpu().detach().numpy()
    sdf_compute = igl.signed_distance(points, v, f)
    tensor_vdb = grid_to_VDB(grid)
    tensor_vdb.feature.jdata = torch.tensor(
        sdf_compute[0][:, None], device=grid.device, dtype=torch.float32)
    return tensor_vdb
