if True:
    import sys
    sys.path.append('../utils')
import trimesh
import igl
import torch
from diffusion_tensor import DiffusionTensor
from PoNQ_grid import PoNQ_grid
from fvdb_utils import mesh_to_grid
import mesh_tools as mt
import point_cloud_utils as pcu
import numpy as np
import os
import argparse


class MeshSampler:
    def __init__(self, mesh_name="canyon", wt_path="../../data/GT_WT", glb_path="../../data/GT_GLB", clip_uv=False):
        # Load WT mesh
        self.v, self.f = igl.read_triangle_mesh(
            "{}/{}.obj".format(wt_path, mesh_name))
        self.n = igl.per_face_normals(
            self.v, self.f, np.array([1e-10, 1e-10, 1e-10]))

        # Load GLB mesh
        glb_mesh = trimesh.load(
            "{}/{}.glb".format(glb_path, mesh_name), force="mesh", process=False)
        self.glb = glb_mesh
        self.glb_v = glb_mesh.vertices
        self.glb_f = glb_mesh.faces
        self.glb_uv = glb_mesh.visual.uv
        self.to_color = glb_mesh.visual.material.to_color
        self.clip_uv = clip_uv
        print('Material factors: roughness {}, metallic {}'.format(
            glb_mesh.visual.material.roughnessFactor, glb_mesh.visual.material.metallicFactor))
        self.normalize()

    def normalize(self):
        self.v = 2*mt.NDCnormalize(self.v)
        self.glb_v = 2*mt.NDCnormalize(self.glb_v)

    def get_closest_watertight_samples(self, grid_points):
        sdf_compute = igl.signed_distance(
            grid_points, self.v, self.f, return_normals=True)
        normals = sdf_compute[3]
        if (normals != normals).any():
            c_normals = (grid_points-sdf_compute[2])/sdf_compute[0][:, None]
            c_normals /= np.sqrt((c_normals**2).sum(-1, keepdims=True))
            print('warning: {} nan normals'.format((normals != normals).sum()))
            normals[np.isnan(normals.sum(-1))
                    ] = c_normals[np.isnan(normals.sum(-1))]
        return sdf_compute[2], normals

    def get_closest_color(self, sampled_points, no_color=False):
        """returns Nx3 colors in [0, 1]"""
        if no_color:
            return 0*sampled_points+.5
        _, glb_fi, glb_bc = pcu.closest_points_on_mesh(
            sampled_points, self.glb_v, self.glb_f)
        uv_query = pcu.interpolate_barycentric_coords(
            self.glb_f, glb_fi, glb_bc, self.glb_uv)
        uv_query[np.isnan(uv_query)] = 0
        if (uv_query.min() < 0 or uv_query.max() > 1) and not self.clip_uv:
            print(
                f'WARNING: UV OUT OF BOUNDS {uv_query.min()} {uv_query.max()}')
        if self.clip_uv:
            uv_query = np.clip(uv_query, 0, 1)

        colors = self.to_color(uv_query)
       
        return colors[:, :3]/255.

    def get_grid_and_samples(self, n=512, device='cuda', no_color=False):
        grid = mesh_to_grid(self.v, self.f, n, device)
        grid_points = grid.grid_to_world(
            grid.ijk.double()).jdata.cpu().detach().numpy()
        sampled_points, sampled_normals = self.get_closest_watertight_samples(
            grid_points)
        sampled_colors = self.get_closest_color(sampled_points, no_color)
        return grid, sampled_points, sampled_normals, sampled_colors


def process_GT(name, size=1024, targets=[512, 256, 128, 64, 32, 16], base_save_path="../../data/GT_sparse_tensors/", no_color=False):
    print(f"Processing {name}")
    ms = MeshSampler(name)
    grid, sampled_points, sampled_normals, sampled_colors = ms.get_grid_and_samples(
        1024, 'cpu', no_color)
    grid = grid.to('cuda')
    if name == "house" or name == "small-town":
        print("invert")
        sampled_colors = 1-sampled_colors
    # scaling important for diffusion model
    sampled_colors = 2*((2*sampled_colors)-1)
    large_base_grid = PoNQ_grid(size)
    large_base_grid.from_mesh(grid, sampled_points,
                              sampled_normals, sampled_colors)
    # save
    save_path = f"{base_save_path}/{name}"
    for t_size in targets:
        try:
            os.mkdir(save_path)
        except:
            pass
        large_base_grid = large_base_grid.get_pool(size//t_size)
        large_base_grid.compute_local_offset()
        DT = DiffusionTensor.get_tensor_from_data(large_base_grid.grid, large_base_grid.normals, large_base_grid.local_offset,
                                                  large_base_grid.colors, torch.ones_like(large_base_grid.local_offset[:, :1]))
        size = t_size
        torch.save(DT, f"{save_path}/{t_size}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess Model')
    # Experiment
    parser.add_argument('-name',
                        default=None, type=str, help="Shape name")
    parser.add_argument('-no_color', default=False, type=bool,
                        help="train without color")
    args = parser.parse_args()
    # Process the whole dataset
    
    if args.no_color:
        print("Not using color")
        
    try:
        os.mkdir('../../data/GT_sparse_tensors')
    except:
        pass
    if args.name is None:
        names = [e[:-4] for e in os.listdir('../../data/GT_GLB')]
        for name in names:
            process_GT(name)
    else:
        process_GT(args.name, no_color=args.no_color)
    
    
