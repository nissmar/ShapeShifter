if True:
    import sys
    sys.path.append('./src/utils')
import argparse
import numpy as np
import os
import pymeshlab as ml
import torch
from tqdm import tqdm
import fvdb
import fvdb.nn as fvnn
import mesh_tools as mt
from diffusion_tensor import DiffusionTensor
from fvdb_utils import grid_to_VDB, vdb_marching_cubes


def create_dilated_grid(grid: fvdb.GridBatch):
    dilated_ijk = grid.ijk.jdata.clone()
    dilated_ijk = dilated_ijk[:, None] + \
        torch.tensor(mt.mesh_grid(3, True), device=dilated_ijk.device)
    dilated_ijk = dilated_ijk.view(-1, 3).long()
    return fvdb.sparse_grid_from_ijk(dilated_ijk, origins=grid.origins, voxel_sizes=grid.voxel_sizes)


def label_dilated_crust(source_tensor: fvnn.VDBTensor, dilated_grid: fvdb.GridBatch):
    '''Label dilated grid based on the normal orientation: 0 (intersects surface), 1 or -1 (resp. strictly outside/inside)'''
    dilated_grid = create_dilated_grid(source_tensor.grid)
    dilated_crust_tensor = grid_to_VDB(
        dilated_grid, torch.ones, additional_feat=[1])
    dilated_centers = dilated_grid.grid_to_world(dilated_grid.ijk.float())
    neighbors = source_tensor.grid.neighbor_indexes(
        dilated_crust_tensor.grid.ijk, 1)

    # assign existing voxels to 0
    in_source_mask = neighbors.jdata[:, 1, 1, 1] != -1

    # --> Decide sign
    flat_neighbors = neighbors.jdata.view(-1, 27)
    normals, vstars, _, _ = DiffusionTensor.get_feature_data(
        source_tensor.get_global().jdata)
    local_delta = (dilated_centers.jdata.unsqueeze(1)-vstars[flat_neighbors])
    local_sdf = (local_delta*normals[flat_neighbors]).sum(-1)
    local_sdf[flat_neighbors == -1] = 0
    local_sdf = local_sdf.sum(-1)/(flat_neighbors != -1).sum(-1)

    dilated_crust_tensor.feature.jdata = local_sdf[:, None]
    dilated_crust_tensor.feature.jdata[in_source_mask] = 0

    return dilated_crust_tensor


def label_dual_grid(dilated_grid: fvdb.GridBatch, dilated_crust_tensor: fvnn.VDBTensor, source_tensor: fvnn.VDBTensor):
    '''label dual of dilated grid based on crust information. Disambiguate internal vertices with normal information'''
    dual_grid = dilated_grid.dual_grid()
    dual_centers = dual_grid.grid_to_world(dual_grid.ijk.float())
    dilated_centers = dilated_grid.grid_to_world(dilated_grid.ijk.float())

    new_feature = dual_grid.splat_trilinear(
        dilated_centers, dilated_crust_tensor.feature)
    dual_tensor = fvdb.nn.VDBTensor(dual_grid, new_feature)

    # label unlabeled (internal) voxels
    is_unlabeled = (dual_tensor.feature.jdata == 0).squeeze()
    offset = -1 + torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [
                               1, 0, 1], [0, 1, 1], [1, 1, 1]], device=is_unlabeled.device)

    primal_ijk = (
        dual_tensor.grid.ijk.jdata[is_unlabeled][:, None, :] + offset[None, :]).view(-1, 3)
    primal_index = source_tensor.grid.ijk_to_index(
        primal_ijk, 1).jdata.view(-1, 8)
    normals, vstars, _, _ = DiffusionTensor.get_feature_data(
        source_tensor.get_global().jdata)

    guess_sign = (normals[primal_index]*(dual_centers.jdata[is_unlabeled]
                  [:, None, :]-vstars[primal_index])).sum(-1).mean(-1)

    dual_tensor.feature.jdata[is_unlabeled] = guess_sign.unsqueeze(-1)
    dual_tensor.feature.jdata = torch.sign(dual_tensor.feature.jdata)
    return dual_tensor


def assign_dilated_vstars(source_tensor: fvnn.VDBTensor, dilated_grid: fvdb.GridBatch):
    '''create vstars in adjacent voxels'''
    dilated_grid = create_dilated_grid(source_tensor.grid)
    dilated_tensor = grid_to_VDB(dilated_grid, torch.ones, additional_feat=[
                                 source_tensor.jdata.shape[-1]])

    neighbors = source_tensor.grid.neighbor_indexes(dilated_tensor.grid.ijk, 1)

    # assign existing voxels to 0
    in_source_mask = neighbors.jdata[:, 1, 1, 1] != -1

    # --> Decide sign
    flat_neighbors = neighbors.jdata.view(-1, 27)

    out_feat = source_tensor.jdata[flat_neighbors]
    out_feat[flat_neighbors == -1] = 0

    out_feat = out_feat.sum(-2)/(flat_neighbors != -1).sum(-1, True)

    dilated_tensor.feature.jdata = out_feat
    dilated_tensor.feature.jdata[in_source_mask] = source_tensor.jdata[neighbors.jdata[:,
                                                                                       1, 1, 1][in_source_mask]]

    return dilated_tensor, in_source_mask


def dual_contouring(dual_tensor: fvnn.VDBTensor):
    ''' dual_tensor: sign on corner vertices
        based on dual edges (neighbors in the graph)'''
    dual_neighbors = dual_tensor.grid.neighbor_indexes(
        dual_tensor.grid.ijk, 1).jdata
    sign_data = dual_tensor.jdata.sign().squeeze()
    if 0 in sign_data:
        print('WARNING, WRONG SIGN DATA')

    dkx = dual_neighbors[:, 2, 1, 1]
    active_dkx = ((sign_data*sign_data[dkx]) == -1)*(dkx != -1)

    dky = dual_neighbors[:, 1, 2, 1]
    active_dky = ((sign_data*sign_data[dky]) == -1)*(dky != -1)

    dkz = dual_neighbors[:, 1, 1, 2]
    active_dkz = ((sign_data*sign_data[dkz]) == -1)*(dkz != -1)

    dkx_face_idx = torch.tensor(
        [[0, -1, -1], [0, 0, -1], [0, 0, 0], [0, -1, 0]], device=dkx.device)
    dky_face_idx = torch.tensor(
        [[-1, 0, 0], [0, 0, 0], [0, 0, -1], [-1, 0, -1]], device=dky.device)
    dkz_face_idx = torch.tensor(
        [[-1, -1, 0], [0, -1, 0], [0, 0, 0], [-1, 0, 0]], device=dkz.device)

    faces = []
    for dk_face_idx, active_dk in zip([dkx_face_idx, dky_face_idx, dkz_face_idx], [active_dkx,  active_dky, active_dkz]):
        face_idx = (
            dk_face_idx[:, None] + dual_tensor.grid.ijk.jdata[active_dk][None, :]).permute((1, 0, 2))
        face_idx[sign_data[active_dk] >
                 0] = face_idx[sign_data[active_dk] > 0].flip(1)
        faces.append(face_idx)
    return torch.concatenate(faces)


def faces_ijk_to_index(candidate_faces: torch.tensor, source_grid: fvdb.GridBatch):
    df = source_grid.ijk_to_index(
        candidate_faces.view(-1, 3)).jdata.view(-1, 4)
    return df


def mesh_dc(source_tensor: fvnn.VDBTensor, filepath: str):

    dilated_grid = create_dilated_grid(source_tensor.grid)
    dilated_crust_tensor = label_dilated_crust(source_tensor, dilated_grid)
    dual_tensor = label_dual_grid(
        dilated_grid, dilated_crust_tensor, source_tensor)
    dilated_tensor, in_source_mask = assign_dilated_vstars(
        source_tensor.get_global(), dilated_grid)

    candidate_faces = dual_contouring(dual_tensor)
    faces = faces_ijk_to_index(candidate_faces, dilated_tensor.grid)
    faces = faces[(faces != -1).all(-1)]
    _, vstars, _, _ = DiffusionTensor.get_feature_data(dilated_tensor.jdata)

    ms = ml.MeshSet()
    nmesh = ml.Mesh(vstars.cpu().double().detach().numpy(),
                    faces.cpu().detach().numpy().astype(np.int32))
    ms.add_mesh(nmesh)
    ms.save_current_mesh(filepath)


def mesh_mc(source_tensor: fvnn.VDBTensor, filepath: str):
    dilated_grid = create_dilated_grid(source_tensor.grid)
    dilated_tensor, in_source_mask = assign_dilated_vstars(
        source_tensor.get_global(), dilated_grid)
    dual_grid = dilated_tensor.grid.dual_grid()
    dual_centers = dual_grid.grid_to_world(dual_grid.ijk.float())
    dilated_centers = dilated_grid.grid_to_world(dilated_grid.ijk.float())
    dual_tensor = grid_to_VDB(dual_grid, additional_feat=[1])
    offset = -1 + torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [
                               1, 0, 1], [0, 1, 1], [1, 1, 1]], device=dual_grid.device)
    primal_ijk = (
        dual_tensor.grid.ijk.jdata[:, None, :] + offset[None, :]).view(-1, 3)
    primal_index = dilated_grid.ijk_to_index(primal_ijk, 1).jdata.view(-1, 8)
    normals, vstars, _, _ = DiffusionTensor.get_feature_data(
        dilated_tensor.jdata)

    guess_sign = (normals[primal_index]*(dual_centers.jdata[:,
                  None, :]-vstars[primal_index])).sum(-1)
    guess_sign[primal_index == -1] = 0
    dual_tensor.feature.jdata = guess_sign.sum(
        -1, True)/(primal_index != -1).sum(-1, True)
    v, f = vdb_marching_cubes(dual_tensor)
    ms = ml.MeshSet()
    nmesh = ml.Mesh(v, f)
    ms.add_mesh(nmesh)
    ms.save_current_mesh(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Model')
    # Experiment
    parser.add_argument('-src',
                        default="output/", type=str, help="mesh name")
    parser.add_argument('-out', default=None, type=str,
                        help="output folder")
    parser.add_argument('-level', default=4,
                        type=int)
    parser.add_argument('-num', default=10,
                        type=int)
    parser.add_argument('-target_face_num', default=10000,
                        type=int)
    parser.add_argument('-use_poisson', default=True,
                        type=bool, help='use poisson or APSS')
    args = parser.parse_args()

    SRC = args.src+'/'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.out is None:
        OUT = SRC.replace("output", "textured_output")
    else:
        OUT = args.out

    try:
        os.mkdir(OUT)
    except:
        pass
    with torch.no_grad():
        for name in tqdm(os.listdir(SRC)):
            base_save_path = '{}/{}'.format(OUT, name)
            data_mat = '/home/nmaruani/ShapeShifter/data/materials/{}.mtl'.format(
                name)
            try:
                os.mkdir(base_save_path)
            except:
                pass
            for i in range(args.num):
                save_path = '{}/00{}'.format(base_save_path, i)
                try:
                    os.mkdir(save_path)
                except:
                    pass

                tens = torch.load(
                    '{}/{}/gen_{}_{}.pt'.format(SRC, name, i, args.level), weights_only=False)

                source_tensor = DiffusionTensor(
                    tens.grid[i], tens.feature[i]).remove_mask()

                mesh_dc(source_tensor, '{}/dc.ply'.format(save_path))
                mesh_mc(source_tensor, '{}/mc.ply'.format(save_path))
                break
