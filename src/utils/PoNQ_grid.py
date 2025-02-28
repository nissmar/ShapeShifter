import torch
import numpy as np
import fvdb

# QEM Utilities


def vstars_from_quadrics(Q, P, eps=.05):
    """Compute optimal vertices positions

    Parameters
    ----------
    Q: torch.tensor 
        Nx4x4 of quadric matrices
    P: torch.tensor 
        Nx3 tensor of positions

    Returns
    -------
    vstars: torch.tensor 
        Nx3 optimal vertices 
    eigs: torch.tensor 
        Nx3 eigen values of quadric matrices
    """
    A = Q[:, :3, :3]
    b = -Q[:, 3, :3]
    u, eigs, vh = torch.linalg.svd(A)
    eigs2 = torch.zeros_like(eigs)
    mask_s = eigs / eigs[:, 0, None] > eps
    eigs2[mask_s] = 1 / eigs[mask_s]
    base_pos = P
    vstars = base_pos + (
        vh.transpose(1, 2)
        @ torch.diag_embed(eigs2)
        @ u.transpose(1, 2)
        @ (b[..., None] - A @ base_pos[..., None])
    ).squeeze(-1)
    return vstars, vh, eigs


def get_flat_Qs(bmat):
    indices4x4 = torch.triu_indices(4, 4, device=bmat.device)
    return bmat[:, indices4x4[0], indices4x4[1]]


def get_square_Qs(flat_As):
    indices4x4 = torch.triu_indices(4, 4, device=flat_As.device)
    Q = torch.zeros((len(flat_As), 4, 4), device=flat_As.device)
    Q[:, indices4x4[0], indices4x4[1]] = flat_As
    Q.transpose(1, 2)[:, indices4x4[0], indices4x4[1]] = flat_As
    return Q


class PoNQ_grid:
    def __init__(self, grid_size: int) -> None:
        self.grid = None
        self.Qs = None
        self.normals = None
        self.colors = None
        self.grid_size = grid_size

    @property
    def feature(self) -> fvdb.JaggedTensor:
        return fvdb.JaggedTensor(torch.cat((get_flat_Qs(self.Qs), self.normals, self.colors), -1))

    @property
    def voxel_centers(self) -> fvdb.JaggedTensor:
        return self.grid.grid_to_world(self.grid.ijk.float())

    def from_feature(self, grid, feature):
        self.grid = grid
        self.Qs = get_square_Qs(feature.jdata[:, :-6])
        self.normals = feature.jdata[:, -6:-3]
        self.colors = feature.jdata[:, -3:]

    def from_mesh(self, grid, points, normals, colors, device='cuda'):
        self.grid = grid
        v_stars = torch.tensor(points, dtype=torch.float32, device=device)
        normals = torch.tensor(normals, dtype=torch.float32, device=device)
        ps = torch.cat((normals, -(normals * v_stars).sum(-1)[..., None]), -1)
        Qs = (torch.matmul(ps[..., :, None], ps[..., None, :]))
        # eps = 1e10
        # Qs[..., np.arange(3), np.arange(3)] += eps
        # Qs[..., 3, :3] += -eps*v_stars
        # Qs[..., :3, 3] += -eps*v_stars
        # trace = (Qs[..., np.arange(3), np.arange(3)]).sum(-1, True)
        # Qs = Qs/trace[..., None]
        self.Qs = Qs
        self.normals = normals
        self.colors = torch.tensor(colors, dtype=torch.float32, device=device)

    def get_pool(self, k_s=2):
        '''Returns new PoNQ Grid'''
        feature, grid = self.grid.avg_pool(k_s, self.feature, k_s)
        new_grid_size = self.grid_size//k_s
        new_grid = PoNQ_grid(new_grid_size)
        new_grid.from_feature(grid, feature)
        # Divide by trace to preserve mean
        trace = (new_grid.Qs[..., np.arange(3), np.arange(3)]).sum(-1, True)
        new_grid.normals /= trace
        new_grid.colors /= trace
        new_grid.Qs /= trace[..., None]
        return new_grid

    def clip_to_voxel(self, new_points, min_value=1e-10):
        points = self.voxel_centers.jdata
        min_qem_diff = new_points-points
        mag = torch.clip(
            min_qem_diff.abs().max(-1, keepdims=True).values, min=min_value)
        div = torch.clip(mag, max=self.grid.voxel_sizes.max()/2)
        return points + min_qem_diff/mag*div

    def compute_local_offset(self, eps=.05, clip=True):
        vstars, _, _ = vstars_from_quadrics(
            self.Qs,  self.voxel_centers.jdata, eps=eps)
        if clip:
            vstars = self.clip_to_voxel(vstars)
        self.local_offset = (vstars-self.voxel_centers.jdata) / \
            self.grid.voxel_sizes.max()
