import fvdb
import torch
from meshplot import plot
import numpy as np


def make_gaussian_filter(feat=1, g_size=9):
    def normal_pdf(x, mean=0, std=1):
        return 1/(std**3*np.sqrt((2*np.pi)**3))*torch.exp(-((x-mean)**2).sum(-1)/(2*std**2))
    gaussian_filter = fvdb.nn.SparseConv3d(
        feat, feat, kernel_size=g_size, stride=1, bias=False)
    xx, yy, zz = np.mgrid[:g_size, :g_size, :g_size]
    mesh_grid = np.column_stack(
        (xx.flatten(), yy.flatten(), zz.flatten()))-(g_size-1)//2
    gaussian_grid = torch.tensor(mesh_grid, dtype=torch.float32)
    gaussian_kernel = normal_pdf(
        gaussian_grid, std=1).reshape(g_size, g_size, g_size)
    gaussian_filter.weight.data *= 0
    gaussian_filter.weight.data[np.arange(
        feat), np.arange(feat), ...] = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def blur_tensor(X, iterations=1, blur_kernel=9):
    with torch.no_grad():
        blur_filter = make_gaussian_filter(
            X.jdata.shape[-1], blur_kernel).to(X.device)
        input_tens = None
        for _ in range(iterations):
            if input_tens is None:
                input_tens = blur_filter(X)
            else:
                input_tens = blur_filter(input_tens)
        return DiffusionTensor.from_vdb(input_tens)


class DiffusionTensor(fvdb.nn.VDBTensor):
    """features: normals, offset (local or global), mask"""
    @staticmethod
    def from_vdb(vdb_tensor: fvdb.nn.VDBTensor):
        return DiffusionTensor(vdb_tensor.grid, vdb_tensor.feature)

    @staticmethod
    def get_feature_data(jdata):
        """
        Returns
        --------
        normals = jdata[:, :3]

        offset = jdata[:, 3:6]

        mask = jdata[:, -1:]
        """
        normals = jdata[:, :3]
        offset = jdata[:, 3:6]
        colors = jdata[:, 6:9]
        mask = jdata[:, -1:]
        return normals, offset, colors, mask

    @staticmethod
    def get_tensor_from_data(grid, normals, local_offset, colors, mask):
        return DiffusionTensor(grid, grid.jagged_like(torch.cat((normals, local_offset, colors, mask), -1)))

    @staticmethod
    def fill_upsampled_with_gt(trilinear_upsampled_tensor, gt_fine_tensor):
        target_tensor = (trilinear_upsampled_tensor*1)
        target_tensor.jdata[..., -1] = -1
        to_change_idx = target_tensor.grid.ijk_to_index(
            gt_fine_tensor.grid.ijk).jdata
        target_tensor.feature.jdata[to_change_idx] = gt_fine_tensor.jdata
        return DiffusionTensor.from_vdb(target_tensor)

    def clip(self):
        self.feature.jdata[:, 3:6] = torch.clip(
            self.feature.jdata[:, 3:6], -.5, .5)

    def trilinear_upsample(self, subdiv_factor=2, normalize_normals=False):
        """Input
        -------
        self: "clean" DiffusionTensor (1 mask)

        Returns
        -------
        Trilinealy interpolated DiffusionTensor
        """

        assert len(self.jdata[..., -1].unique()) == 1
        diff_tens = self.get_global()

        up_grid = self.grid.subdivided_grid(subdiv_factor=subdiv_factor)
        new_centers = up_grid.grid_to_world(up_grid.ijk.float())
        up_feat = self.grid.sample_trilinear(new_centers, diff_tens.feature)
        normalized_normals, global_offset, colors, mask = self.get_feature_data(
            up_feat.jdata)
        if normalize_normals:
            normalized_normals /= (normalized_normals **
                                   2).sum(-1, keepdims=True).sqrt()
        else:
            normalized_normals /= mask

        colors /= mask
        global_offset /= mask
        # mask = 2*mask-1 # normalize to -1, 1
        diff_tens = DiffusionTensor.get_tensor_from_data(
            up_grid, normalized_normals, global_offset, colors, mask)
        diff_tens = diff_tens.get_local()
        # mask based on offset
        new_mask = 1.-2.*diff_tens.jdata[..., 3:6].abs().max(-1).values
        diff_tens.jdata[..., -1] = torch.clamp(new_mask, -1., 1.)
        return diff_tens

    def get_global(self):
        normals, local_offset, colors, mask = self.get_feature_data(
            self.jdata)
        voxel_centers = self.grid.grid_to_world(self.grid.ijk.float())
        global_offset = local_offset*self.grid.voxel_sizes.max()+voxel_centers.jdata
        return DiffusionTensor.get_tensor_from_data(self.grid, normals, global_offset, colors, mask)

    def get_local(self):
        normals, global_offset, colors, mask = self.get_feature_data(
            self.jdata)
        voxel_centers = self.grid.grid_to_world(self.grid.ijk.float())
        local_offset = (global_offset-voxel_centers.jdata) / \
            self.grid.voxel_sizes.max()
        return DiffusionTensor.get_tensor_from_data(self.grid, normals, local_offset, colors, mask)

    def to_custom_dense(self, blur_kernel=9):
        '''Last coordinate (mask) set to -1'''
        dense_x = self.to_dense()
        # set last coordinate to -1
        mask = (dense_x[..., -1]) == 0
        dense_x_flat = dense_x.view(-1, dense_x.shape[-1])
        dense_x_flat[mask.flatten(), -1] = -1
        dense_x = dense_x_flat.view(dense_x.shape)
        ijk_min = self.grid.ijk.jdata.min(0).values
        vdb_tensor = self.from_dense(
            dense_x, ijk_min=ijk_min, origins=self.grid.origins, voxel_sizes=self.grid.voxel_sizes)

        # add blur
        to_change = vdb_tensor.jdata[..., -1] < 0
        blur_x = blur_tensor(vdb_tensor, blur_kernel=blur_kernel)
        # blur_x.feature.jdata[..., :-1] /= blur_x.jdata.abs().max(0).values[None, :-1]
        blur_x.feature.jdata[..., -1] = -1
        vdb_tensor.feature.jdata[to_change] = blur_x.feature.jdata[to_change]
        return DiffusionTensor.from_vdb(vdb_tensor)

    def to_batch(self, batch_size=1):
        return fvdb.nn.VDBTensor(
            fvdb.jcat([self.grid for _ in range(batch_size)]),
            fvdb.jcat([self.jdata for _ in range(batch_size)])
        )

    def remove_mask(self, threshold=0):
        in_mask = [self.feature[i].jdata[:, -1] >
                   threshold for i in range(self.grid_count)]
        jagged_ijks = fvdb.JaggedTensor(
            [self.grid[i].ijk.jdata[in_mask[i]] for i in range(self.batch_size)])
        new_grid = fvdb.sparse_grid_from_ijk(
            jagged_ijks, origins=self.grid.origins, voxel_sizes=self.grid.voxel_sizes)
        feat = fvdb.JaggedTensor(
            [self.feature[i].jdata[in_mask[i]] for i in range(self.batch_size)])
        feat.jdata[..., -1] = 1
        return DiffusionTensor(new_grid, feat)

    def colored_PC(self, point_size=None, return_plot=True, use_normals=False):
        normals, global_offset, colors, mask = self.get_feature_data(
            self.jdata)
        if use_normals:
            normals = normals.cpu().detach().numpy()
            normals /= np.sqrt((normals ** 2).sum(-1, keepdims=True))
            c = normals
        else:
            c = colors.cpu().detach().numpy()/2

        c = (1+c)/2.
        vstars = global_offset.cpu().detach().numpy()
        if return_plot:
            return plot(vstars, c=c, shading={"point_size": point_size})
        return vstars, normals, c
