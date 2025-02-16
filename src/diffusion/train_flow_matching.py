if True:
    import sys
    sys.path.append('./src/utils')
from diffusion_tensor import DiffusionTensor
from fvdb_utils import *
from fvdb_diffusion import SparseDiffusion
from model import DiffusionCNN
from datetime import datetime
import fvdb.nn as fvnn
import igl
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import mesh_tools as mt
import yaml


def clip_data(X0, X0_BLUR, size):
    ind = torch.randint(0, len(X0.grid.ijk.jdata), (X0.grid_count,))
    centers = X0.grid.ijk.jdata[ind]
    new_ijk_min = centers - size
    new_ijk_max = centers + size
    cf, cg = X0.grid.clip(X0.feature, new_ijk_min, new_ijk_max)
    new_X0 = fvnn.VDBTensor(cg, cf)
    cf, cg = X0_BLUR.grid.clip(X0_BLUR.feature, new_ijk_min, new_ijk_max)
    new_X0_BLUR = fvnn.VDBTensor(cg, cf)
    return new_X0, new_X0_BLUR


def get_gt_data(cfg, level, model_name):
    if level == 0:
        res_1 = cfg["base_resolution"]
        X0 = torch.load(
            '{}/{}/{}.pt'.format(cfg["src_path"], model_name, res_1), weights_only=False)
        return X0.to_custom_dense().to_batch(cfg["batch_size"])
    else:
        res_1 = cfg["base_resolution"]*2**(level-1)
        res_2 = cfg["upsample_fac"]*res_1
        X = torch.load(
            '{}/{}/{}.pt'.format(cfg["src_path"], model_name, res_1), weights_only=False)
        X0 = torch.load(
            '{}/{}/{}.pt'.format(cfg["src_path"], model_name, res_2), weights_only=False)
        X_UP = X.trilinear_upsample(cfg["upsample_fac"])
        X0 = DiffusionTensor.fill_upsampled_with_gt(X_UP, X0)
        X = X.to_batch(cfg["batch_size"])
        X0 = X0.to_batch(cfg["batch_size"])
        X_UP = X_UP.to_batch(cfg["batch_size"])
        X_UP.grid = X0.grid
        return X, X_UP, X0


class SparseFlowMatching(nn.Module):
    def __init__(
        self,
        model,
        timesteps=1000,
        blur_fac=.8,
        loss=nn.functional.mse_loss,
        model_upsampler=None,
    ):
        super().__init__()
        self.model = model
        self.model_upsampler = model_upsampler
        self.channels = 1
        self.timesteps = timesteps
        self.blur_fac = blur_fac
        self.loss = loss

    def add_x0_noise(self, X0_BLUR):
        return (self.blur_fac)*X0_BLUR.jdata + (1-self.blur_fac)*torch.randn_like(X0_BLUR.jdata)

    def sample_xt(self, t, X0, X0_BLUR=None):
        x_1 = X0.jdata
        x_0 = torch.randn_like(
            X0.jdata) if X0_BLUR is None else self.add_x0_noise(X0_BLUR)
        x_t = (1 - t) * x_0 + t * x_1
        return fvnn.VDBTensor(grid=X0.grid, feature=X0.grid.jagged_like(x_t))

    def forward(self, X0, X0_BLUR=None):
        t = torch.rand(X0.grid_count, 1, device=X0.device)
        t = t[X0.feature.jidx.long()]
        XT = self.sample_xt(t, X0, X0_BLUR)
        return self.loss(self.model(XT, t.flatten()).jdata, X0.jdata)

    # Reverse
    def p1_to_flow(self, XT, T):
        p1 = self.model(XT, T)
        p1.feature.jdata = (p1.jdata-XT.jdata)/(1-T[:, None])
        return p1

    @torch.no_grad()
    def reverse_step(self, XT, t_start, t_end):
        TSTART = t_start.view(1).expand(len(XT.jdata))
        TEND = t_end.view(1).expand(len(XT.jdata))
        p1 = self.p1_to_flow(XT, TSTART)
        p1.feature.jdata *= (TEND-TSTART)[:, None]/2.
        p1.feature.jdata += XT.jdata
        p2 = self.p1_to_flow(p1, TSTART + (TEND-TSTART)/2.)
        p2.feature.jdata *= (TEND-TSTART)[:, None]
        p2.feature.jdata += XT.jdata
        return p2

    @torch.no_grad()
    def reverse_sample(self, XB, n_steps):
        time_steps = torch.linspace(0, 1.0, n_steps + 1, device=XB.device)
        self.model.eval()
        for i in tqdm(range(n_steps-1)):
            XB = self.reverse_step(XB, time_steps[i], time_steps[i+1])
        XB = self.model(XB, time_steps[-2].view(1).expand(len(XB.jdata)))
        return XB


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser(
        description='Diffusion training with colors')
    # Experiment
    parser.add_argument('-model_name', type=str, help="mesh name")
    parser.add_argument('-level', type=int,
                        help="number of sudivisions")
    parser.add_argument('-config', type=str,
                        help="config path")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    if args.level > 0:
        X, X_UP, X0 = get_gt_data(cfg, args.level, args.model_name)
        model_upsampler = torch.load(
            './checkpoints/upsamplers_colors/{}_{}.pt'.format(args.model_name, args.level), weights_only=False)

        # ABLATION:UPSAMPLER
        # model_upsampler = torch.load(
        #     './checkpoints/stupid_upsampler/stupid.pt', weights_only=False)

        model_upsampler.eval()
        with torch.no_grad():
            X0_BLUR = model_upsampler(X, X_UP).detach()
        X0_BLUR.grid = X0.grid

    else:
        X0 = get_gt_data(cfg, args.level, args.model_name)

    # Neural network
    L = []
    LOSS_EMA = None
    model = DiffusionCNN(channels=cfg["features"], layers=cfg["layers"], time_emb=cfg["time_emb"],
                         one_layers=cfg["one_layers"], first_ks=cfg["first_ks"],
                         in_channels=X0.jdata.shape[-1], out_channels=X0.jdata.shape[-1]).to(device)
    mt.count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["lr"]
                                 )
    sparse_fm = SparseFlowMatching(
        model,
        timesteps=cfg["diffusion_timesteps"],
        loss=nn.functional.mse_loss,
        model_upsampler=model_upsampler if args.level > 0 else None,
    ).cuda()

    sparse_fm.args = args
    sparse_fm.cfg = cfg
    current_time = datetime.today().strftime('%d-%m-%H:%M')

    def train_epoch(optimizer, sparse_fm):
        global LOSS_EMA
        optimizer.zero_grad()
        if args.level > 0:
            loss = sparse_fm(*clip_data(X0, X0_BLUR, cfg["clip_size"]))
        else:
            loss = sparse_fm(X0)
        torch.nn.utils.clip_grad_norm_(sparse_fm.model.parameters(), 1.)
        loss.backward()
        optimizer.step()
        if LOSS_EMA is None:
            LOSS_EMA = loss.item()
        else:
            LOSS_EMA = 0.99 * LOSS_EMA + 0.01 * loss.item()
        L.append(LOSS_EMA)

    model.train()
    for i in tqdm(range(cfg["epochs"])):
        train_epoch(optimizer, sparse_fm)

        if i % cfg["save_every"] == 0 or i == cfg["epochs"]-1:
            plt.clf()
            plt.plot(L, label=args.model_name)
            plt.yscale('log')
            plt.legend()
            plt.savefig(
                'experiments_fm/{}_{}_{}.png'.format(args.model_name, args.level, current_time))

    torch.save(sparse_fm, 'experiments_fm/{}_{}_{}.pt'.format(
        args.model_name, args.level, current_time))
    print('experiments_fm/{}_{}_{}.pt'.format(args.model_name,
          args.level, current_time))
