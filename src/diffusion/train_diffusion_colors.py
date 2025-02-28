if True:
    import sys
    sys.path.append('./src/utils')
from diffusion_tensor import DiffusionTensor
from fvdb_utils import *
from fvdb_diffusion import SparseDiffusion
from model import DiffusionCNN
from datetime import datetime
import fvdb.nn as fvnn
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
    diffusion = SparseDiffusion(
        model,
        timesteps=cfg["diffusion_timesteps"],
        max_T=cfg["max_T"] if args.level > 0 else None,
        loss=nn.functional.mse_loss,
        model_upsampler=model_upsampler if args.level > 0 else None,
    ).cuda()

    diffusion.args = args
    diffusion.cfg = cfg
    current_time = datetime.today().strftime('%d-%m-%H:%M')

    def train_epoch(optimizer, diffusion):
        global LOSS_EMA
        optimizer.zero_grad()
        if args.level > 0:
            loss = diffusion(*clip_data(X0, X0_BLUR, cfg["clip_size"]))
        else:
            loss = diffusion(X0)
        torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.)
        loss.backward()
        optimizer.step()
        if LOSS_EMA is None:
            LOSS_EMA = loss.item()
        else:
            LOSS_EMA = 0.99 * LOSS_EMA + 0.01 * loss.item()
        L.append(LOSS_EMA)

    model.train()
    for i in tqdm(range(cfg["epochs"])):
        train_epoch(optimizer, diffusion)

        if i % cfg["save_every"] == 0 or i == cfg["epochs"]-1:
            plt.clf()
            plt.plot(L, label=args.model_name)
            plt.yscale('log')
            plt.legend()
            plt.savefig(
                'experiments/{}_{}_{}.png'.format(args.model_name, args.level, current_time))

    torch.save(diffusion, 'experiments/{}_{}_{}.pt'.format(
        args.model_name, args.level, current_time))
    print('experiments/{}_{}_{}.pt'.format(args.model_name,
          args.level, current_time))
