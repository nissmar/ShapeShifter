if True:
    import sys
    sys.path.append('./src/utils')
import mesh_tools as mt
import torch
from tqdm import tqdm
from fvdb_utils import *
from diffusion_tensor import DiffusionTensor
import matplotlib.pyplot as plt
from model import UpSampler
import yaml
import argparse
# from train_diffusion_colors import get_upsampler_data

if __name__ == '__main__':
    device = 'cuda'
    
    parser = argparse.ArgumentParser(
        description='Upsamplers training')
    # Experiment
    parser.add_argument('-model_name',
                        default="house", type=str, help="mesh name")
    parser.add_argument('-level', default=1, type=int,
                        help="number of sudivisions")
    parser.add_argument('-config', type=str,
                        help="config path")
    args = parser.parse_args()


    
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
        
        
    res_1 = cfg["base_resolution"]*2**(args.level-1)
    res_2 = cfg["upsample_fac"]*res_1
    X = torch.load(
        '{}/{}/{}.pt'.format(cfg["src_path"], args.model_name, res_1), weights_only=False)
    Y = torch.load(
        '{}/{}/{}.pt'.format(cfg["src_path"], args.model_name, res_2), weights_only=False)
    X = X.to_batch(cfg["batch_size"])
    Y = Y.to_batch(cfg["batch_size"])
    X_UP = Y.grid

    model_upsampler = UpSampler(
        X.jdata.shape[-1], cfg["features"], X.jdata.shape[-1], cfg["layers"], cfg["upsample_fac"], cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model_upsampler.parameters(),
                                  lr=cfg["lr"]
                                  )
    L = []
    mt.count_parameters(model_upsampler)

    def train_epoch():
        optimizer.zero_grad()
        loss = (((model_upsampler(X, X_UP) - Y).jdata)**2).mean()
        loss.backward()
        optimizer.step()
        L.append(loss.item())

    model_upsampler.train()
    for _ in tqdm(range(cfg["epochs"])):
        train_epoch()

    plt.plot(L, label=args.model_name)
    plt.yscale('log')
    plt.legend()
    plt.savefig(
        'checkpoints/upsamplers_colors/{}_{}.png'.format(args.model_name, args.level))
    model_upsampler.eval()
    torch.save(model_upsampler,
               'checkpoints/upsamplers_colors/{}_{}.pt'.format(args.model_name, args.level))
