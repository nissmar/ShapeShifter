
import torch
import torch.nn as nn
from einops import repeat
import torch.nn.functional as F
from torch.special import expm1
import math
import fvdb.nn as fvnn
from tqdm import tqdm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, s: float = 0.008):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class SparseDiffusion(nn.Module):  # Inspired by bitfusion by lucidrain
    def __init__(
        self,
        model,
        timesteps=1000,
        max_T=None,
        noise_schedule='cosine',
        time_difference=0.,
        loss=F.mse_loss,
        model_upsampler=None,
    ):
        super().__init__()
        self.model = model
        self.model_upsampler = model_upsampler
        self.channels = 1

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.timesteps = timesteps
        if max_T is None:
            max_T = timesteps
        self.max_T = max_T
        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400
        self.time_difference = time_difference
        self.loss = loss

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device, steps=None):
        if steps is None:
            steps = self.max_T + 1
        times = torch.linspace(self.max_T/self.timesteps,
                               0., steps, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, noisy_grid: fvnn.VDBTensor, X_Blur: fvnn.VDBTensor = None, clip=None):

        time_pairs = self.get_sampling_timesteps(1, device=self.device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):

            # add the time delay
            time_next = (time_next).clamp(min=0.)

            # get predicted x0
            x_start = self.model(noisy_grid, time.repeat(len(noisy_grid.jidx)))
            if not clip is None:
                x_start.feature.jdata = torch.clip(x_start.jdata, -clip, clip)
            if time_next == 0:
                return x_start

            # Optionnal: clip x0
            if not X_Blur is None:
                gamma = time_next/self.max_T
                start_data = (1-gamma[:, None])*x_start.jdata + \
                    gamma[:, None]*X_Blur.jdata
            else:
                start_data = x_start.jdata

            # get log(snr)
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # derive posterior mean and variance
            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (noisy_grid.jdata *
                                 (1 - c) / alpha + c * start_data)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise
            noise = torch.randn_like(noisy_grid.jdata)
            noisy_grid.feature.jdata = mean + \
                (0.5 * log_variance).exp() * noise

    @torch.no_grad()
    def ddim_sample(self, noisy_grid: fvnn.VDBTensor, steps=None):
        time_difference = self.time_difference
        time_pairs = self.get_sampling_timesteps(
            1, device=self.device, steps=steps)

        x_start = None

        for times, times_next in tqdm(time_pairs, desc='sampling loop time step'):
            # add the time delay
            times_next = (times_next - time_difference).clamp(min=0.)

            # get times and noise levels
            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # predict x0
            x_start = self.model(
                noisy_grid, times.repeat(len(noisy_grid.jidx)))
            if times_next == 0:
                return x_start

            # get predicted noise
            pred_noise_jdata = (noisy_grid.jdata - alpha *
                                x_start.jdata) / sigma.clamp(min=1e-8)
            noisy_grid.feature.jdata = x_start.jdata * \
                alpha_next + pred_noise_jdata * sigma_next

        return (noisy_grid)

    @torch.no_grad()
    def sample(self, noisy_grid: fvnn.VDBTensor):
        return self.ddim_sample(noisy_grid)

    def q_sample(self, X: fvnn.VDBTensor, times: torch.tensor, X_Blur: fvnn.VDBTensor = None):
        assert len(times) == len(X.feature.jidx)
        # compute constant
        noise_level = self.log_snr(times)
        alpha, sigma = log_snr_to_alpha_sigma(noise_level)

        # random noise
        noise = torch.randn_like(X.jdata)

        # compute gamma
        if not X_Blur is None:
            gamma = times/self.max_T
            target_X = (1-gamma[:, None])*X.jdata + gamma[:, None]*X_Blur.jdata
        else:
            target_X = X.jdata

        # corrupted X
        noised_img = alpha[:, None] * target_X + sigma[:, None] * noise
        return fvnn.VDBTensor(grid=X.grid, feature=X.grid.jagged_like(noised_img)), target_X

    def forward(self, X: fvnn.VDBTensor, X_Blur: fvnn.VDBTensor = None):

        # random times
        times = torch.zeros((X.grid_count,), device=self.device).float().uniform_(
            0., self.max_T/self.timesteps)
        times = times[X.feature.jidx.long()]

        noisy_latents, target_X = self.q_sample(X, times, X_Blur)

        # prediction
        pred = self.model(noisy_latents, times)
        # return self.loss(pred.jdata, target_X) # TODO replace with target_X?
        return self.loss(pred.jdata, X.jdata)  # TODO replace with target_X?
