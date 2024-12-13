import torch.nn as nn
import torch
import fvdb
import fvdb.nn as fvnn

def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    emb = 2**emb * torch.pi
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)

class DiffusionCNN(nn.Module):
    def __init__(self, channels, layers=2, time_emb=6, one_layers=1, first_ks=3, in_channels=1, out_channels=1, dropout=.01):
        super(DiffusionCNN, self).__init__()
        self.out_channels = out_channels
        self.time_emb = time_emb
        self.net = [
            fvnn.SparseConv3d(in_channels+self.time_emb,
                              channels, kernel_size=first_ks, stride=1),
            fvnn.Dropout(dropout),
            fvnn.SiLU(inplace=True)]
        for _ in range(layers-1):
            self.net += [
                fvnn.SparseConv3d(channels, channels, kernel_size=3, stride=1),
                fvnn.Dropout(dropout),
                fvnn.SiLU(inplace=True)
            ]
        for _ in range(one_layers):
            self.net += [
                fvnn.SparseConv3d(channels, channels, kernel_size=1, stride=1),
                fvnn.SiLU(inplace=True)
            ]
        self.net.append(fvnn.SparseConv3d(
            channels, out_channels, kernel_size=1, stride=1))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, t, cond=None):
        t = sinusoidal_embedding(t, self.time_emb)
        new_x = fvnn.VDBTensor(x.grid, x.grid.jagged_like(
            torch.cat((x.feature.jdata, t), -1)))
        return self.net(new_x)


class UpSampler(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, encoder_layers: int = 3, mult=2, dropout=.05):
        super().__init__()
        encoder = [fvnn.SparseConv3d(in_channels, hidden_channels, kernel_size=3, stride=1),
                   fvnn.Dropout(dropout),
                   fvnn.ReLU(inplace=True)]

        for _ in range(encoder_layers-1):
            encoder += [
                fvnn.SparseConv3d(
                    hidden_channels, hidden_channels, kernel_size=3, stride=1),
                fvnn.Dropout(dropout),
                fvnn.ReLU(inplace=True)
            ]
        self.encoder = nn.Sequential(*encoder)

        self.t_conv = fvnn.SparseConv3d(
            hidden_channels, hidden_channels, kernel_size=mult, stride=mult, transposed=True)

        self.decoder = nn.Sequential(*[
            fvnn.Dropout(dropout),
            fvnn.ReLU(inplace=True),
            fvnn.SparseConv3d(hidden_channels, hidden_channels,
                              kernel_size=1, stride=1),
            fvnn.Dropout(dropout),
            fvnn.ReLU(inplace=True),
            fvnn.SparseConv3d(hidden_channels, out_channels, kernel_size=1, stride=1)])
        
    def forward(self, input: fvnn.VDBTensor, x_upsample: fvnn.VDBTensor):
        x = self.encoder(input)
        x = self.t_conv(x, out_grid=x_upsample.grid)
        return self.decoder(x) + x_upsample
