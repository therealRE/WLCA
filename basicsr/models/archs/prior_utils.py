import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


eps = 1e-5


def gaussian_basis_filters(scale, device, dtype, k=3):
    scale = torch.as_tensor(scale, device=device, dtype=dtype)
    base = torch.tensor(2.0, device=device, dtype=dtype)
    std = torch.pow(base, scale)

    filter_radius = max(1, int(torch.ceil(k * std + 0.5).item()))
    coords = torch.arange(-filter_radius, filter_radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')

    g = torch.exp(-(xx / std) ** 2 / 2) * torch.exp(-(yy / std) ** 2 / 2)
    g = g / (torch.sum(g) + eps)

    dgdx = -xx / (std ** 3 * 2 * math.pi) * torch.exp(-(xx / std) ** 2 / 2) * torch.exp(-(yy / std) ** 2 / 2)
    dgdx = dgdx / (torch.sum(torch.abs(dgdx)) + eps)

    dgdy = -yy / (std ** 3 * 2 * math.pi) * torch.exp(-(yy / std) ** 2 / 2) * torch.exp(-(xx / std) ** 2 / 2)
    dgdy = dgdy / (torch.sum(torch.abs(dgdy)) + eps)

    basis_filter = torch.stack([g, dgdx, dgdy], dim=0)[:, None, :, :]
    return basis_filter



def E_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    return Ex ** 2 + Ey ** 2 + Elx ** 2 + Ely ** 2 + Ellx ** 2 + Elly ** 2



def W_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Wx = Ex / (E + eps)
    Wlx = Elx / (E + eps)
    Wllx = Ellx / (E + eps)
    Wy = Ey / (E + eps)
    Wly = Ely / (E + eps)
    Wlly = Elly / (E + eps)
    return Wx ** 2 + Wy ** 2 + Wlx ** 2 + Wly ** 2 + Wllx ** 2 + Wlly ** 2



def C_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Clx = (Elx * E - El * Ex) / (E ** 2 + eps)
    Cly = (Ely * E - El * Ey) / (E ** 2 + eps)
    Cllx = (Ellx * E - Ell * Ex) / (E ** 2 + eps)
    Clly = (Elly * E - Ell * Ey) / (E ** 2 + eps)
    return Cllx ** 2 + Clly ** 2 + Clx ** 2 + Cly ** 2



def N_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Nlx = (Elx * E - El * Ex) / (E ** 2 + eps)
    Nly = (Ely * E - El * Ey) / (E ** 2 + eps)
    Nllx = (Ellx * E ** 2 - Ell * Ex * E - 2 * Elx * El * E + 2 * El ** 2 * Ex) / (E ** 3 + eps)
    Nlly = (Elly * E ** 2 - Ell * Ey * E - 2 * Ely * El * E + 2 * El ** 2 * Ey) / (E ** 3 + eps)
    return Nlx ** 2 + Nly ** 2 + Nllx ** 2 + Nlly ** 2



def H_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Hx = (Ell * Elx - El * Ellx) / (El ** 2 + Ell ** 2 + eps)
    Hy = (Ell * Ely - El * Elly) / (El ** 2 + Ell ** 2 + eps)
    return Hx ** 2 + Hy ** 2


inv_switcher = {
    'E': E_inv,
    'W': W_inv,
    'C': C_inv,
    'N': N_inv,
    'H': H_inv,
}


class CIConv2d(nn.Module):
    def __init__(self, invariant, k=3, scale=0.9, max_scale=2.5):
        super().__init__()
        assert invariant in inv_switcher, 'invalid invariant'
        self.inv_function = inv_switcher[invariant]
        self.k = k
        self.max_scale = float(max_scale)
        self.scale = nn.Parameter(torch.tensor([float(scale)], dtype=torch.float32), requires_grad=True)
        self.register_buffer(
            'gcm',
            torch.tensor(
                [[0.06, 0.63, 0.27], [0.30, 0.04, -0.35], [0.34, -0.60, 0.17]],
                dtype=torch.float32,
            ),
        )

    def bounded_scale(self):
        return self.max_scale * torch.tanh(self.scale / self.max_scale)

    def forward(self, batch):
        if batch.shape[1] != 3:
            raise ValueError(f'CIConv2d expects 3-channel RGB input, but got {batch.shape[1]} channels.')

        in_shape = batch.shape
        flat = batch.view(in_shape[0], in_shape[1], -1)
        gcm = self.gcm.to(device=batch.device, dtype=batch.dtype).unsqueeze(0)
        converted = torch.matmul(gcm, flat)
        converted = converted.view(in_shape[0], 3, in_shape[2], in_shape[3])
        E, El, Ell = torch.split(converted, 1, dim=1)

        w = gaussian_basis_filters(
            scale=self.bounded_scale().squeeze(0),
            device=batch.device,
            dtype=batch.dtype,
            k=self.k,
        )
        padding = w.shape[-1] // 2

        E_out = F.conv2d(E, w, padding=padding)
        El_out = F.conv2d(El, w, padding=padding)
        Ell_out = F.conv2d(Ell, w, padding=padding)

        E, Ex, Ey = torch.split(E_out, 1, dim=1)
        El, Elx, Ely = torch.split(El_out, 1, dim=1)
        Ell, Ellx, Elly = torch.split(Ell_out, 1, dim=1)

        inv_out = self.inv_function(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly)
        inv_out = F.instance_norm(torch.log(inv_out + eps))
        return inv_out


class MultiInvariantPrior(nn.Module):
    def __init__(self, invariants: Sequence[str] = ('W', 'H', 'C'), k=3, scale=0.9, clamp_output=True):
        super().__init__()
        if isinstance(invariants, str):
            invariants = [invariants]
        invariants = list(invariants)
        if len(invariants) == 0:
            raise ValueError('prior invariants must not be empty.')

        self.invariant_names = invariants
        self.extractors = nn.ModuleList([CIConv2d(inv, k=k, scale=scale) for inv in invariants])
        self.clamp_output = clamp_output

    @property
    def out_channels(self):
        return len(self.extractors)

    def forward(self, x):
        prior = torch.cat([extractor(x) for extractor in self.extractors], dim=1)
        if self.clamp_output:
            prior = torch.clamp(prior, 0.0, 1.0)
        return prior
