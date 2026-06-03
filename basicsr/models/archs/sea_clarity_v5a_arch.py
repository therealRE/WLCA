"""
SeaClarityNetV5A: V4 UIEB restoration network.

V4-A upgrades on SeaClarityNetV5A V3:
- Gated Conv-FFN after each scale aggregation block for feature purification.
- Stable learned decoder feature upsampling.
- Stronger LL-band large-kernel color branch.
- Enhanced spatial wavelet gate using restored/LQ LL difference.

V4-A+B additionally adds a local spatial color adapter after global RGB affine correction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ste_clamp(x, min_value=0.0, max_value=1.0):
    """Clamp values in forward while preserving identity gradients in training."""
    return x + (x.clamp(min_value, max_value) - x).detach()


class SeaClarityNetV5A_Final(nn.Module):
    def __init__(
        self,
        en_feature_num=48,
        en_inter_num=32,
        de_feature_num=64,
        de_inter_num=32,
        sam_number=2,
        training=True,
        use_online_prior=False,
        prior_invariants=None,
        prior_k=3,
        prior_scale=0.9,
        decoder_freq_levels=None,
        residual_init=0.1,
        residual_decay_start=0.0,
        residual_decay_end=1.0,
        residual_decay_min=1.0,
        refine_hidden=None,
        refine_blocks=None,
        gate_hidden=32,
        spatial_gate_bias=None,
        residual_strength_bias=3.0,
        ll_hidden=None,
        ll_blocks=None,
        ll_large_blocks=2,
        ffn_expansion=2,
        local_color_hidden=32,
        local_color_scale=0.05,
        wavelet_band_hidden=48,
        wavelet_band_blocks=3,
        wavelet_use_shared_head=False,
        band_gate_bias=0.0,
        **kwargs,
    ):
        super().__init__()

        if refine_hidden is not None:
            wavelet_band_hidden = refine_hidden
        if refine_blocks is not None:
            wavelet_band_blocks = refine_blocks
        if spatial_gate_bias is not None:
            band_gate_bias = spatial_gate_bias

        self.encoder = SpectrumEncoder(
            feature_num=en_feature_num,
            inter_num=en_inter_num,
            sam_number=sam_number,
            ffn_expansion=ffn_expansion,
        )
        self.decoder = LumaDecoder(
            en_num=en_feature_num,
            feature_num=de_feature_num,
            inter_num=de_inter_num,
            sam_number=sam_number,
            ffn_expansion=ffn_expansion,
        )

        self.haar = HaarWaveletTransform()
        self.color_adapter = ColorAffineAdapter(
            feat_channels=de_feature_num,
            hidden_channels=64,
            scale=0.1,
        )
        ######################
        self.ll_bilateral_mlp_adapter = LLBilateralMLPColorAdapter(
            feat_channels=64,
            hidden_dim=8,
            grid_depth=8,
            grid_down=4,
            mid_channels=64,
            param_delta_scale=0.10,
            local_residual_scale_init=0.50,
        )
        #########################
        # Each band head sees restored-band(3), original-LQ-band(3), and decoder features.
        band_in_channels = de_feature_num + 6
        detail_hidden = max(int(wavelet_band_hidden) // 2, 16)
        detail_blocks = max(int(wavelet_band_blocks) - 1, 1)
        ll_hidden = int(ll_hidden) if ll_hidden is not None else max(int(wavelet_band_hidden), 64)
        ll_blocks = int(ll_blocks) if ll_blocks is not None else max(int(wavelet_band_blocks) + 1, 4)

        self.band_head_ll = LowFrequencyColorHead(
            in_channels=band_in_channels,
            hidden_channels=ll_hidden,
            out_channels=3,
            num_blocks=ll_blocks,
            large_blocks=ll_large_blocks,
        )
        self.high_band_head = WaveletBandHead(
            in_channels=band_in_channels,
            hidden_channels=detail_hidden,
            out_channels=3,
            num_blocks=detail_blocks,
        )

        self.band_gate = EnhancedSpatialWaveletBandGateV2(
            in_channels=de_feature_num + 9,
            hidden_channels=gate_hidden,
            bias_init=band_gate_bias,
        )
        self.residual_strength_map = ResidualStrengthMap(
            in_channels=12,
            hidden_channels=gate_hidden,
            bias_init=residual_strength_bias,
        )

        residual_init = float(max(min(residual_init, 0.95), 0.01))
        self.residual_scale_logit = nn.Parameter(
            torch.tensor(math.log(residual_init / (1.0 - residual_init)))
        )

        self.residual_decay_start = float(max(min(residual_decay_start, 1.0), 0.0))
        self.residual_decay_end = float(max(min(residual_decay_end, 1.0), self.residual_decay_start))
        self.residual_decay_min = float(max(min(residual_decay_min, 1.0), 0.0))
        self.current_progress = 0.0

        self._initialize_weights()
        self.color_adapter.reset_identity()
        self.band_head_ll.reset_residual_head()
        self.high_band_head.reset_residual_head()
        self.band_gate.reset_gate(band_gate_bias)
        self.residual_strength_map.reset_strength(residual_strength_bias)

    def set_training_progress(self, progress):
        self.current_progress = float(max(min(progress, 1.0), 0.0))

    def _residual_decay_factor(self):
        if self.residual_decay_min >= 0.9999 or self.residual_decay_end <= self.residual_decay_start:
            return 1.0
        p = self.current_progress
        if p <= self.residual_decay_start:
            return 1.0
        if p >= self.residual_decay_end:
            return self.residual_decay_min
        ratio = (p - self.residual_decay_start) / (self.residual_decay_end - self.residual_decay_start)
        return 1.0 + ratio * (self.residual_decay_min - 1.0)

    @staticmethod
    def _crop_to_size(x, h, w):
        return x[:, :, :h, :w]

    def _band_input(self, restored_band, lq_band, feat_half):
        if feat_half.shape[-2:] != restored_band.shape[-2:]:
            feat_half = F.interpolate(feat_half, size=restored_band.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([restored_band, lq_band, feat_half], dim=1)


#################修改版
    def forward(self, x, s=None, guide_x=None, return_prior=False):
        _, _, H, W = x.shape

        rate = 2 ** 5
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate

        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        y_1, y_2, y_3 = self.encoder(x)
        base_full, base_half, base_quarter, fusion_feat_half = self.decoder(y_1, y_2, y_3)

        color_base = self.color_adapter(base_full, x, fusion_feat_half)

        base_ll_raw, base_lh, base_hl, base_hh = self.haar.dwt(color_base)
        lq_ll, lq_lh, lq_hl, lq_hh = self.haar.dwt(x)

        # ---------------------------------------------------------
        # New module: LL-domain bilateral-grid pixel-adaptive MLP.
        # It corrects low-frequency color/illumination residuals.
        # ---------------------------------------------------------
        base_ll_adapted = self.ll_bilateral_mlp_adapter(
            base_ll_raw,
            lq_ll,
            fusion_feat_half
        )
        delta_ll_bmlp = base_ll_adapted - base_ll_raw
        base_ll = base_ll_adapted

        gate_ll, gate_lh, gate_hl, gate_hh = self.band_gate(
            fusion_feat_half,
            base_ll,
            lq_ll
        )

        delta_ll = self.band_head_ll(
            self._band_input(base_ll, lq_ll, fusion_feat_half)
        )
        delta_ll = delta_ll + delta_ll_bmlp

        delta_lh = self.high_band_head(
            self._band_input(base_lh, lq_lh, fusion_feat_half)
        )
        delta_hl = self.high_band_head(
            self._band_input(base_hl, lq_hl, fusion_feat_half)
        )
        delta_hh = self.high_band_head(
            self._band_input(base_hh, lq_hh, fusion_feat_half)
        )

        wavelet_residual = self.haar.idwt(
            gate_ll * delta_ll,
            gate_lh * delta_lh,
            gate_hl * delta_hl,
            gate_hh * delta_hh,
        )

        residual = torch.tanh(wavelet_residual)
        residual_scale = torch.sigmoid(self.residual_scale_logit) * self._residual_decay_factor()
        residual_strength = self.residual_strength_map(color_base, x, base_ll, lq_ll)
        raw_full = color_base + residual_scale * residual_strength * residual

        if self.training:
            out_full = ste_clamp(raw_full, 0.0, 1.0)
            out_half_raw = ste_clamp(base_half, 0.0, 1.0)
            out_quarter_raw = ste_clamp(base_quarter, 0.0, 1.0)
        else:
            out_full = torch.clamp(raw_full, 0.0, 1.0)
            out_half_raw = torch.clamp(base_half, 0.0, 1.0)
            out_quarter_raw = torch.clamp(base_quarter, 0.0, 1.0)

        out_full = self._crop_to_size(out_full, H, W)
        out_half = self._crop_to_size(out_half_raw, max(H // 2, 1), max(W // 2, 1))
        out_quarter = self._crop_to_size(out_quarter_raw, max(H // 4, 1), max(W // 4, 1))

        if return_prior:
            return out_full, out_half, out_quarter, None
        return out_full, out_half, out_quarter

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class HaarWaveletTransform(nn.Module):
    """Fixed 2D Haar DWT / IDWT implemented with tensor slicing."""
    def dwt(self, x):
        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 0::2, 1::2]
        c = x[:, :, 1::2, 0::2]
        d = x[:, :, 1::2, 1::2]

        ll = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5
        return ll, lh, hl, hh

    def idwt(self, ll, lh, hl, hh):
        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5

        bsz, ch, h, w = ll.shape
        out = torch.zeros(bsz, ch, h * 2, w * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = a
        out[:, :, 0::2, 1::2] = b
        out[:, :, 1::2, 0::2] = c
        out[:, :, 1::2, 1::2] = d
        return out


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class WaveletBandHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=48, out_channels=3, num_blocks=3):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.blocks = nn.Sequential(*[ResidualConvBlock(hidden_channels) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def reset_residual_head(self):
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.act(self.in_conv(x))
        x = self.blocks(x)
        x = self.out_conv(x)
        return x


class GatedConvFFN(nn.Module):
    """Zero-initialized gated depthwise FFN for feature purification after aggregation."""
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=True
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.project_out.weight)
        if self.project_out.bias is not None:
            nn.init.zeros_(self.project_out.bias)

    def forward(self, x):
        y = self.project_in(x)
        y = self.dwconv(y)
        y1, y2 = torch.chunk(y, 2, dim=1)
        y = F.gelu(y1) * y2
        y = self.project_out(y)
        return x + y


class LearnedFeatureUpsample(nn.Module):
    """Stable learned upsampling: bilinear base plus zero-initialized learned correction."""
    def __init__(self, channels):
        super().__init__()
        self.expand = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.refine = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.reset_identity()

    def reset_identity(self):
        # Keep the first training iterations close to the previous bilinear feature path.
        last = self.refine[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        residual = self.refine(self.expand(x))
        return base + residual


class MultiKernelColorBlock(nn.Module):
    """Multi-scale low-frequency color block with 3/7/11 depthwise kernels.

    The final pointwise projection is zero-initialized so the block starts as an
    identity mapping and learns only useful low-frequency corrections.
    """
    def __init__(self, channels):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True)
        self.dw7 = nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels, bias=True)
        self.dw11 = nn.Conv2d(channels, channels, kernel_size=11, stride=1, padding=5, groups=channels, bias=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.reset_identity()

    def reset_identity(self):
        last = self.fuse[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, x):
        y3 = self.dw3(x)
        y7 = self.dw7(x)
        y11 = self.dw11(x)
        y = self.fuse(torch.cat([y3, y7, y11], dim=1))
        return x + y



class ConvGELU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)

def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))

class LLBilateralMLPColorAdapter(nn.Module):

    def __init__(
        self,
        feat_channels=64,
        hidden_dim=8,
        grid_depth=8,
        grid_down=4,
        mid_channels=64,
        param_delta_scale=0.10,
        local_residual_scale_init=0.50,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.grid_depth = grid_depth
        self.grid_down = grid_down
        self.param_delta_scale = param_delta_scale

        # base_ll, lq_ll, |base_ll-lq_ll|, fusion_feat_half
        cond_channels = 3 + 3 + 3 + feat_channels

        self.cond_proj = nn.Sequential(
            ConvGELU(cond_channels, mid_channels, 3),
            ConvGELU(mid_channels, mid_channels, 3),
        )

        self.grid_refine = nn.Sequential(
            ConvGELU(mid_channels, mid_channels, 3),
            ConvGELU(mid_channels, mid_channels, 3),
        )

        # First MLP layer: W1: hidden_dim x 3, b1: hidden_dim
        # params = hidden_dim*3 + hidden_dim
        self.p1_dim = hidden_dim * 3 + hidden_dim

        # Second MLP layer: W2: 3 x hidden_dim, b2: 3
        # params = 3*hidden_dim + 3
        self.p2_dim = 3 * hidden_dim + 3

        self.grid1_head = nn.Conv2d(
            mid_channels,
            self.p1_dim * grid_depth,
            kernel_size=1
        )
        self.grid2_head = nn.Conv2d(
            mid_channels,
            self.p2_dim * grid_depth,
            kernel_size=1
        )

        # Two guidance maps, one for each grid.
        self.guide_head = nn.Sequential(
            ConvGELU(mid_channels, mid_channels // 2, 3),
            nn.Conv2d(mid_channels // 2, 2, kernel_size=1),
            nn.Sigmoid()
        )

        # A local spatial gate for conservative LL residual generation.
        self.inject_gate = nn.Sequential(
            ConvGELU(mid_channels, mid_channels // 2, 3),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.local_residual_scale_logit = nn.Parameter(
            torch.tensor(_logit(local_residual_scale_init), dtype=torch.float32)
        )

        self._init_identity_like_mlp()

    def _init_identity_like_mlp(self):

        nn.init.zeros_(self.grid1_head.weight)
        nn.init.zeros_(self.grid1_head.bias)
        nn.init.zeros_(self.grid2_head.weight)
        nn.init.zeros_(self.grid2_head.bias)

        base_p1 = torch.zeros(1, self.p1_dim, 1, 1)

        # Build W1 independently to avoid memory overlap.
        w1 = torch.zeros(1, self.hidden_dim, 3, 1, 1)

        # Copy RGB/LL channels into the first three hidden neurons.
        for c in range(min(3, self.hidden_dim)):
            w1[:, c, c, :, :] = 1.0

        base_p1[:, :self.hidden_dim * 3, :, :] = w1.reshape(
            1, self.hidden_dim * 3, 1, 1
        ).clone()

        # b1 remains zero.
        self.register_buffer("base_p1", base_p1)

        # Second-layer base parameters: W2 and b2
        # Initialize to zero so that the MLP branch initially outputs zero residual.
        base_p2 = torch.zeros(1, self.p2_dim, 1, 1)
        self.register_buffer("base_p2", base_p2)

    def _slice_grid(self, grid, guide):

        B, P, D, Hg, Wg = grid.shape
        _, _, H, W = guide.shape

        dtype = grid.dtype
        device = grid.device

        y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        xx = xx.unsqueeze(0).expand(B, H, W)
        yy = yy.unsqueeze(0).expand(B, H, W)

        zz = guide[:, 0].clamp(0.0, 1.0) * 2.0 - 1.0

        # grid_sample for 5D input expects coordinates in the order x, y, z.
        sample_grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(1)
        # [B, 1, H, W, 3]

        sliced = F.grid_sample(
            grid,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        # [B, P, 1, H, W]

        return sliced.squeeze(2)

    def _apply_pixel_adaptive_mlp(self, x_ll, p1, p2):

        B, _, H, W = x_ll.shape

        w1 = p1[:, :self.hidden_dim * 3].view(B, self.hidden_dim, 3, H, W)
        b1 = p1[:, self.hidden_dim * 3:].view(B, self.hidden_dim, H, W)

        hidden = (w1 * x_ll.unsqueeze(1)).sum(dim=2) + b1
        hidden = F.relu(hidden, inplace=False)

        w2 = p2[:, :3 * self.hidden_dim].view(B, 3, self.hidden_dim, H, W)
        b2 = p2[:, 3 * self.hidden_dim:].view(B, 3, H, W)

        residual_ll = (w2 * hidden.unsqueeze(1)).sum(dim=2) + b2
        return residual_ll

    def forward(self, base_ll, lq_ll, fusion_feat_half):

        if fusion_feat_half.shape[-2:] != base_ll.shape[-2:]:
            fusion_feat_half = F.interpolate(
                fusion_feat_half,
                size=base_ll.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        diff_ll = torch.abs(base_ll - lq_ll)
        cond = torch.cat([base_ll, lq_ll, diff_ll, fusion_feat_half], dim=1)

        cond_feat = self.cond_proj(cond)

        B, _, H, W = base_ll.shape
        Hg = max(1, H // self.grid_down)
        Wg = max(1, W // self.grid_down)

        grid_feat = F.interpolate(
            cond_feat,
            size=(Hg, Wg),
            mode="bilinear",
            align_corners=False
        )
        grid_feat = self.grid_refine(grid_feat)

        grid1 = self.grid1_head(grid_feat)
        grid2 = self.grid2_head(grid_feat)

        grid1 = grid1.view(B, self.p1_dim, self.grid_depth, Hg, Wg)
        grid2 = grid2.view(B, self.p2_dim, self.grid_depth, Hg, Wg)

        guide = self.guide_head(cond_feat)
        guide1 = guide[:, 0:1]
        guide2 = guide[:, 1:2]

        p1_delta = self._slice_grid(grid1, guide1)
        p2_delta = self._slice_grid(grid2, guide2)

        p1 = self.base_p1 + self.param_delta_scale * torch.tanh(p1_delta)
        p2 = self.base_p2 + self.param_delta_scale * torch.tanh(p2_delta)

        residual_ll = self._apply_pixel_adaptive_mlp(base_ll, p1, p2)

        local_gate = self.inject_gate(cond_feat)
        local_scale = torch.sigmoid(self.local_residual_scale_logit)

        delta_ll = local_scale * local_gate * torch.tanh(residual_ll)

        corrected_ll = base_ll + delta_ll
        return corrected_ll

class LowFrequencyColorHead(nn.Module):
    """Stronger LL-band branch for color, brightness, haze, and illumination correction."""
    def __init__(self, in_channels, hidden_channels=64, out_channels=3, num_blocks=4, large_blocks=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualConvBlock(hidden_channels))
        for _ in range(large_blocks):
            blocks.append(MultiKernelColorBlock(hidden_channels))
        self.blocks = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def reset_residual_head(self):
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.act(self.in_conv(x))
        x = self.blocks(x)
        return self.out_conv(x)

class EnhancedSpatialWaveletBandGateV2(nn.Module):
    """Spatial four-band gate with large-kernel context and SE channel attention.

    Inputs include decoder features, restored LL, LQ LL, and their absolute
    difference. The output still predicts four spatial gates for LL/LH/HL/HH.
    """
    def __init__(self, in_channels, hidden_channels=32, bias_init=0.0):
        super().__init__()
        squeeze_channels = max(hidden_channels // 4, 8)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, stride=1, padding=3, groups=hidden_channels, bias=True),
            nn.GELU(),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, squeeze_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(squeeze_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(hidden_channels, 4, kernel_size=1, stride=1, padding=0, bias=True)
        self.reset_gate(bias_init)

    def reset_gate(self, bias_init=0.0):
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, float(bias_init))

    def forward(self, feat_half, base_ll, lq_ll):
        if feat_half.shape[-2:] != base_ll.shape[-2:]:
            feat_half = F.interpolate(feat_half, size=base_ll.shape[-2:], mode='bilinear', align_corners=False)
        diff_ll = torch.abs(base_ll - lq_ll)
        feat = self.body(torch.cat([feat_half, base_ll, lq_ll, diff_ll], dim=1))
        feat = feat * self.se(feat)
        gate = torch.sigmoid(self.out(feat))
        return torch.chunk(gate, 4, dim=1)


class ResidualStrengthMap(nn.Module):
    """Spatial residual strength map for adaptive wavelet residual injection.

    The bias is initialized positive so the map starts close to one, preserving
    the V4-A behavior while allowing the model to suppress over-correction in
    easy or already-restored regions.
    """
    def __init__(self, in_channels=12, hidden_channels=32, bias_init=3.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, stride=1, padding=3, groups=hidden_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.reset_strength(bias_init)

    def reset_strength(self, bias_init=3.0):
        last = self.body[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, float(bias_init))

    def forward(self, color_base, lq, base_ll, lq_ll):
        diff_full = torch.abs(color_base - lq)
        diff_ll = torch.abs(base_ll - lq_ll)
        diff_ll = F.interpolate(diff_ll, size=color_base.shape[-2:], mode='bilinear', align_corners=False)
        strength = torch.sigmoid(self.body(torch.cat([color_base, lq, diff_full, diff_ll], dim=1)))
        return strength


class ColorAffineAdapter(nn.Module):
    """Image-level RGB affine adapter initialized as identity."""
    def __init__(self, feat_channels, hidden_channels=64, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(feat_channels + 6, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 12, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def reset_identity(self):
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, base, lq, feat_half):
        feat = F.interpolate(feat_half, size=base.shape[-2:], mode='bilinear', align_corners=False)
        cond = torch.cat([base, lq, feat], dim=1)
        params = self.mlp(self.pool(cond))

        matrix_delta = params[:, :9].view(-1, 3, 3, 1, 1)
        bias_delta = params[:, 9:].view(-1, 3, 1, 1)
        eye = torch.eye(3, device=base.device, dtype=base.dtype).view(1, 3, 3, 1, 1)

        matrix = eye + self.scale * torch.tanh(matrix_delta)
        bias = self.scale * torch.tanh(bias_delta)
        out = torch.sum(matrix * base.unsqueeze(1), dim=2) + bias
        return out


class SpatialWaveletBandGate(nn.Module):
    """Spatial four-band gate predicted from decoder feature and LL color cues."""
    def __init__(self, in_channels, hidden_channels=32, bias_init=0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.reset_gate(bias_init)

    def reset_gate(self, bias_init=0.0):
        last = self.gate[2]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, float(bias_init))

    def forward(self, feat_half, base_ll, lq_ll):
        if feat_half.shape[-2:] != base_ll.shape[-2:]:
            feat_half = F.interpolate(feat_half, size=base_ll.shape[-2:], mode='bilinear', align_corners=False)
        gate = self.gate(torch.cat([feat_half, base_ll, lq_ll], dim=1))
        return torch.chunk(gate, 4, dim=1)



class LumaDecoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number, ffn_expansion=2):
        super().__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = DecoderStage(feature_num, inter_num, sam_number, level=3, ffn_expansion=ffn_expansion)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = DecoderStage(feature_num, inter_num, sam_number, level=2, ffn_expansion=ffn_expansion)

        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = DecoderStage(feature_num, inter_num, sam_number, level=1, ffn_expansion=ffn_expansion)

    def forward(self, y_1, y_2, y_3):
        x_3 = self.preconv_3(y_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1, feat_1 = self.decoder_1(x_1, return_feat=True)

        return out_1, out_2, out_3, feat_1


class SpectrumEncoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number, ffn_expansion=2):
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.encoder_1 = EncoderStage(feature_num, inter_num, level=1, sam_number=sam_number, ffn_expansion=ffn_expansion)
        self.encoder_2 = EncoderStage(2 * feature_num, inter_num, level=2, sam_number=sam_number, ffn_expansion=ffn_expansion)
        self.encoder_3 = EncoderStage(4 * feature_num, inter_num, level=3, sam_number=sam_number, ffn_expansion=ffn_expansion)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)

        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return out_feature_1, out_feature_2, out_feature_3


class EncoderStage(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number, ffn_expansion=2):
        super().__init__()
        self.rdb = ResidualDenseBlock(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)

        self.sam_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = ScaleAggregationBlock(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
            self.ffn_blocks.append(GatedConvFFN(feature_num, expansion=ffn_expansion))

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )

        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)

        for sam_block, ffn_block in zip(self.sam_blocks, self.ffn_blocks):
            out_feature = sam_block(out_feature)
            out_feature = ffn_block(out_feature)

        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class DecoderStage(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number, level, ffn_expansion=2):
        super().__init__()
        self.rdb = ResidualDenseBlock(feature_num, (1, 2, 1), inter_num)

        self.sam_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = ScaleAggregationBlock(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
            self.ffn_blocks.append(GatedConvFFN(feature_num, expansion=ffn_expansion))

        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)
        self.feat_up = LearnedFeatureUpsample(feature_num)
        self.level = level

    def forward(self, x, return_feat=False):
        x = self.rdb(x)

        for sam_block, ffn_block in zip(self.sam_blocks, self.ffn_blocks):
            x = sam_block(x)
            x = ffn_block(x)

        out = self.conv(x)
        out = F.pixel_shuffle(out, 2)

        if return_feat:
            return out, x

        feature = self.feat_up(x)
        return out, feature


class DenseBlock(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super().__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(
                in_channel=c,
                out_channel=inter_num,
                kernel_size=3,
                dilation_rate=d_list[i],
                padding=d_list[i],
            )
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class ScaleAggregationBlock(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super().__init__()
        self.basic_block = DenseBlock(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DenseBlock(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DenseBlock(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = ChannelScaleFusion(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear', align_corners=False)
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear', align_corners=False)

        y = self.fusion(y_0, y_2, y_4)
        y = x + y
        return y


class ChannelScaleFusion(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = torch.sigmoid(out)

        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4
        return x


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super().__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(
                in_channel=c,
                out_channel=inter_num,
                kernel_size=3,
                dilation_rate=d_list[i],
                padding=d_list[i],
            )
            self.conv_layers.append(dense_conv)
            c = c + inter_num

        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            dilation=dilation_rate,
        )

    def forward(self, x_input):
        return self.conv(x_input)


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
                dilation=dilation_rate,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_input):
        return self.conv(x_input)
