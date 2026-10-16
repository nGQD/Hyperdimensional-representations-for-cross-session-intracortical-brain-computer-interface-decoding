"""Causal neural-population decoders with explicit HDC and control front ends.

The HDC front end binds a channel's bounded continuous rate to a channel role
and a value-level code, then superposes the resulting complex basis vectors.
The implementation uses a single basis-weight GEMM; it never constructs a
batch-by-time-by-channel-by-hyperdimension tensor.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    family: str = "hdc"
    channels: int = 176
    outputs: int = 7
    hidden: int = 128
    dimension: int = 256  # complex coordinates; real/imag width is 2*dimension
    temporal_bins: int = 8
    levels: int = 8
    codebook_seed: int = 90210
    recurrence: bool = True
    online_norm: bool = True
    smoothing: float = 0.9
    dropout: float = 0.2
    epochs: int = 50
    lr: float = 0.0002
    weight_decay: float = 0.0001
    window: int = 256
    stride: int = 64
    batch: int = 256
    amp: bool = False
    no_roles: bool = False
    no_channel_roles: bool = False


FAMILIES = ("hdc", "random", "mlp", "plain")


def _generator(seed: int, device="cpu"):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def _complex_mul(real, imag, role_real, role_imag):
    """Multiply complex coordinates represented by paired real tensors."""
    return real * role_real - imag * role_imag, real * role_imag + imag * role_real


class PopulationEncoder(nn.Module):
    """Map each population bin to a real/imag hypervector of width 2D."""
    def __init__(self, config: Config):
        super().__init__()
        c = config
        if c.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {c.family!r}")
        if c.channels < 1 or c.dimension < 1 or c.levels < 2:
            raise ValueError("channels/dimension must be positive and levels >= 2")
        self.config = c
        n, levels, d = c.channels, c.levels, c.dimension
        self.basis_size = n * levels
        self.width = 2 * d

        # Shared anchor positions. Each normalized input rate contributes to
        # at most two neighboring anchors, using piecewise-linear interpolation.
        self.register_buffer("anchors", torch.linspace(0.0, 1.0, levels))

        g = _generator(c.codebook_seed)
        if c.family == "hdc":
            # Correlated anchor phases produce a smooth value kernel; linear
            # interpolation is a finite codebook approximation to it.
            frequency = torch.randn(d, generator=g) * (2.0 * math.pi)
            value_phase = self.anchors[:, None] * frequency[None, :]
            value_real, value_imag = value_phase.cos(), value_phase.sin()
            channel_phase = torch.rand(n, d, generator=g) * (2.0 * math.pi)
            if c.no_channel_roles:
                channel_phase.zero_()
            channel_real, channel_imag = channel_phase.cos(), channel_phase.sin()
            # C x L x D complex role/value bindings, flattened before runtime.
            bound_real = (channel_real[:, None, :] * value_real[None, :, :]
                          - channel_imag[:, None, :] * value_imag[None, :, :])
            bound_imag = (channel_real[:, None, :] * value_imag[None, :, :]
                          + channel_imag[:, None, :] * value_real[None, :, :])
            basis = torch.cat([bound_real, bound_imag], dim=-1).reshape(n * levels, 2 * d)
            # Each channel contributes a unit phasor; scale the superposition
            # so each real/imag coordinate has variance near one half.
            basis = basis / math.sqrt(n)
            self.register_buffer("basis_matrix", basis.contiguous())
        elif c.family == "random":
            # Independent mixing control over exactly the same interpolated
            # channel x value-level basis. Match HDC coordinate scale.
            random_g = _generator(c.codebook_seed + 1_000_003)
            basis = torch.randn(n * levels, 2 * d, generator=random_g) / math.sqrt(2 * n)
            self.register_buffer("basis_matrix", basis.contiguous())
        elif c.family == "mlp":
            # MLP is constructed in Decoder over the same K-bin raw support.
            pass
        else:  # plain: conventional learned linear projection over K-bin input.
            pass

    def basis_weights(self, x):
        """Piecewise-linear interpolation weights, shape [B,T,C,L]."""
        # Bounded continuous rates preserve ordering while preventing extreme
        # z-scores from dominating the codebook.
        v = (torch.tanh(x.float()) + 1.0) * 0.5
        scaled = v * (self.config.levels - 1)
        anchors = torch.arange(self.config.levels, device=x.device, dtype=scaled.dtype)
        # Dense triangular basis is exactly piecewise-linear interpolation on
        # this integer-spaced grid and avoids one-hot integer temporaries.
        return F.relu(1.0 - (scaled[..., None] - anchors).abs())

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.config.channels:
            raise ValueError(f"expected [batch,time,{self.config.channels}], got {tuple(x.shape)}")
        b, t, _ = x.shape
        weights = self.basis_weights(x).reshape(b * t, self.basis_size)
        if self.config.family in ("plain", "mlp"):
            raise RuntimeError("raw-lag families are encoded by Decoder's causal input stack")
        # GEMM over flattened channel x level basis: O(B*T*C*L + B*T*C*L*D)
        # arithmetic, with only the small basis weights materialized.
        return (weights @ self.basis_matrix).reshape(b, t, self.width)


class CausalRoleBundler(nn.Module):
    """Fixed complex temporal role-binding and causal superposition."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        d, k = config.dimension, config.temporal_bins
        g = _generator(config.codebook_seed + 104729)
        phase = torch.rand(k, d, generator=g) * (2.0 * math.pi)
        if config.no_roles:
            phase.zero_()
        self.register_buffer("role_real", phase.cos())
        self.register_buffer("role_imag", phase.sin())

    def forward(self, q, history=None):
        if q.ndim != 3 or q.shape[-1] != 2 * self.config.dimension:
            raise ValueError("bundler expects [batch,time,2*dimension]")
        b, t, _ = q.shape
        k, d = self.config.temporal_bins, self.config.dimension
        if history is None:
            history = q.new_zeros((b, k - 1, 2 * d))
        elif history.shape != (b, k - 1, 2 * d):
            raise ValueError("history must have shape [batch, temporal_bins-1, 2*dimension]")
        seq = torch.cat((history, q), dim=1)
        out_real = q.new_zeros((b, t, d))
        out_imag = q.new_zeros((b, t, d))
        for lag in range(k):
            part = seq[:, k - 1 - lag:k - 1 - lag + t]
            real, imag = part[..., :d], part[..., d:]
            br, bi = _complex_mul(real, imag, self.role_real[lag], self.role_imag[lag])
            out_real = out_real + br
            out_imag = out_imag + bi
        out = torch.cat((out_real, out_imag), dim=-1) / math.sqrt(k)
        next_history = seq[:, -(k - 1):] if k > 1 else seq[:, :0]
        return out, next_history


class CausalRawStack(nn.Module):
    """Newest-first raw neural lag stack shared by MLP and linear controls."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def forward(self, x, history=None):
        b, t, c = x.shape
        k = self.config.temporal_bins
        if history is None:
            history = x.new_zeros((b, k - 1, c))
        elif history.shape != (b, k - 1, c):
            raise ValueError("raw history must have shape [batch, temporal_bins-1, channels]")
        seq = torch.cat((history, x), dim=1)
        stacked = torch.cat(
            [seq[:, k - 1 - lag:k - 1 - lag + t] for lag in range(k)], dim=-1
        )
        next_history = seq[:, -(k - 1):] if k > 1 else seq[:, :0]
        return stacked, next_history


class Decoder(nn.Module):
    """Streaming decoder API: ``(prediction, hidden, history)``."""
    def __init__(self, config: Config = Config()):
        super().__init__()
        if config.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {config.family!r}")
        if config.temporal_bins < 1:
            raise ValueError("temporal_bins must be >= 1")
        if not 0.0 <= config.smoothing < 1.0:
            raise ValueError("smoothing must be in [0,1)")
        self.config = config
        self.encoder = (PopulationEncoder(config) if config.family in ("hdc", "random") else None)
        self.bundler = CausalRoleBundler(config)
        self.raw_stack = CausalRawStack(config)
        if config.family in ("hdc", "random"):
            self.input_projection = nn.Linear(2 * config.dimension, config.hidden)
        elif config.family == "mlp":
            width = max(1, round((2 * config.dimension * config.hidden) /
                                 (config.channels * config.temporal_bins + config.hidden + 1)))
            self.mlp_width = width
            self.input_projection = nn.Sequential(
                nn.Linear(config.channels * config.temporal_bins, width), nn.GELU(),
                nn.Linear(width, config.hidden)
            )
        else:
            self.input_projection = nn.Linear(config.channels * config.temporal_bins, config.hidden)
        self.input_norm = nn.LayerNorm(config.hidden)
        self.gru = (nn.GRU(config.hidden, config.hidden, batch_first=True)
                    if config.recurrence else None)
        self.nonrecurrent = (None if config.recurrence else
                             nn.Sequential(nn.Linear(config.hidden, config.hidden), nn.GELU()))
        self.state_norm = nn.LayerNorm(config.hidden)
        self.readout = nn.Sequential(
            nn.Linear(config.hidden, config.hidden), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden, config.outputs)
        )

    def forward(self, x, hidden=None, history=None):
        if self.config.family in ("hdc", "random"):
            q = self.encoder(x)
            contextual, history = self.bundler(q, history)
        else:
            contextual, history = self.raw_stack(x.float(), history)
        projected = self.input_norm(self.input_projection(contextual))
        if self.gru is not None:
            state, hidden = self.gru(projected, hidden)
        else:
            state, hidden = self.nonrecurrent(projected), None
        prediction = self.readout(self.state_norm(state))
        return prediction, hidden, history

    def parameter_count(self):
        """Trainable scalar count; fixed codebooks are buffers, not parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def parameter_counts(config: Config = Config()):
    """Report exact per-family trainable counts for the shared architecture."""
    return {family: Decoder(Config(**{**config.__dict__, "family": family})).parameter_count()
            for family in FAMILIES}
