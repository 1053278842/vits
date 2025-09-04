import math 
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

def _reflect_pad_1d_tensor(y: torch.Tensor, pad_amount: int) -> torch.Tensor:
    """
    Reflect-pad along the last dimension in a robust way.
    Supports:
      - 1D tensor: shape (T,)
      - 2D tensor: shape (B, T)
      - ND tensor: pads last dimension for arbitrary leading dims
    """
    if pad_amount <= 0:
        return y
    if y.dim() == 1:
        # 1D signal
        return F.pad(y, (pad_amount, pad_amount), mode='reflect')
    elif y.dim() == 2:
        # (batch, time)
        # F.pad supports 3D with (B, C, T) so unsqueeze channel dim
        return F.pad(y.unsqueeze(1), (pad_amount, pad_amount), mode='reflect').squeeze(1)
    else:
        # For ND (e.g., (B, C, T) or more), reshape to (-1, T) and pad
        orig_shape = y.shape
        last_dim = orig_shape[-1]
        leading = int(np.prod(orig_shape[:-1]))
        y_reshaped = y.contiguous().view(leading, last_dim)
        y_padded = F.pad(y_reshaped, (pad_amount, pad_amount), mode='reflect')
        new_last = y_padded.shape[-1]
        return y_padded.view(*orig_shape[:-1], new_last)


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        # keep hann window on same dtype & device as input
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    # pad_amount for STFT center padding (original code used (n_fft-hop_size)/2)
    pad_amount = int((n_fft - hop_size) / 2)

    # Robust reflect-pad across possible input shapes
    # Expect y is 1D waveform (T,) or 2D (batch, T). Handle both.
    y = _reflect_pad_1d_tensor(y, pad_amount)

    # Compute STFT, return_complex=True for modern PyTorch
    spec_c = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )

    # spec_c shape:
    #  - if input was 1D: (freq, time) complex
    #  - if input was (B, T): (B, freq, time) complex
    # We want magnitude (sqrt(real^2 + imag^2)), so use .abs()
    spec = spec_c.abs()

    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    pad_amount = int((n_fft - hop_size) / 2)
    y = _reflect_pad_1d_tensor(y, pad_amount)

    spec_c = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )

    spec = spec_c.abs()

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
