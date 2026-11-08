"""Phase congruency from a log-Gabor filter bank (Kovesi 1999, 2003).

Phase congruency marks features where the Fourier components of the image are
maximally in phase. Because it is a ratio of local energy to total amplitude, it
responds to the *shape* of a feature rather than to its contrast. It does not undo
shadows: a cast shadow's edge is a real feature too. It removes the dependence on
how bright the structure happens to be.

This is the classic even/odd filter-bank formulation with Kovesi's frequency spread
weighting; a noise threshold estimated from the smallest-scale amplitude replaces his
Rayleigh model. Whole rasters are processed one orientation at a time, because holding
every filter response of a 15-megapixel reference would need about 3 GB.

Two more details matter on real images. The image is mirrored at its border before the FFT,
because the FFT would otherwise see a false edge where opposite borders meet. And the
ratio gets an amplitude floor, because far from any feature both energy and amplitude
shrink towards zero while their ratio stays near one, marking flat ground as a feature.
"""
from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np
from scipy import fft as sfft

_MIN_WAVELENGTH = 4.0
_SCALE_FACTOR = 2.1
_SIGMA_ON_F = 0.55
_ANGULAR_SIGMA = np.pi / 6.0
_PAD_PX = 48  # about the longest filter wavelength in use
_AMPLITUDE_FLOOR = 0.01  # share of the strong-feature amplitude below which there is no signal
_SPREAD_CUTOFF = 0.5  # Kovesi's frequency-spread weighting
_SPREAD_GAIN = 10.0


def _pad(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    return cv2.copyMakeBorder(data, _PAD_PX, _PAD_PX, _PAD_PX, _PAD_PX, cv2.BORDER_REFLECT_101)


def _unpad(array: np.ndarray) -> np.ndarray:
    return array[..., _PAD_PX:-_PAD_PX, _PAD_PX:-_PAD_PX]


def _orientation_responses(image: np.ndarray, scales: int, orientations: int) -> Iterator[np.ndarray]:
    """Yields, per orientation, complex responses of shape ``(scales, H, W)``.

    Real part = even-symmetric response, imaginary part = odd-symmetric response.
    """
    data = _pad(image)
    h, w = data.shape
    spectrum = sfft.fft2(data - data.mean(), workers=-1)

    fy = np.fft.fftfreq(h).astype(np.float32)[:, None]
    fx = np.fft.fftfreq(w).astype(np.float32)[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    radius[0, 0] = 1.0  # avoids log(0); the DC term is zeroed below
    theta = np.arctan2(-fy, fx)
    # A low-pass keeps wrap-around at the highest frequencies out of the bank.
    lowpass = 1.0 / (1.0 + (radius / 0.45) ** 30)
    log_radius = np.log(radius)
    denominator = 2.0 * float(np.log(_SIGMA_ON_F)) ** 2

    radial = []
    for s in range(scales):
        centre = float(np.log(1.0 / (_MIN_WAVELENGTH * _SCALE_FACTOR**s)))
        bank = np.exp(-((log_radius - centre) ** 2) / denominator) * lowpass
        bank[0, 0] = 0.0
        radial.append(bank.astype(np.float32))

    for o in range(orientations):
        angle = o * np.pi / orientations
        delta = np.arctan2(np.sin(theta - angle), np.cos(theta - angle))
        # One-sided angular window: the inverse FFT then yields an analytic (even + i*odd) signal.
        angular = np.exp(-(delta**2) / (2.0 * _ANGULAR_SIGMA**2)).astype(np.float32)
        out = np.zeros((scales, image.shape[0], image.shape[1]), np.complex64)
        for s in range(scales):
            out[s] = _unpad(sfft.ifft2(spectrum * radial[s] * angular, workers=-1))
        yield out


def log_gabor_responses(image: np.ndarray, scales: int = 4, orientations: int = 6) -> np.ndarray:
    """All responses at once, shape ``(orientations, scales, H, W)``. Meant for tiles, not whole rasters."""
    return np.stack(list(_orientation_responses(image, scales, orientations)))


def _accumulate(responses: np.ndarray, noise_k: float) -> tuple[np.ndarray, np.ndarray]:
    """(weighted energy above the noise threshold, total amplitude) of one orientation."""
    amplitude = np.abs(responses)
    total = amplitude.sum(axis=0)
    energy = np.sqrt(responses.real.sum(axis=0) ** 2 + responses.imag.sum(axis=0) ** 2)
    # Noise: the smallest-scale amplitude is dominated by it on natural images.
    tau = float(np.median(amplitude[0])) / np.sqrt(np.log(4.0))
    threshold = tau * noise_k * responses.shape[0] * 0.5
    # Frequency spread: if only one scale responds, energy trivially equals amplitude and the ratio
    # means nothing. The tail of the coarsest filter would otherwise mark flat ground near any edge.
    spread = (total / (amplitude.max(axis=0) + 1e-6) - 1.0) / max(responses.shape[0] - 1, 1)
    weight = 1.0 / (1.0 + np.exp((_SPREAD_CUTOFF - spread) * _SPREAD_GAIN))
    return weight * np.maximum(energy - threshold, 0.0), total


def _ratio(numerator: np.ndarray, denominator: np.ndarray, epsilon: float) -> np.ndarray:
    floor = _AMPLITUDE_FLOOR * float(np.percentile(denominator, 99.0))
    return np.clip(numerator / (denominator + floor + epsilon), 0.0, 1.0).astype(np.float32)


def phase_congruency_from_responses(responses: np.ndarray, noise_k: float = 2.0, epsilon: float = 1e-3) -> np.ndarray:
    """Phase congruency in [0, 1] from ``log_gabor_responses`` output (float32, H x W)."""
    numerator = np.zeros(responses.shape[2:], np.float32)
    denominator = np.zeros(responses.shape[2:], np.float32)
    for orientation in responses:
        energy, amplitude = _accumulate(orientation, noise_k)
        numerator += energy
        denominator += amplitude
    return _ratio(numerator, denominator, epsilon)


def phase_congruency(
    image: np.ndarray, scales: int = 4, orientations: int = 6, noise_k: float = 2.0, epsilon: float = 1e-3
) -> np.ndarray:
    """Phase congruency in [0, 1] of a whole image, streaming one orientation at a time."""
    numerator = np.zeros(image.shape, np.float32)
    denominator = np.zeros(image.shape, np.float32)
    for responses in _orientation_responses(image, scales, orientations):
        energy, amplitude = _accumulate(responses, noise_k)
        numerator += energy
        denominator += amplitude
    return _ratio(numerator, denominator, epsilon)
