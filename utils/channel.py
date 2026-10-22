"""
Statistical wireless channel for UNITE latents sent as complex baseband symbols (utils/latent_link.to_iq).

A scenario (configs/channels.yaml) draws, per transmission: an average SNR (path loss + log-normal
shadowing), Rician/Rayleigh small-scale fading with a Doppler-dependent time variation across the
symbols of one image (sum-of-sinusoids, Clarke/Jakes), AWGN, and an imperfect channel estimate at
the receiver, which then equalizes (MMSE or zero-forcing).
"""

import math

import numpy as np
import yaml

SPEED_OF_LIGHT = 299_792_458.0


def load_scenarios(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["link"], cfg["scenarios"]


def doppler_hz(speed_kmh, carrier_ghz):
    return speed_kmh / 3.6 * carrier_ghz * 1e9 / SPEED_OF_LIGHT


def _cn(rng, shape, var=1.0):
    """Circularly-symmetric complex Gaussian with variance `var` (scalar or broadcastable array)."""
    return np.sqrt(np.asarray(var) / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


class Channel:
    def __init__(self, scenario, link):
        self.p = scenario
        self.equalizer = link.get("equalizer", "mmse")
        self.sinusoids = int(link.get("sinusoids", 32))
        self.fd = doppler_hz(float(scenario.get("speed_kmh", 0)), float(link["carrier_ghz"]))
        self.symbol_period = 1.0 / (float(link["symbol_rate_msps"]) * 1e6)
        k_db = scenario.get("k_factor_db")
        self.k = 0.0 if k_db is None else 10 ** (k_db / 10)

    def _scattered(self, draws, symbols, rng):
        """Unit-power Rayleigh process per draw: constant without motion, sum of sinusoids otherwise."""
        if self.fd <= 0:
            return np.broadcast_to(_cn(rng, (draws, 1)), (draws, symbols))
        t = np.arange(symbols) * self.symbol_period
        g = np.zeros((draws, symbols), dtype=np.complex128)
        angles = rng.uniform(0, 2 * np.pi, (draws, self.sinusoids))
        phases = rng.uniform(0, 2 * np.pi, (draws, self.sinusoids))
        for m in range(self.sinusoids):
            g += np.exp(1j * (2 * np.pi * self.fd * np.cos(angles[:, m:m + 1]) * t + phases[:, m:m + 1]))
        return g / math.sqrt(self.sinusoids)

    def fading(self, draws, symbols, rng):
        los = math.sqrt(self.k / (self.k + 1)) * np.exp(1j * rng.uniform(0, 2 * np.pi, (draws, 1)))
        return los + math.sqrt(1 / (self.k + 1)) * self._scattered(draws, symbols, rng)

    def __call__(self, x, rng):
        """x: complex [draws, symbols] with unit average power -> (equalized x_hat, per-draw info)."""
        draws, symbols = x.shape
        snr_db = rng.normal(self.p["snr_mean_db"], self.p.get("shadowing_std_db", 0.0), draws)
        noise_var = 10 ** (-snr_db / 10)
        h = self.fading(draws, symbols, rng)
        y = h * x + _cn(rng, (draws, symbols), noise_var[:, None])

        h_hat = h
        if self.p.get("csi_nmse_db") is not None:
            h_hat = h + _cn(rng, (draws, symbols), 10 ** (self.p["csi_nmse_db"] / 10))
        if self.equalizer == "zf":
            x_hat = y / np.where(np.abs(h_hat) < 1e-6, 1e-6, h_hat)
        else:  # MMSE: minimizes the error of each symbol, shrinks symbols in deep fades instead of amplifying noise
            x_hat = np.conj(h_hat) * y / (np.abs(h_hat) ** 2 + noise_var[:, None])

        gain = np.mean(np.abs(h) ** 2, axis=1)
        info = {
            "snr_db": snr_db,                                   # average SNR drawn for this transmission
            "fading_gain_db": 10 * np.log10(gain),              # mean |h|^2 over the image
            "effective_snr_db": snr_db + 10 * np.log10(gain),   # SNR after fading, before equalization
        }
        return x_hat.astype(np.complex64), info
