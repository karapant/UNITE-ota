"""
Standalone transmitter (encoder) and receiver (decoder) halves of a trained UNITE model.

Training is unchanged; these modules are built from the same gen_tok config and reuse
UNITE's own build/encode/decode code, so they load weights from a UNITE checkpoint
1:1 and behave exactly like UNITE.reconstruct_batch when chained.

    PC 1 (TX):  img [B,3,256,256] in [0,1] -> UNITEEncoder -> z [B,256,32]
    PC 2 (RX):  z [B,256,32] -> UNITEDecoder -> img [B,3,256,256] in [0,1]

Neither half needs LPIPS / transport / the other half's weights.
"""

import json
import resource
import sys
import time
from contextlib import contextmanager

import torch
import torch.nn as nn

from models.unite import UNITE

TX_PREFIXES = ("patch_embed.", "encoder.", "latent_tokens", "encoder_layer_norm.")
RX_PREFIXES = ("up_sample_decoder.", "decoder.")
IGNORED_PREFIXES = ("lpips_loss.",)


class UNITEEncoder(nn.Module):
    """Transmitter: image -> layer-normed latent tokens z (the decoder's input)."""

    _build_encoder = UNITE._build_encoder
    _encode_raw = UNITE.encode

    def __init__(self, cfg, num_latent_tokens=256, num_classes=1000):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_latent_tokens = num_latent_tokens
        self._build_encoder()

    @torch.no_grad()
    def forward(self, imgs):
        z = self._encode_raw(imgs)
        return self.encoder_layer_norm(z)


class UNITEDecoder(nn.Module):
    """Receiver: latent tokens z -> image in [0, 1]."""

    _build_decoder = UNITE._build_decoder

    def __init__(self, cfg, num_latent_tokens=256, num_classes=1000):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = int(cfg.get("patch_size", 16))
        self.diffusion_input_dim = int(cfg.get("diffusion_input_dim", 32))
        self.num_image_patches = (256 // self.patch_size) ** 2
        self._build_decoder()

    @torch.no_grad()
    def forward(self, z):
        z = self.up_sample_decoder(z)
        logits = self.decoder(z, drop_cls_token=False)
        return self.decoder.unpatchify(logits).clamp(0, 1)


# ---------------------------------------------------------------------------
# Splitting a UNITE checkpoint / loading the exported halves
# ---------------------------------------------------------------------------

def _clean_key(k):
    # torch.compile(model.decoder) stores keys as "decoder._orig_mod.*"
    return k.replace("_orig_mod.", "")


def split_state_dict(state_dict):
    """Split a full UNITE state dict into (tx_state, rx_state). Fails on unknown keys."""
    tx, rx = {}, {}
    for k, v in state_dict.items():
        k = _clean_key(k)
        if k.startswith(TX_PREFIXES):
            tx[k] = v
        elif k.startswith(RX_PREFIXES):
            rx[k] = v
        elif not k.startswith(IGNORED_PREFIXES):
            raise KeyError(f"Key '{k}' belongs to neither encoder nor decoder")
    return tx, rx


def save_split(path, module, state_dict):
    torch.save({
        "cfg": module.cfg,
        "num_latent_tokens": module.num_latent_tokens,
        "num_classes": module.num_classes,
        "state_dict": state_dict,
    }, path)


def default_device():
    """CUDA if available, then Apple-silicon MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def peak_rss_mb():
    """Peak resident memory of this process so far (ru_maxrss is bytes on macOS, KiB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


class Profiler:
    """Wall-clock timing of TX/RX stages and memory footprint, reported at the end of a run."""

    def __init__(self, role, device):
        self.role, self.device = role, torch.device(device)
        self.times = {}  # stage -> [(seconds, items), ...]
        self.info = {}

    def _sync(self):  # GPU work is asynchronous: wait for it so timings are real
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @contextmanager
    def time(self, stage, items=1):
        """Time a stage; `items` can also be set on the yielded object once known."""
        run = type("Run", (), {"items": items})()
        self._sync()
        t0 = time.perf_counter()
        yield run
        self._sync()
        self.times.setdefault(stage, []).append((time.perf_counter() - t0, run.items))

    def report(self, model=None, **extra):
        stats = {"role": self.role, "device": str(self.device), **self.info, **extra}
        if model is not None:
            stats["model_params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
            stats["model_memory_mb"] = round(sum(t.numel() * t.element_size()
                                                 for t in [*model.parameters(), *model.buffers()]) / 2**20, 1)
        for stage, runs in self.times.items():
            secs, items = sum(r[0] for r in runs), sum(r[1] for r in runs)
            stats[f"{stage}_total_s"] = round(secs, 3)
            stats[f"{stage}_ms_per_item"] = round(1000 * secs / max(items, 1), 2)
            if len(runs) > 1:  # first batch includes one-time warm-up; report steady state separately
                rest_s, rest_n = sum(r[0] for r in runs[1:]), sum(r[1] for r in runs[1:])
                stats[f"{stage}_first_batch_ms_per_item"] = round(1000 * runs[0][0] / runs[0][1], 2)
                stats[f"{stage}_steady_ms_per_item"] = round(1000 * rest_s / max(rest_n, 1), 2)
        stats["peak_rss_mb"] = round(peak_rss_mb(), 1)
        if self.device.type == "cuda":
            stats["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated(self.device) / 2**20, 1)
        elif self.device.type == "mps":  # no peak counter on MPS: memory held by the Metal driver at the end
            stats["gpu_memory_mb"] = round(torch.mps.driver_allocated_memory() / 2**20, 1)
        tag = self.role.upper()
        for k, v in stats.items():
            print(f"[{tag}] {k:<32} {v}")
        print(f"[{tag}-PROFILE] {json.dumps(stats)}")
        return stats


def load_split(path, cls, device="cpu"):
    """Load an exported encoder.pt (cls=UNITEEncoder) or decoder.pt (cls=UNITEDecoder)."""
    ckpt = torch.load(path, map_location="cpu")
    model = cls(ckpt["cfg"], num_latent_tokens=ckpt["num_latent_tokens"], num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval()
