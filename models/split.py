"""
Standalone transmitter (encoder) and receiver (decoder) halves of a trained UNITE model.

Training is unchanged; these modules are built from the same gen_tok config and reuse
UNITE's own build/encode/decode code, so they load weights from a UNITE checkpoint
1:1 and behave exactly like UNITE.reconstruct_batch when chained.

    PC 1 (TX):  img [B,3,256,256] in [0,1] -> UNITEEncoder -> z [B,256,32]
    PC 2 (RX):  z [B,256,32] -> UNITEDecoder -> img [B,3,256,256] in [0,1]

Neither half needs LPIPS / transport / the other half's weights.
"""

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


def load_split(path, cls, device="cpu"):
    """Load an exported encoder.pt (cls=UNITEEncoder) or decoder.pt (cls=UNITEDecoder)."""
    ckpt = torch.load(path, map_location="cpu")
    model = cls(ckpt["cfg"], num_latent_tokens=ckpt["num_latent_tokens"], num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval()
