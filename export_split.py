#!/usr/bin/env python
"""Split a UNITE training checkpoint into standalone encoder.pt (TX) and decoder.pt (RX).

python export_split.py --ckpt outputs/lets-unite/checkpoints/0400000.pt \
    --config configs/imagenet.yaml --out-dir exported/
"""

import argparse
import os

import yaml
import torch

from models.split import UNITEEncoder, UNITEDecoder, split_state_dict, save_split


def main():
    parser = argparse.ArgumentParser(description="Export UNITE encoder/decoder for split deployment.")
    parser.add_argument("--ckpt", type=str, required=True, help="UNITE training checkpoint (.pt).")
    parser.add_argument("--config", type=str, required=True, help="YAML config used for training.")
    parser.add_argument("--out-dir", type=str, default="exported")
    parser.add_argument("--weights", choices=["ema", "model"], default="ema",
                        help="Which weights to export (EMA is what evaluation uses).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        gen_tok_cfg = dict(yaml.safe_load(f)["gen_tok"])

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tx_state, rx_state = split_state_dict(ckpt[args.weights])

    encoder = UNITEEncoder(gen_tok_cfg)
    decoder = UNITEDecoder(gen_tok_cfg)
    encoder.load_state_dict(tx_state, strict=True)
    decoder.load_state_dict(rx_state, strict=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, module, state in (("encoder", encoder, tx_state), ("decoder", decoder, rx_state)):
        path = os.path.join(args.out_dir, f"{name}.pt")
        save_split(path, module, state)
        n_params = sum(p.numel() for p in module.parameters())
        print(f"Saved {name} ({n_params / 1e6:.1f}M params, {args.weights} weights, step {ckpt.get('step')}) -> {path}")


if __name__ == "__main__":
    main()
