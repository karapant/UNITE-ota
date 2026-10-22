#!/usr/bin/env python
"""Transmitter: encode images to UNITE latents and send/write them.

  # Digital link: send payloads to the receiver PC over TCP
  python tx.py --encoder exported/encoder.pt --images img_dir/ --send 192.168.1.20:5000

  # Payload file (hand to your own modem/link)
  python tx.py --encoder exported/encoder.pt --images a.jpg b.jpg --out latents.bin

  # Analog over the air: complex64 baseband samples (GNU Radio file format) + side info
  python tx.py --encoder exported/encoder.pt --images a.jpg --iq-out tx_iq
      -> tx_iq.c64 (unit-power symbols, 4096 per image), tx_iq.json (scales, shape)
"""

import argparse
import json
import socket
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.split import UNITEEncoder, load_split
from utils.crop import CenterCropTransform
from utils.latent_link import pack, send_msg, to_iq

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}


def list_images(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.iterdir() if f.suffix in IMG_EXTS) if p.is_dir() else [p]
    return files


def main():
    parser = argparse.ArgumentParser(description="UNITE transmitter (encoder side).")
    parser.add_argument("--encoder", type=str, required=True, help="Exported encoder.pt")
    parser.add_argument("--images", nargs="+", required=True, help="Image files and/or directories.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--format", choices=["fp16", "fp32", "iq"], default="fp16", help="Payload format.")
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument("--send", type=str, help="host:port of the receiver (TCP).")
    out.add_argument("--out", type=str, help="Write all payloads to this file.")
    out.add_argument("--iq-out", type=str, help="Prefix for raw complex64 samples (.c64) + side info (.json).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the encoder's input noise.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    encoder = load_split(args.encoder, UNITEEncoder, args.device)
    transform = transforms.Compose([CenterCropTransform(256), transforms.ToTensor()])
    files = list_images(args.images)

    sock = None
    if args.send:
        host, port = args.send.rsplit(":", 1)
        sock = socket.create_connection((host, int(port)))
    out_file = open(args.out, "wb") if args.out else None
    iq_file = open(f"{args.iq_out}.c64", "wb") if args.iq_out else None
    iq_scales = []

    for i in range(0, len(files), args.batch_size):
        batch = files[i:i + args.batch_size]
        imgs = torch.stack([transform(Image.open(f).convert("RGB")) for f in batch]).to(args.device)
        z = encoder(imgs)

        if iq_file:
            symbols, scale = to_iq(z)
            iq_file.write(symbols.astype("<c8").tobytes())
            iq_scales += scale.tolist()
            print(f"[TX] {len(batch)} images -> {symbols.size} IQ symbols")
            continue

        payload = pack(z, args.format)
        if sock:
            send_msg(sock, payload)
        else:
            out_file.write(len(payload).to_bytes(8, "little") + payload)
        print(f"[TX] {len(batch)} images -> {len(payload)} bytes ({args.format})")

    if sock:
        sock.close()
    if out_file:
        out_file.close()
    if iq_file:
        iq_file.close()
        with open(f"{args.iq_out}.json", "w") as f:
            json.dump({"num_tokens": z.shape[1], "dim": z.shape[2], "scales": iq_scales,
                       "files": [f.name for f in files]}, f, indent=1)


if __name__ == "__main__":
    main()
