#!/usr/bin/env python
"""Receiver: receive UNITE latents and decode them to images.

  python rx.py --decoder exported/decoder.pt --listen 5000 --out-dir received/
  python rx.py --decoder exported/decoder.pt --in latents.bin --out-dir received/
  python rx.py --decoder exported/decoder.pt --iq-in rx_iq --out-dir received/
      (rx_iq.c64: received complex64 samples, already synchronized/equalized;
       rx_iq.json: side info written by tx.py)
"""

import argparse
import json
import os
import socket

import numpy as np
import torch
from torchvision.utils import save_image

from models.split import UNITEDecoder, load_split
from utils.latent_link import unpack, recv_msg, from_iq


def iter_payload_file(path):
    with open(path, "rb") as f:
        while size := f.read(8):
            yield f.read(int.from_bytes(size, "little"))


def iter_socket(port):
    server = socket.create_server(("0.0.0.0", port))
    print(f"[RX] Listening on port {port}")
    conn, addr = server.accept()
    print(f"[RX] Connected: {addr}")
    with conn, server:
        while (data := recv_msg(conn)) is not None:
            yield data


def main():
    parser = argparse.ArgumentParser(description="UNITE receiver (decoder side).")
    parser.add_argument("--decoder", type=str, required=True, help="Exported decoder.pt")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--listen", type=int, help="TCP port to listen on.")
    src.add_argument("--in", dest="in_file", type=str, help="Payload file written by tx.py --out.")
    src.add_argument("--iq-in", type=str, help="Prefix of received .c64 samples + .json side info.")
    parser.add_argument("--out-dir", type=str, default="received")
    parser.add_argument("--batch-size", type=int, default=16, help="Decode batch size for --iq-in.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    decoder = load_split(args.decoder, UNITEDecoder, args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.iq_in:
        with open(f"{args.iq_in}.json") as f:
            side = json.load(f)
        n_sym = side["num_tokens"] * side["dim"] // 2
        symbols = np.fromfile(f"{args.iq_in}.c64", dtype="<c8").reshape(-1, n_sym)
        scales = np.asarray(side["scales"], dtype=np.float32)
        latents = (from_iq(symbols[i:i + args.batch_size], scales[i:i + args.batch_size],
                           side["num_tokens"], side["dim"])
                   for i in range(0, len(symbols), args.batch_size))
    else:
        source = iter_socket(args.listen) if args.listen else iter_payload_file(args.in_file)
        latents = (unpack(data)[0] for data in source)

    count = 0
    for z in latents:
        imgs = decoder(z.to(args.device))
        for img in imgs:
            save_image(img, os.path.join(args.out_dir, f"{count:06d}.png"))
            count += 1
        print(f"[RX] Decoded {imgs.shape[0]} images (total {count})")


if __name__ == "__main__":
    main()
