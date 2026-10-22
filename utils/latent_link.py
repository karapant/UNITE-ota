"""
Serialization of UNITE latents z [B, 256, 32] for transport between the TX and RX PCs.

Two representations:
  - Payload bytes (pack/unpack): self-describing header + data, for any digital link
    (TCP/UDP, Wi-Fi, a modem, a file). Formats: "fp16" (16 KiB/image), "fp32", "iq".
  - Complex baseband symbols (to_iq/from_iq): consecutive latent dims paired into
    complex64 symbols (256*32/2 = 4096 per image), normalized to unit average power per
    image. The per-image scale is side information the receiver needs to undo the
    normalization. Raw complex64 is the GNU Radio file-source/-sink sample format.
"""

import struct
import numpy as np
import torch

MAGIC = b"UNTL"
_HEADER = struct.Struct("<4sBHHH")  # magic, fmt, B, N, D
_FMTS = {"fp32": 0, "fp16": 1, "iq": 2}
_FMT_NAMES = {v: k for k, v in _FMTS.items()}


def to_iq(z):
    """z [B, N, D] (D even) -> (symbols complex64 [B, N*D/2], scale float32 [B])."""
    z = z.detach().float().cpu()
    x = torch.view_as_complex(z.reshape(z.shape[0], -1, 2).contiguous())
    scale = x.abs().pow(2).mean(dim=1).sqrt().clamp_min(1e-12)
    return (x / scale[:, None]).numpy(), scale.numpy()


def from_iq(symbols, scale, num_tokens=256, dim=32):
    """Inverse of to_iq. symbols: complex [B, N*D/2] (e.g. after the channel), scale: [B]."""
    x = torch.from_numpy(np.array(symbols, dtype=np.complex64))
    x = x * torch.tensor(np.asarray(scale), dtype=torch.float32)[:, None]
    return torch.view_as_real(x).reshape(x.shape[0], num_tokens, dim)


def pack(z, fmt="fp16"):
    """Latent z [B, N, D] -> bytes."""
    B, N, D = z.shape
    header = _HEADER.pack(MAGIC, _FMTS[fmt], B, N, D)
    if fmt == "iq":
        symbols, scale = to_iq(z)
        return header + scale.astype("<f4").tobytes() + symbols.astype("<c8").tobytes()
    dtype = "<f2" if fmt == "fp16" else "<f4"
    return header + z.detach().float().cpu().numpy().astype(dtype).tobytes()


def unpack(data):
    """bytes -> (latent z float32 [B, N, D], fmt)."""
    magic, code, B, N, D = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("Not a UNITE latent payload")
    fmt = _FMT_NAMES[code]
    body = memoryview(data)[_HEADER.size:]
    if fmt == "iq":
        scale = np.frombuffer(body[:4 * B], dtype="<f4")
        symbols = np.frombuffer(body[4 * B:], dtype="<c8").reshape(B, N * D // 2)
        return from_iq(symbols, scale, N, D), fmt
    dtype = "<f2" if fmt == "fp16" else "<f4"
    z = np.frombuffer(body, dtype=dtype).reshape(B, N, D).astype(np.float32)
    return torch.from_numpy(z), fmt


# Length-prefixed framing for stream sockets (TCP)
def send_msg(sock, data):
    sock.sendall(struct.pack("<Q", len(data)) + data)


def recv_msg(sock):
    def _exact(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)
    size = _exact(8)
    return None if size is None else _exact(struct.unpack("<Q", size)[0])
