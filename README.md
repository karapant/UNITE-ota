# UNITE Split: Encoder/Decoder Deployment for Over-the-Air Latent Transmission

This repository builds on **[UNITE](https://github.com/ShivamDuggal4/UNITE-tokenization-generation)** ([paper](https://arxiv.org/abs/2603.22283)) by Duggal, Bai et al. (MIT, Adobe). For model details, setup, datasets, and training, see the original repository; the model and training code here are unchanged.

## What this repository adds

A trained UNITE model is split into two standalone halves that run on **different machines** and exchange only the latent representation, e.g. over a wireless link:

```
 PC 1 (transmitter)                                          PC 2 (receiver)
 image ─► encoder.pt ─► latent z (256×32) ─►  ~ ~ link ~ ~  ─► decoder.pt ─► image
```

| Added file | Purpose |
|:--|:--|
| `models/split.py` | `UNITEEncoder` / `UNITEDecoder`: standalone TX and RX modules, built from UNITE's own code, and checkpoint splitting |
| `export_split.py` | Splits a UNITE training checkpoint into `encoder.pt` (~132M params) and `decoder.pt` (~87M params) |
| `tx.py` | Transmitter: images → latents → TCP, payload file, or raw IQ samples |
| `rx.py` | Receiver: TCP, payload file, or raw IQ samples → latents → images |
| `utils/latent_link.py` | Latent serialization (fp16/fp32/IQ), complex baseband mapping, TCP framing |

- **No change to training.** The model is trained exactly as in the original UNITE and split afterwards.
- **Identical results.** Chaining the encoder and decoder reproduces UNITE's `reconstruct_batch` bit-exactly.
- **Each side is independent.** The receiver needs neither the encoder weights nor LPIPS, and the transmitter doesn't need the decoder.

## Usage

Train UNITE as described in the [original repository](https://github.com/ShivamDuggal4/UNITE-tokenization-generation), then:

```bash
# 1. Split a checkpoint (EMA weights by default)
python export_split.py --ckpt outputs/lets-unite/checkpoints/0400000.pt --config configs/imagenet.yaml --out-dir exported

# 2a. Digital link (TCP), e.g. over Wi-Fi: start the receiver first
python rx.py --decoder exported/decoder.pt --listen 5000 --out-dir received/          # PC 2
python tx.py --encoder exported/encoder.pt --images imgs/ --send <rx-ip>:5000          # PC 1

# 2b. Payload file, for your own modem/link stack
python tx.py --encoder exported/encoder.pt --images imgs/ --out latents.bin
python rx.py --decoder exported/decoder.pt --in latents.bin --out-dir received/

# 2c. Analog over the air (e.g. SDR / GNU Radio): raw complex64 baseband samples
python tx.py --encoder exported/encoder.pt --images imgs/ --iq-out tx_iq               # -> tx_iq.c64 + tx_iq.json
python rx.py --decoder exported/decoder.pt --iq-in rx_iq --out-dir received/           # rx_iq.c64 + rx_iq.json
```

### Latent formats

Each 256×256 image is represented by a latent of 256 tokens × 32 dims = 8192 values.

| Format | Per image | Notes |
|:--|:--|:--|
| `fp16` (default) | 16 KiB | payload bytes for any digital link |
| `fp32` | 32 KiB | lossless latent |
| `iq` | 4096 complex64 symbols | consecutive latent dims paired into I/Q, unit average power per image |

### Notes on IQ mode

- `rx_iq.c64` must contain the received symbols **already synchronized and equalized**, in transmit order. Timing and frequency synchronization and channel estimation belong in your radio chain.
- `tx_iq.json` holds side info (per-image power scale and shape). It must reach the receiver too, e.g. over a separate digital link; rename it to `rx_iq.json`.
- The model is trained **without channel noise**, so reconstruction quality degrades as SNR decreases.

## Acknowledgement

All credit for the UNITE model and training code goes to the original authors:

```bibtex
@article{duggal2026unite,
  title={End-to-End Training for Unified Tokenization and Latent Denoising},
  author={Shivam Duggal and Xingjian Bai and Zongze Wu and Richard Zhang and Eli Shechtman and Antonio Torralba and Phillip Isola and William T. Freeman},
  journal={arXiv preprint arXiv:2603.22283},
  year={2026}
}
```
