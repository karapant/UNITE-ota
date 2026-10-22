#!/usr/bin/env python
"""Monte Carlo evaluation of the split UNITE pipeline over wireless channel scenarios.

For every scenario in configs/channels.yaml, every image and every draw:
    encode (once per image) -> IQ symbols -> channel (SNR, fading, noise, CSI error, equalization)
    -> decode -> PSNR / LPIPS against the original.
Channel draws use fixed seeds per (scenario, image), so different models see identical channels.

Writes <out>/report.md (tables + figures), draws.csv (one row per draw), summary.json.

  python channel_experiments.py --exported ota_test/exported/models_UNITE-B --images ota_test/images
"""

import argparse
import csv
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.split import UNITEDecoder, UNITEEncoder, Profiler, default_device, load_split, peak_rss_mb
from utils.channel import Channel, doppler_hz, load_scenarios
from utils.crop import CenterCropTransform
from utils.latent_link import from_iq, pack, to_iq

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Figure styling: validated categorical order (fixed, never cycled), text/surface tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------- metrics & statistics
def psnr(rec, target):
    mse = ((rec - target) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).cpu().numpy()


def lpips_dist(model, rec, target):
    if model is None:
        return np.full(rec.shape[0], np.nan)
    with torch.no_grad():  # LPIPS expects images in [-1, 1]
        d = model(rec * 2 - 1, target.expand_as(rec) * 2 - 1, reduction="none")
    return d.flatten().cpu().numpy()


def cluster_ci(per_image, stat, rng, n=1000):
    """95% CI of stat(values), resampling whole images (draws of one image are not independent)."""
    if len(per_image) < 2:
        return float("nan"), float("nan")
    ests = [stat(np.concatenate([per_image[j] for j in rng.integers(0, len(per_image), len(per_image))]))
            for _ in range(n)]
    return tuple(float(v) for v in np.percentile(ests, [2.5, 97.5]))


def to_uint8(x):
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


# ---------------------------------------------------------------------------- figures
def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def make_figures(out, names, rows, summary, clean, examples, args, plt):
    figs = {}
    color = {n: SERIES[k] for k, n in enumerate(names)}
    by = {n: [r for r in rows if r["scenario"] == n] for n in names}

    # 1. CDF of PSNR per scenario (lines: legend + table carry identity)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style(ax)
    for n in names:
        v = np.sort([r["psnr"] for r in by[n]])
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color[n], linewidth=2, label=n)
    ref = float(np.mean([c["psnr"] for c in clean]))
    ax.axvline(ref, color=INK_2, linewidth=1)
    ax.text(ref, 1.02, " no channel", color=INK_2, fontsize=9, va="bottom", ha="left", transform=ax.get_xaxis_transform())
    ax.set_xlabel("PSNR vs original (dB)")
    ax.set_ylabel("fraction of transmissions ≤ x")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    ax.set_title("Reconstruction quality per scenario (CDF over all images and draws)", color=INK, fontsize=11, loc="left")
    figs["cdf"] = "figures/psnr_cdf.png"
    fig.savefig(out / figs["cdf"], dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # 2. Small multiples: PSNR loss vs effective SNR, one panel per scenario
    cols = min(len(names), 3)
    nrows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 3.2 * nrows), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for idx in range(len(names), nrows * cols):  # empty cells: hide, and give the panel above its x-axis back
        axes.flat[idx].set_visible(False)
        axes.flat[idx - cols].tick_params(labelbottom=True)
        axes.flat[idx - cols].set_xlabel("effective SNR (dB)")
    for ax, n in zip(axes.flat, names):
        style(ax)
        ax.scatter([r["effective_snr_db"] for r in by[n]], [r["delta_psnr"] for r in by[n]],
                   s=16, color=color[n], alpha=0.4, edgecolors=SURFACE, linewidths=0.5)
        ax.axhline(-args.outage_db, color=INK_2, linewidth=1)
        ax.set_title(n, color=INK, fontsize=10, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("effective SNR (dB)")
    for ax in axes[:, 0]:
        ax.set_ylabel("PSNR change vs no channel (dB)")
    fig.suptitle(f"Quality loss vs channel quality (line = outage threshold, −{args.outage_db:g} dB)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    figs["scatter"] = "figures/psnr_loss_vs_snr.png"
    fig.savefig(out / figs["scatter"], dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # 3. Outage probability per scenario with 95% CI, value at the bar tip
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(names) + 1.2), facecolor=SURFACE)
    style(ax)
    ax.grid(False, axis="y")
    ys = np.arange(len(names))[::-1]
    vals = [100 * summary[n]["outage"] for n in names]
    lo = [100 * summary[n]["outage_ci"][0] for n in names]
    hi = [100 * summary[n]["outage_ci"][1] for n in names]
    ax.barh(ys, vals, height=0.45, color=[color[n] for n in names])
    err = np.array([np.subtract(vals, lo), np.subtract(hi, vals)])
    if np.isfinite(err).all():
        ax.errorbar(vals, ys, xerr=np.clip(err, 0, None), fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
    for y, v, h in zip(ys, vals, hi):
        ax.text((h if np.isfinite(h) else v) + 1, y, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(ys, names)
    ax.set_xlim(0, max(105, max(vals) + 15))
    ax.set_xlabel(f"outage probability (%): PSNR more than {args.outage_db:g} dB below no-channel")
    ax.set_title("Outage per scenario (whiskers: 95% CI over images)", color=INK, fontsize=11, loc="left")
    figs["outage"] = "figures/outage.png"
    fig.savefig(out / figs["outage"], dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # 4. Example image: original | no channel | median draw | 10th-percentile draw, per scenario
    fig, axes = plt.subplots(len(names), 4, figsize=(10, 2.6 * len(names)), facecolor=SURFACE, squeeze=False)
    heads = ["original", "no channel", "median draw", "10th percentile draw"]
    for row, n in enumerate(names):
        r0 = sorted((r for r in by[n] if r["image"] == 0), key=lambda r: r["psnr"])
        picks = [r0[len(r0) // 2], r0[int(0.1 * (len(r0) - 1))]]
        panels = [(examples["original"], "")] + [(examples["clean"], f"{clean[0]['psnr']:.1f} dB")] + \
                 [(examples[n][p["draw"]], f"{p['psnr']:.1f} dB @ {p['effective_snr_db']:.0f} dB SNR") for p in picks]
        for col, (img, cap) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(heads[col], color=INK, fontsize=10)
            if cap:
                ax.set_xlabel(cap, color=INK_2, fontsize=8)
            if col == 0:
                ax.set_ylabel(n, color=INK, fontsize=10)
    fig.tight_layout()
    figs["examples"] = "figures/examples.png"
    fig.savefig(out / figs["examples"], dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return figs


# ---------------------------------------------------------------------------- report
def fmt_ci(v, ci, f="{:.2f}"):
    if not all(np.isfinite(ci)):
        return f.format(v)
    return f"{f.format(v)} [{f.format(ci[0])}, {f.format(ci[1])}]"


def write_report(out, args, link, scenarios, names, summary, clean, sizes, prof_stats, figs, wall_s, model_info):
    image_ms = 4096 / (float(link["symbol_rate_msps"]) * 1e3)
    L = ["# Channel scenario report", ""]
    L += [f"- **Model:** `{args.exported or args.encoder}` ({model_info})",
          f"- **Images:** {len(clean)} from `{args.images}` · **draws:** {args.draws} per image per scenario "
          f"· **total:** {len(clean) * args.draws * len(names):,} transmissions",
          f"- **Device:** {prof_stats['device']} · **seed:** {args.seed} (identical channel draws for every model)",
          f"- **Link:** {link['carrier_ghz']} GHz carrier, {link['symbol_rate_msps']} Msym/s "
          f"(one image = 4096 symbols = {image_ms:.1f} ms), {link.get('equalizer', 'mmse').upper()} equalizer",
          f"- **Outage:** PSNR more than {args.outage_db:g} dB below the image's no-channel reconstruction",
          f"- Wall time {wall_s / 60:.1f} min. Raw data: `draws.csv`, `summary.json`.", ""]

    L += ["## Scenarios", "",
          "| scenario | description | mean SNR | shadowing σ | fading | speed | Doppler | Doppler × image | CSI NMSE |",
          "|:--|:--|--:|--:|:--|--:|--:|--:|--:|"]
    for n in names:
        p = scenarios[n]
        fd = doppler_hz(float(p.get("speed_kmh", 0)), float(link["carrier_ghz"]))
        fading = "Rayleigh" if p.get("k_factor_db") is None else f"Rician K={p['k_factor_db']} dB"
        L.append(f"| {n} | {p.get('description', '')} | {p['snr_mean_db']} dB | {p.get('shadowing_std_db', 0)} dB | "
                 f"{fading} | {p.get('speed_kmh', 0)} km/h | {fd:.0f} Hz | {fd * image_ms / 1e3:.3f} | "
                 f"{p.get('csi_nmse_db', '–')} dB |")
    L += ["", "*Doppler × image* ≈ fading cycles during one image: ≪ 0.1 means the channel is constant per image, "
          "≳ 1 means it changes within the image.", ""]

    L += ["## Results", "",
          "| scenario | drawn SNR (mean) | effective SNR (mean) | PSNR mean [95% CI] | median | p10 | p5 | "
          "Δ vs no channel | LPIPS mean | outage [95% CI] |",
          "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    L.append(f"| *no channel* | – | – | {np.mean([c['psnr'] for c in clean]):.2f} | – | – | – | 0 | "
             f"{np.nanmean([c['lpips'] for c in clean]):.3f} | – |")
    for n in names:
        s = summary[n]
        L.append(f"| {n} | {s['snr_db']:.1f} dB | {s['effective_snr_db']:.1f} dB | {fmt_ci(s['psnr'], s['psnr_ci'])} | "
                 f"{s['psnr_median']:.2f} | {s['psnr_p10']:.2f} | {s['psnr_p5']:.2f} | {s['delta_psnr']:+.2f} | "
                 f"{s['lpips']:.3f} | {fmt_ci(100 * s['outage'], [100 * c for c in s['outage_ci']], '{:.1f}')} % |")
    L += ["", "PSNR in dB against the original image; LPIPS lower is better. CIs resample whole images, so they "
          "widen with few images.", ""]

    L += ["## Figures", "", f"![PSNR CDF]({figs['cdf']})", "", f"![PSNR loss vs SNR]({figs['scatter']})", "",
          f"![Outage]({figs['outage']})", "", f"![Examples]({figs['examples']})", ""]

    raw = 256 * 256 * 3
    lat = sizes[0]["latent_fp16_bytes"]
    L += ["## Image size vs latent size", "",
          f"Each image is center-cropped to 256×256 RGB ({raw:,} bytes raw) and encoded to a 256×32 latent: "
          f"{lat:,} bytes as fp16 payload, or 4,096 complex channel uses over the air "
          f"(bandwidth ratio 4096 / {raw:,} = 1/{raw // 4096}).", "",
          "| image | original | file size | raw 256² RGB | latent (fp16) | file / latent | raw / latent |",
          "|:--|--:|--:|--:|--:|--:|--:|"]
    for z in sizes:
        L.append(f"| {z['file']} | {z['width']}×{z['height']} | {z['file_bytes']:,} B | {raw:,} B | {z['latent_fp16_bytes']:,} B | "
                 f"{z['file_bytes'] / z['latent_fp16_bytes']:.1f}× | {raw / z['latent_fp16_bytes']:.1f}× |")
    fb = np.array([z["file_bytes"] for z in sizes], dtype=float)
    px = np.array([z["width"] * z["height"] for z in sizes], dtype=float)
    L += [f"| **mean ± std** | {px.mean() / 1e3:.0f}K ± {px.std() / 1e3:.0f}K px | {fb.mean():,.0f} ± {fb.std():,.0f} B | "
          f"{raw:,} B | {lat:,} B | {(fb / lat).mean():.1f} ± {(fb / lat).std():.1f}× | {raw / lat:.1f}× |", "",
          "The latent size is fixed per image; the variation comes from the originals (JPEG size depends on content "
          "and resolution).", ""]

    def p(k, unit=" ms", f="{:.2f}"):
        return f.format(prof_stats[k]) + unit if k in prof_stats else "–"
    L += ["## Profiling", "", "| stage | value |", "|:--|--:|",
          f"| model load (encoder + decoder + LPIPS) | {p('load_total_s', ' s')} |",
          f"| encode per image (once) | {p('encode_ms_per_item')} |",
          f"| channel per draw (fading, noise, equalization, de-mapping) | {p('channel_ms_per_item')} |",
          f"| decode per draw | {p('decode_ms_per_item')} |",
          f"| metrics per draw (PSNR + LPIPS) | {p('metrics_ms_per_item')} |",
          f"| throughput | {prof_stats['draws_per_s']:.1f} draws/s |",
          f"| encoder / decoder parameters | {model_info} |",
          f"| peak process memory | {prof_stats['peak_rss_mb']:.0f} MB |"]
    if "gpu_memory_mb" in prof_stats:
        L.append(f"| GPU memory held (MPS) | {prof_stats['gpu_memory_mb']:.0f} MB |")
    if "peak_gpu_mb" in prof_stats:
        L.append(f"| peak GPU memory (CUDA) | {prof_stats['peak_gpu_mb']:.0f} MB |")
    L += ["", "## Caveats", "",
          "- The model was trained **without channel noise**; these results are the baseline before any noise-aware training.",
          "- Scenario parameters are starting points to calibrate against 3GPP TR 38.901 and the target link budget.",
          "- Draws are independent transmissions; consecutive-image correlation (e.g. outage durations) is not modelled.", ""]
    (out / "report.md").write_text("\n".join(L))


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Monte Carlo channel-scenario evaluation of the UNITE TX/RX split.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--exported", type=str, help="Directory with encoder.pt and decoder.pt (export_split.py).")
    src.add_argument("--encoder", type=str, help="encoder.pt (use with --decoder).")
    ap.add_argument("--decoder", type=str)
    ap.add_argument("--images", type=str, required=True, help="Directory of test images.")
    ap.add_argument("--config", type=str, default="configs/channels.yaml")
    ap.add_argument("--scenarios", nargs="*", help="Subset of scenarios (default: all in the config).")
    ap.add_argument("--draws", type=int, default=100, help="Channel draws per image per scenario.")
    ap.add_argument("--batch", type=int, default=50, help="Draws decoded per batch.")
    ap.add_argument("--outage-db", type=float, default=3.0, help="Outage: PSNR this far below the no-channel PSNR.")
    ap.add_argument("--out", type=str, default="ota_test/channel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=default_device())
    ap.add_argument("--no-lpips", action="store_true", help="Skip LPIPS (faster).")
    args = ap.parse_args()

    t0 = time.perf_counter()
    enc_path = Path(args.exported, "encoder.pt") if args.exported else Path(args.encoder)
    dec_path = Path(args.exported, "decoder.pt") if args.exported else Path(args.decoder)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    link, scenarios = load_scenarios(args.config)
    names = args.scenarios or list(scenarios)
    unknown = [n for n in names if n not in scenarios]
    if unknown:
        raise SystemExit(f"Unknown scenarios {unknown}; available: {list(scenarios)}")
    if len(names) > len(SERIES):
        raise SystemExit(f"At most {len(SERIES)} scenarios per report (one color each); pass --scenarios")

    prof = Profiler("channel", args.device)
    with prof.time("load"):
        enc = load_split(str(enc_path), UNITEEncoder, args.device)
        dec = load_split(str(dec_path), UNITEDecoder, args.device)
        lpips = None
        if not args.no_lpips:
            from modules.autoencoding_utils.lpips import LPIPS
            lpips = LPIPS().to(args.device).eval()
    n_enc = sum(p.numel() for p in enc.parameters()) / 1e6
    n_dec = sum(p.numel() for p in dec.parameters()) / 1e6
    model_info = f"{n_enc:.1f}M / {n_dec:.1f}M params"

    files = sorted(f for f in Path(args.images).iterdir() if f.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images in {args.images}")
    prep = transforms.Compose([CenterCropTransform(256), transforms.ToTensor()])
    originals, sizes = [], []
    for f in files:
        im = Image.open(f).convert("RGB")
        sizes.append({"file": f.name, "file_bytes": f.stat().st_size, "width": im.width, "height": im.height})
        originals.append(prep(im))
    originals = torch.stack(originals)
    n_img = len(files)

    # 1. encode each image once (fixed encoder noise per image)
    symbols, scales, latents = [], [], []
    for i in tqdm(range(n_img), desc="encode", unit="img", dynamic_ncols=True):
        torch.manual_seed(args.seed + i)
        with prof.time("encode"):
            z = enc(originals[i:i + 1].to(args.device))
        z = z.cpu()
        latents.append(z)
        s, sc = to_iq(z)
        symbols.append(s[0])
        scales.append(float(sc[0]))
        sizes[i]["latent_fp16_bytes"] = len(pack(z, "fp16"))
    num_tokens, dim = latents[0].shape[1:]

    # 2. no-channel reference reconstruction per image
    clean, examples = [], {"original": to_uint8(originals[0])}
    with torch.no_grad():
        for i in tqdm(range(n_img), desc="no-channel reference", unit="img", dynamic_ncols=True):
            target = originals[i:i + 1].to(args.device)
            rec = dec(latents[i].to(args.device))
            clean.append({"psnr": float(psnr(rec, target)[0]), "lpips": float(lpips_dist(lpips, rec, target)[0])})
            if i == 0:
                examples["clean"] = to_uint8(rec[0])

    # 3. Monte Carlo over scenarios x images x draws
    rows = []
    total = len(names) * n_img * args.draws
    with torch.no_grad(), tqdm(total=total, desc="channel draws", unit="draw", dynamic_ncols=True) as bar:
        for name in names:
            channel = Channel(scenarios[name], link)
            keep = []
            for i in range(n_img):
                rng = np.random.default_rng([args.seed, zlib.crc32(name.encode()), i])
                target = originals[i:i + 1].to(args.device)
                bar.set_postfix_str(f"{name} · image {i + 1}/{n_img}")
                for d0 in range(0, args.draws, args.batch):
                    b = min(args.batch, args.draws - d0)
                    with prof.time("channel", b):
                        x_hat, info = channel(np.repeat(symbols[i][None], b, axis=0), rng)
                        z_hat = from_iq(x_hat, np.full(b, scales[i], dtype=np.float32), num_tokens, dim)
                    with prof.time("decode", b):
                        rec = dec(z_hat.to(args.device))
                    with prof.time("metrics", b):
                        q = psnr(rec, target)
                        lp = lpips_dist(lpips, rec, target)
                    for j in range(b):
                        rows.append({"scenario": name, "image": i, "file": files[i].name, "draw": d0 + j,
                                     "snr_db": float(info["snr_db"][j]),
                                     "effective_snr_db": float(info["effective_snr_db"][j]),
                                     "fading_gain_db": float(info["fading_gain_db"][j]),
                                     "psnr": float(q[j]), "lpips": float(lp[j]),
                                     "psnr_clean": clean[i]["psnr"], "delta_psnr": float(q[j]) - clean[i]["psnr"]})
                    if i == 0:
                        keep.append(to_uint8(rec))
                    bar.update(b)
            examples[name] = torch.cat(keep)

    # 4. statistics
    rng = np.random.default_rng(args.seed)
    summary = {}
    for name in names:
        r = [x for x in rows if x["scenario"] == name]
        per_img_psnr = [np.array([x["psnr"] for x in r if x["image"] == i]) for i in range(n_img)]
        per_img_out = [np.array([x["delta_psnr"] < -args.outage_db for x in r if x["image"] == i], dtype=float)
                       for i in range(n_img)]
        v = np.concatenate(per_img_psnr)
        o = np.concatenate(per_img_out)
        summary[name] = {
            "snr_db": float(np.mean([x["snr_db"] for x in r])),
            "effective_snr_db": float(np.mean([x["effective_snr_db"] for x in r])),
            "psnr": float(v.mean()), "psnr_ci": cluster_ci(per_img_psnr, np.mean, rng),
            "psnr_median": float(np.median(v)), "psnr_p10": float(np.percentile(v, 10)),
            "psnr_p5": float(np.percentile(v, 5)),
            "delta_psnr": float(np.mean([x["delta_psnr"] for x in r])),
            "lpips": float(np.nanmean([x["lpips"] for x in r])),
            "outage": float(o.mean()), "outage_ci": cluster_ci(per_img_out, np.mean, rng),
        }

    # 5. outputs
    with open(out / "draws.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    wall_s = time.perf_counter() - t0
    prof_stats = prof.report(None, images=n_img, draws=total, draws_per_s=round(total / max(wall_s, 1e-9), 2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = make_figures(out, names, rows, summary, clean, examples, args, plt)
    write_report(out, args, link, scenarios, names, summary, clean, sizes, prof_stats, figs, wall_s, model_info)
    json.dump({"args": vars(args), "link": link, "scenarios": {n: scenarios[n] for n in names},
               "no_channel": clean, "summary": summary, "sizes": sizes, "profile": prof_stats},
              open(out / "summary.json", "w"), indent=1, default=float)

    print(f"\n  {'scenario':<15} {'PSNR mean':>10} {'p10':>7} {'Δ vs clean':>11} {'LPIPS':>7} {'outage':>8}")
    print(f"  {'no channel':<15} {np.mean([c['psnr'] for c in clean]):>10.2f} {'':>7} {'':>11} "
          f"{np.nanmean([c['lpips'] for c in clean]):>7.3f}")
    for name in names:
        s = summary[name]
        print(f"  {name:<15} {s['psnr']:>10.2f} {s['psnr_p10']:>7.2f} {s['delta_psnr']:>+11.2f} "
              f"{s['lpips']:>7.3f} {100 * s['outage']:>7.1f}%")
    print(f"\n  report: {out / 'report.md'}   ({wall_s / 60:.1f} min, peak memory {peak_rss_mb():.0f} MB)")


if __name__ == "__main__":
    main()
