#!/usr/bin/env bash
# End-to-end test of the split TX/RX pipeline:
#   1. pick a model: on the cluster, downloaded, or already split; only the missing steps run
#      (downloads are kept in ota_test/checkpoints/, splits in ota_test/exported/)
#   2. get N validation images from the cluster (one per class, spread over the classes; cached for offline runs)
#   3. split the checkpoint into encoder.pt / decoder.pt (export_split.py)
#   4. run rx.py and tx.py locally over TCP, then report PSNR, a side-by-side comparison, and
#      profiling (model load, encode/decode time per image, payload size, memory footprint)
#   5. optionally: Monte Carlo over the wireless channel scenarios in configs/channels.yaml
#      (channel_experiments.py) -> ota_test/channel/report.md
#
#   ./test_ota.sh        (everything is kept in ota_test/, which is gitignored)
#
# Cluster access uses .env (KUBE_CONTEXT, K8S_NAMESPACE, POD_LABELS, DATA_PATH, RESULTS_DIR); without it,
# or when the cluster is unreachable, local checkpoints and cached images are used.
# Optional overrides: PYTHON (default python3), EXPORT_CONFIG (default configs/imagenet.yaml).
set -euo pipefail
cd "$(dirname "$0")"
shopt -s nullglob

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    B=$'\e[1m' D=$'\e[2m' RED=$'\e[31m' GRN=$'\e[32m' YLW=$'\e[33m' CYN=$'\e[36m' N=$'\e[0m'
else
    B="" D="" RED="" GRN="" YLW="" CYN="" N=""
fi
log()  { printf '\n%s▶%s %s%s%s\n' "$CYN" "$N" "$B" "$*" "$N"; }
ok()   { printf '  %s✓%s %s\n' "$GRN" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$YLW" "$N" "$*" >&2; }
die()  { printf '\n  %s✗ %s%s\n' "$RED" "$*" "$N" >&2; exit 1; }

size_of()  { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
mtime_of() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1"; }
when()     { date -r "$1" '+%Y-%m-%d %H:%M' 2>/dev/null || date -d "@$1" '+%Y-%m-%d %H:%M'; }
gb()       { awk -v b="$1" 'BEGIN { printf "%.2f GB", b / 1e9 }'; }

PY=${PYTHON:-python3}
OUT=ota_test
mkdir -p "$OUT/checkpoints"

HAVE_ENV=0
if [[ -f .env ]]; then
    overrides=$(env | grep -E "^($(grep -oE '^[A-Z_][A-Z0-9_]*=' .env | tr -d '=' | paste -sd'|' -))=" || true)
    set -a; source .env
    while IFS= read -r kv; do [[ -n "$kv" ]] && export "$kv"; done <<< "$overrides"
    set +a
    HAVE_ENV=1
fi

POD="${JOB_NAME:-unite-ota}-fetch-$(date +%s)-$$"  # unique per run, so parallel runs don't interfere
kc() { kubectl --context "$KUBE_CONTEXT" -n "$K8S_NAMESPACE" "$@"; }
trap '(( CLUSTER )) && kc delete pod "$POD" --ignore-not-found --wait=false >/dev/null 2>&1; true' EXIT

# Appends stdin to $2 with a progress bar; $1 = total size in bytes, $3 = bytes already in $2
PROGRESS_PY='
import sys, time
total, path, start = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
done, t0, last = start, time.time(), 0.0
with open(path, "ab") as out:
    while chunk := sys.stdin.buffer.read(1 << 20):
        out.write(chunk)
        done += len(chunk)
        now = time.time()
        if now - last > 0.25 or done == total:
            last = now
            frac = done / total if total else 1.0
            rate = (done - start) / max(now - t0, 1e-6)
            eta = int((total - done) / rate) if rate > 0 else 0
            bar = "#" * int(30 * frac)
            sys.stderr.write(f"\r    [{bar:<30}] {100 * frac:5.1f}%  {done / 1e9:.2f}/{total / 1e9:.2f} GB"
                             f"  {rate / 1e6:5.1f} MB/s  ETA {eta // 60:02d}:{eta % 60:02d}")
sys.stderr.write("\n")
'

download() {  # $1 = remote file, $2 = size, $3 = local file; resumes from $3.part and verifies the checksum
    local remote=$1 size=$2 dest=$3 part="$3.part" have tries=0 remote_sum local_sum
    while true; do
        have=0; [[ -f "$part" ]] && have=$(size_of "$part")
        (( have == size )) && break
        if (( have > size )); then warn "Partial file is larger than the original, starting over"; rm -f "$part"; continue; fi
        (( tries++ < 5 )) || die "Download keeps dropping (have $(gb "$have") of $(gb "$size")); run again to resume"
        # Resume at a whole MiB: busybox dd seeks large files correctly (busybox tail -c +N does not past 2 GB)
        local block=1048576 skip=$((have / 1048576))
        if (( have > 0 )); then
            "$PY" -c 'import os, sys; os.truncate(sys.argv[1], int(sys.argv[2]))' "$part" $((skip * block))
            have=$((skip * block))
            printf '    %sresuming at %s%s\n' "$D" "$(gb "$have")" "$N"
        fi
        kc exec "$POD" -- dd if="$remote" bs=$block skip=$skip 2>/dev/null \
            | "$PY" -c "$PROGRESS_PY" "$size" "$part" "$have" || true
    done
    printf '    %sverifying checksum ...%s' "$D" "$N"
    remote_sum=$(kc exec "$POD" -- sha256sum "$remote" | cut -d' ' -f1)
    local_sum=$(shasum -a 256 "$part" 2>/dev/null || sha256sum "$part"); local_sum=${local_sum%% *}
    printf '\r'
    if [[ "$remote_sum" != "$local_sum" ]]; then
        rm -f "$part"; die "Checksum mismatch, the download was discarded; run again"
    fi
    mv "$part" "$dest"
    ok "Checksum verified, saved $dest (kept for next time)"
}

# ---------------------------------------------------------------------------- 1. cluster connection
CLUSTER=0
if (( HAVE_ENV )) && command -v kubectl >/dev/null 2>&1; then
    log "Connecting to the cluster"
    STORAGE_ROOT=$(dirname "$RESULTS_DIR")
    VAL_DIR="$(dirname "${DATA_PATH%/}")/val"
    for attempt in 1 2 3 4 5; do  # the admission webhook is occasionally unavailable
        if kc run "$POD" --image=alpine --restart=Never --labels="${POD_LABELS:-}" --command -- sleep 3600 >/dev/null 2>&1; then
            CLUSTER=1; break
        fi
        sleep 3
    done
    if (( CLUSTER )) && kc wait --for=condition=Ready "pod/$POD" --timeout=2m >/dev/null 2>&1; then
        ok "Connected to $K8S_NAMESPACE"
    else
        CLUSTER=0; warn "Cluster not reachable (VPN or login?): showing local checkpoints only"
    fi
else
    warn "No .env or kubectl: showing local checkpoints only"
fi

# ---------------------------------------------------------------------------- 2. choose a model
# A model is known by name (derived from its cluster path) and can be: on the cluster, downloaded
# (ota_test/checkpoints/<name>.pt, or a training checkpoint in the repo root), and/or split
# (ota_test/exported/<name>/{encoder,decoder}.pt). Only the missing steps are run.
local_ckpt_of() {  # $1 = name -> path of a local checkpoint, or nothing
    if [[ -f "$OUT/checkpoints/$1.pt" ]]; then echo "$OUT/checkpoints/$1.pt"
    elif [[ -f "$1.pt" ]]; then echo "$1.pt"; fi
}
split_ready() {  # $1 = name, $2 = mtime of the cluster checkpoint (0 if unknown)
    local enc="$OUT/exported/$1/encoder.pt" ckpt
    [[ "${FORCE_SPLIT:-0}" != 1 && -f "$enc" && -f "$OUT/exported/$1/decoder.pt" ]] || return 1
    ckpt=$(local_ckpt_of "$1")
    [[ -z "$ckpt" || "$enc" -nt "$ckpt" ]] && (( $(mtime_of "$enc") >= ${2:-0} ))
}

kinds=(); names=(); rpaths=(); sizes=(); rmtimes=(); labels=()
add() { kinds+=("$1"); names+=("$2"); rpaths+=("$3"); sizes+=("$4"); rmtimes+=("$5"); labels+=("$6"); }
listed() { local n; for n in ${names[@]+"${names[@]}"}; do [[ "$n" == "$1" ]] && return 0; done; return 1; }

if (( CLUSTER )); then
    now=$(kc exec "$POD" -- date +%s)
    listing=$(kc exec "$POD" -- sh -c "find '$STORAGE_ROOT' -maxdepth 4 -name '*.pt' -type f -exec stat -c '%s %Y %n' {} + 2>/dev/null" \
        | awk -v root="$STORAGE_ROOT/" '{ rel = substr($3, length(root) + 1); print ((rel ~ /^results\//) ? 1 : 0), -$2, $0 }' \
        | sort -k1,1n -k2,2n | cut -d' ' -f3- || true)
    while read -r size mtime path; do
        [[ -n "$path" ]] || continue
        rel=${path#"$STORAGE_ROOT"/}
        add cluster "$(echo "$rel" | sed 's#/checkpoints/#_#; s#/#_#g; s#\.pt$##')" "$path" "$size" "$mtime" "$rel"
    done <<< "$listing"
fi
for f in "$OUT"/checkpoints/*.pt ./*.pt "$OUT"/exported/*/encoder.pt; do  # models only on this machine
    case "$f" in */encoder.pt) n=$(basename "$(dirname "$f")") ;; *) n=$(basename "${f%.pt}") ;; esac
    listed "$n" && continue
    ckpt=$(local_ckpt_of "$n")
    if [[ -n "$ckpt" ]]; then add local "$n" "" "$(size_of "$ckpt")" 0 "$ckpt"
    else add local "$n" "" $(( $(size_of "$OUT/exported/$n/encoder.pt") + $(size_of "$OUT/exported/$n/decoder.pt") )) 0 "$n (encoder + decoder)"; fi
done
(( ${#names[@]} )) || die "No models found locally or on the cluster"

log "Choose a model"
printf '  %s✓ split = encoder/decoder ready, starts right away   ✓ local = downloaded, needs a split%s\n' "$D" "$N"
section=""
for i in "${!names[@]}"; do
    if [[ "${kinds[$i]}" != "$section" ]]; then
        section=${kinds[$i]}
        if [[ $section == local ]]; then printf '\n  %sOn this machine%s\n' "$B" "$N"
        else printf '\n  %sOn the cluster%s %s(%s, downloaded on first use)%s\n' "$B" "$N" "$D" "$STORAGE_ROOT" "$N"; fi
    fi
    n=${names[$i]} note=""
    [[ ${kinds[$i]} == cluster ]] && note=$(when "${rmtimes[$i]}")
    ckpt=$(local_ckpt_of "$n")
    if split_ready "$n" "${rmtimes[$i]}"; then note+="  ${GRN}✓ split${N}"
    elif [[ -n "$ckpt" && ( ${kinds[$i]} == local || "$(size_of "$ckpt")" == "${sizes[$i]}" ) ]]; then note+="  ${GRN}✓ local${N}"
    elif [[ ${kinds[$i]} == cluster ]] && (( now - ${rmtimes[$i]} < 120 )); then note+="  ${YLW}written <2 min ago, may be incomplete${N}"; fi
    printf '    %s[%d]%s %-48s %7s  %s\n' "$CYN" "$((i + 1))" "$N" "${labels[$i]}" "$(gb "${sizes[$i]}")" "$note"
done

read -rp $'\n  Model to use [1]: ' choice; choice=${choice:-1}
[[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#names[@]} )) || die "Invalid choice: $choice"
read -rp "  Number of test images [8]: " num_images; num_images=${num_images:-8}
[[ "$num_images" =~ ^[0-9]+$ ]] && (( num_images >= 1 )) || die "Invalid number: $num_images"
c=$((choice - 1)); name=${names[$c]}
exported="$OUT/exported/$name"

# ---------------------------------------------------------------------------- 3. get the checkpoint (only if not split yet)
if split_ready "$name" "${rmtimes[$c]}"; then
    ok "Encoder/decoder ready: $exported (no download or split needed; FORCE_SPLIT=1 to redo)"
    need_split=0
else
    need_split=1
    local_ckpt=$(local_ckpt_of "$name")
    if [[ ${kinds[$c]} == cluster && ( -z "$local_ckpt" || "$(size_of "$local_ckpt")" != "${sizes[$c]}" ) ]]; then
        local_ckpt="$OUT/checkpoints/$name.pt"
        log "Downloading ${labels[$c]} → $local_ckpt"
        download "${rpaths[$c]}" "${sizes[$c]}" "$local_ckpt"
    else
        ok "Using local checkpoint $local_ckpt"
    fi
fi

if (( CLUSTER )); then
    log "Downloading $num_images validation images"
    rm -rf "$OUT/images" "$OUT/images_raw"; mkdir -p "$OUT/images" "$OUT/images_raw"
    kc exec "$POD" -- sh -c "cd '$VAL_DIR' && ls | awk -v n=$num_images '{ c[NR] = \$0 } END { if (n > NR) n = NR; for (i = 0; i < n; i++) print c[int(i * NR / n) + 1] }' \
        | while read -r cls; do echo \"\$cls/\$(ls \"\$cls\" | head -1)\"; done | tar -cf - -T -" | tar -xf - -C "$OUT/images_raw"
    for f in "$OUT"/images_raw/*/*; do mv "$f" "$OUT/images/$(basename "$(dirname "$f")")_$(basename "$f")"; done
    rm -rf "$OUT/images_raw"
    ok "$(ls "$OUT/images" | wc -l | tr -d ' ') images in $OUT/images"
    kc delete pod "$POD" --wait=false >/dev/null 2>&1 || true
    CLUSTER=0
else
    cached=$(ls "$OUT/images" 2>/dev/null | wc -l | tr -d ' ')
    (( cached > 0 )) || die "No cached test images in $OUT/images and the cluster is not reachable"
    (( cached != num_images )) && warn "Offline: using the $cached cached images from the last run"
    ok "Using $cached cached images in $OUT/images"
fi

# ---------------------------------------------------------------------------- 4. split
if (( need_split )); then
    log "Splitting checkpoint into encoder/decoder"
    "$PY" export_split.py --ckpt "$local_ckpt" --config "${EXPORT_CONFIG:-configs/imagenet.yaml}" --out-dir "$exported"
fi

# ---------------------------------------------------------------------------- 5. TX -> RX
txrx_batch=${TXRX_BATCH:-1}  # 1 = each image is sent as soon as it is encoded (streaming, per-image progress)
total_imgs=$(ls "$OUT/images" | wc -l | tr -d ' ')
log "Running RX and TX over TCP (localhost), $total_imgs images, batch $txrx_batch"
rm -rf "$OUT/received"
port=$("$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$PY" -u rx.py --decoder "$exported/decoder.pt" --listen "$port" --out-dir "$OUT/received" > "$OUT/rx.log" 2>&1 &
rx_pid=$!
for _ in $(seq 1 240); do
    grep -q "Listening" "$OUT/rx.log" 2>/dev/null && break
    kill -0 "$rx_pid" 2>/dev/null || { cat "$OUT/rx.log" >&2; die "RX failed to start"; }
    sleep 0.25
done
"$PY" -u tx.py --encoder "$exported/encoder.pt" --images "$OUT/images" --batch-size "$txrx_batch" \
    --send "127.0.0.1:$port" > "$OUT/tx.log" 2>&1 &
tx_pid=$!

bar() {  # $1 = done, $2 = total, $3 = width
    local fill=$(( $2 > 0 ? $1 * $3 / $2 : 0 ))
    printf '%s' "$GRN"; (( fill > 0 )) && printf '█%.0s' $(seq 1 "$fill")  # BSD seq counts down for "seq 1 0"
    printf '%s' "$D";   (( $3 - fill > 0 )) && printf '░%.0s' $(seq 1 $(( $3 - fill )))
    printf '%s' "$N"
}
frames=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏); t_start=$SECONDS; k=0
while kill -0 "$tx_pid" 2>/dev/null || kill -0 "$rx_pid" 2>/dev/null; do
    if [[ -t 1 ]]; then
        sent=$(awk '/^\[TX\] [0-9]+ images/ { s += $2 } END { print s + 0 }' "$OUT/tx.log" 2>/dev/null)
        recv=$(awk '/^\[RX\] Decoded/ { gsub(/[()]/, ""); r = $NF } END { print r + 0 }' "$OUT/rx.log" 2>/dev/null)
        if (( sent == 0 )); then tx_txt="loading encoder"; else tx_txt="$sent/$total_imgs sent"; fi
        if (( recv == 0 )); then rx_txt="waiting"; else rx_txt="$recv/$total_imgs decoded"; fi
        printf '\r  %s%s%s TX %s %-16s RX %s %-16s %s%ds%s\e[K' "$CYN" "${frames[k++ % 10]}" "$N" \
            "$(bar "$sent" "$total_imgs" 20)" "$tx_txt" "$(bar "$recv" "$total_imgs" 20)" "$rx_txt" "$D" $(( SECONDS - t_start )) "$N"
    fi
    sleep 0.2
done
[[ -t 1 ]] && printf '\r\e[K'
wait "$tx_pid" || { cat "$OUT/tx.log" >&2; die "TX failed"; }
wait "$rx_pid" || { cat "$OUT/rx.log" >&2; die "RX failed"; }
ok "$total_imgs images sent and decoded in $(( SECONDS - t_start ))s (including model loading)"

# ---------------------------------------------------------------------------- 6. report
log "Results"
"$PY" - "$OUT" <<'EOF'
import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from utils.crop import CenterCropTransform

out = Path(sys.argv[1])
originals = sorted((out / "images").iterdir())
received = sorted((out / "received").glob("*.png"))
prep = transforms.Compose([CenterCropTransform(256), transforms.ToTensor()])
pairs, psnrs = [], []
for src, rec in zip(originals, received):
    a = prep(Image.open(src).convert("RGB"))
    b = transforms.ToTensor()(Image.open(rec).convert("RGB"))
    psnr = 10 * torch.log10(1.0 / torch.mean((a - b) ** 2)).item()
    psnrs.append(psnr)
    pairs += [a, b]
    print(f"    {src.name:<40} PSNR {psnr:5.2f} dB")
print(f"    mean PSNR over {len(psnrs)} images: {sum(psnrs) / len(psnrs):.2f} dB")
save_image(make_grid(pairs, nrow=2, padding=4, pad_value=1.0), out / "comparison.png")
print(f"    side-by-side (original | received): {out / 'comparison.png'}")
EOF

log "Profiling (wall clock on this machine; 'steady' excludes the first, warm-up batch)"
"$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
prof = {}
for log in ("tx.log", "rx.log"):
    for line in open(out / log):
        if "-PROFILE] " in line:
            p = json.loads(line.split("-PROFILE] ", 1)[1])
            prof[p["role"]] = p
tx, rx = prof.get("tx", {}), prof.get("rx", {})

def row(label, key_tx, key_rx=None, unit=""):
    cells = [tx.get(key_tx), rx.get(key_rx or key_tx)]
    if all(c is None for c in cells):
        return
    cells = [f"{c}{unit}" if c is not None else "-" for c in cells]
    print(f"    {label:<32} {cells[0]:>14} {cells[1]:>14}")

print(f"    {'':<32} {'TX (encoder)':>14} {'RX (decoder)':>14}")
row("device", "device")
row("model parameters", "model_params_m", unit=" M")
row("model weights in memory", "model_memory_mb", unit=" MB")
row("model load time", "load_total_s", unit=" s")
row("encode / decode per image", "encode_ms_per_item", "decode_ms_per_item", " ms")
row("  steady state per image", "encode_steady_ms_per_item", "decode_steady_ms_per_item", " ms")
row("  first batch per image", "encode_first_batch_ms_per_item", "decode_first_batch_ms_per_item", " ms")
row("(de)serialize per image", "serialize_ms_per_item", "deserialize_ms_per_item", " ms")
row("payload per image", "payload_bytes_per_image", unit=" B")
row("peak memory after model load", "peak_rss_after_load_mb", unit=" MB")
row("peak memory (whole run)", "peak_rss_mb", unit=" MB")
row("peak GPU memory (CUDA)", "peak_gpu_mb", unit=" MB")
row("GPU memory held (MPS)", "gpu_memory_mb", unit=" MB")
json.dump(prof, open(out / "profile.json", "w"), indent=1)
print(f"    raw numbers: {out / 'profile.json'}")
EOF

# ---------------------------------------------------------------------------- 7. channel scenarios (optional)
n_scen=$(grep -cE '^  [a-z_]+:$' configs/channels.yaml || true)
printf '\n'
read -rp "  Run the $n_scen wireless channel scenarios (configs/channels.yaml)? [Y/n] " run_channel || run_channel=n
if [[ -z "$run_channel" || "$run_channel" =~ ^[Yy] ]]; then
    read -rp "  Channel draws per image per scenario [100]: " draws || draws=""
    draws=${draws:-100}
    [[ "$draws" =~ ^[0-9]+$ ]] && (( draws >= 1 )) || die "Invalid number: $draws"
    n_img=$(ls "$OUT/images" | wc -l | tr -d ' ')
    log "Channel scenarios: $n_scen scenarios × $n_img images × $draws draws = $(( n_scen * n_img * draws )) transmissions"
    "$PY" channel_experiments.py --exported "$exported" --images "$OUT/images" --draws "$draws" --out "$OUT/channel"
    ok "Report: $OUT/channel/report.md  (figures in $OUT/channel/figures, all draws in $OUT/channel/draws.csv)"
fi
