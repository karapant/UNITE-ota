#!/usr/bin/env bash
# Train UNITE on a Kubernetes (Kueue) GPU cluster. All settings come from .env (see .env.template).
#
#   ./train_cluster.sh [all] [local|remote]   build + push image, prepare dataset/weights, submit training
#   ./train_cluster.sh build [local|remote]   build the image and push it to the registry:
#                                 local (default) = docker on this machine,
#                                 remote = the cluster's BuildKit servers (progress is streamed here)
#   ./train_cluster.sh prepare    check the cluster for ImageNet + FID weights, download what is missing
#   ./train_cluster.sh train      submit the training job (auto-resumes from the latest checkpoint)
#   ./train_cluster.sh status     show jobs / pods / Kueue workloads
#   ./train_cluster.sh logs       follow the training logs
#   ./train_cluster.sh stop       delete the training job (checkpoints are kept)
#   ./train_cluster.sh manifest [train|prepare]   print a job manifest without submitting
#
# Any .env value can be overridden per run, e.g.: NUM_GPUS=1 EXPERIMENT_NAME=smoke ./train_cluster.sh train
#
# The image contains the code (tracked files of the working tree, without docs), all Python
# dependencies, and the LPIPS/VGG16 weights.
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] || { echo "Missing .env: cp .env.template .env and fill it in." >&2; exit 1; }
# Variables set on the command line (e.g. NUM_GPUS=1 ./train_cluster.sh train) take precedence over .env
overrides=$(env | grep -E "^($(grep -oE '^[A-Z_][A-Z0-9_]*=' .env | tr -d '=' | paste -sd'|' -))=" || true)
set -a; source .env
while IFS= read -r kv; do [[ -n "$kv" ]] && export "$kv"; done <<< "$overrides"
set +a

for v in KUBE_CONTEXT K8S_NAMESPACE KUEUE_QUEUE GPU_PRODUCT REGISTRY REGISTRY_PROJECT IMAGE_NAME BASE_IMAGE \
         DATA_PATH INCEPTION_WEIGHTS IN256_FID_STATS RESULTS_DIR JOB_NAME EXPERIMENT_NAME CONFIG \
         NUM_GPUS CPU_PER_GPU MEM_PER_GPU_GI SHM_SIZE_GI TOTAL_BATCH_SIZE PER_GPU_BATCH; do
    [[ -n "${!v:-}" ]] || { echo "Missing $v in .env" >&2; exit 1; }
done

# Files baked into the image: tracked files, minus docs and the cluster tooling itself.
IMAGE_FILES=(-- . ":(exclude)*.md" ":(exclude)assets" ":(exclude).env.template" ":(exclude)$(basename "$0")")
# Image tag = hash of the image contents (those files + the build() function below, i.e. Dockerfile and
# in-pod scripts), so docs or job-setting changes never require a rebuild.
image_inputs() {
    git ls-files -z "${IMAGE_FILES[@]}" | xargs -0 shasum
    sed -n '/^build() {/,/^build_local() {/p' "$0"
}
TAG="c-$(image_inputs | shasum | cut -c1-12)"
IMAGE="$REGISTRY/$REGISTRY_PROJECT/$IMAGE_NAME:$TAG"
SLUG=$(echo "$EXPERIMENT_NAME" | tr '[:upper:]_' '[:lower:]-' | tr -cd 'a-z0-9-' | cut -c1-30)
TRAIN_JOB="$JOB_NAME-train-$SLUG"
PREPARE_JOB="$JOB_NAME-prepare"

kc() { kubectl --context "$KUBE_CONTEXT" -n "$K8S_NAMESPACE" "$@"; }
log() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------------------- build
CLEANUP=()
trap 'for c in ${CLEANUP[@]+"${CLEANUP[@]}"}; do eval "$c"; done' EXIT

build() {  # $1 = local|remote
    local mode=${1:-local} ctx
    [[ $mode == local || $mode == remote ]] || { echo "Unknown build mode '$mode' (use local or remote)" >&2; exit 1; }
    ctx=$(mktemp -d)
    CLEANUP+=("rm -rf '$ctx'")
    git ls-files -z "${IMAGE_FILES[@]}" | tar --null -T - -cf - | tar -xf - -C "$ctx"
    mkdir -p "$ctx/.cluster"

    cat > "$ctx/.cluster/train.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace
mkdir -p "$RESULTS_DIR/$EXPERIMENT_NAME"
# The image filesystem is read-only on the cluster, and LPIPS takes a lock file next to the
# torchvision weights: expose the baked weights through a writable TORCH_HOME.
mkdir -p /tmp/torch/hub/checkpoints
ln -sf /opt/torch/hub/checkpoints/*.pth /tmp/torch/hub/checkpoints/
export TORCH_HOME=/tmp/torch
export TRAIN_DATA_DIR="$DATA_PATH" WANDB_DIR="$RESULTS_DIR/$EXPERIMENT_NAME"
[[ -n "${WANDB_API_KEY:-}" ]] || export WANDB_MODE=offline

accum=$(( TOTAL_BATCH_SIZE / (NUM_GPUS * PER_GPU_BATCH) ))
if (( accum < 1 )); then accum=1; fi
python - "$CONFIG" "$accum" <<'PY'
import os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
t = cfg["training"]
t["total_batch_size"] = int(os.environ["TOTAL_BATCH_SIZE"])
t["grad_accum_steps"] = int(sys.argv[2])
if os.environ.get("EPOCHS"):
    t["epochs"] = int(os.environ["EPOCHS"])
yaml.safe_dump(cfg, open("/tmp/config.yaml", "w"), sort_keys=False)
print("[train] config:", {k: t[k] for k in ("total_batch_size", "grad_accum_steps", "epochs")})
PY

ckpt="${RESUME_CKPT:-}"
if [[ -z "$ckpt" ]]; then
    ckpt=$(ls -1 "$RESULTS_DIR/$EXPERIMENT_NAME/checkpoints/"*.pt 2>/dev/null | sort | tail -n1 || true)
fi
echo "[train] resume from: ${ckpt:-none}"
nvidia-smi -L || true
exec torchrun --standalone --nproc_per_node="$NUM_GPUS" main_train.py \
    --config /tmp/config.yaml --data-path "$DATA_PATH" --results-dir "$RESULTS_DIR" \
    --experiment-name "$EXPERIMENT_NAME" --image-size 256 ${ckpt:+--ckpt "$ckpt"}
EOF

    cat > "$ctx/.cluster/prepare.py" <<'EOF'
"""Check for ImageNet (ImageFolder layout) and FID weights; download whatever is missing."""
import math, os, shutil, tarfile, urllib.request

REPO = "timm/imagenet-1k-wds"  # 1024 shuffled train shards, 64 val shards
FID_URLS = {
    "INCEPTION_WEIGHTS": "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth",
    "IN256_FID_STATS": "https://raw.githubusercontent.com/LTH14/JiT/main/fid_stats/jit_in256_stats.npz",
}
train_dir = os.environ["DATA_PATH"].rstrip("/")
val_dir = os.path.join(os.path.dirname(train_dir), "val")


def summary(d):
    classes = [e for e in os.scandir(d) if e.is_dir()] if os.path.isdir(d) else []
    return len(classes), sum(len(os.listdir(c.path)) for c in classes)


def ready(d):
    if not os.path.isdir(d):
        return False
    if os.path.exists(os.path.join(d, ".complete")):
        return True
    in_progress = any(f.startswith(".done-") for f in os.listdir(d))
    return not in_progress and summary(d)[0] >= 1000


def extract_shard(name, out_dir, token):
    marker = os.path.join(out_dir, f".done-{name}")
    if os.path.exists(marker):
        return
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, name, repo_type="dataset", token=token, cache_dir="/tmp/hf")
    samples = {}
    with tarfile.open(path) as tar:
        for m in tar:
            if m.isfile():
                key, ext = os.path.basename(m.name).split(".", 1)
                samples.setdefault(key, {})[ext.lower()] = tar.extractfile(m).read()
    for key, s in samples.items():
        img = s.get("jpg") or s.get("jpeg") or s.get("png")
        if img is None or "cls" not in s:
            continue
        cls_dir = os.path.join(out_dir, f"{int(s['cls']):04d}")
        os.makedirs(cls_dir, exist_ok=True)
        with open(os.path.join(cls_dir, f"{key}.jpg"), "wb") as f:
            f.write(img)
    os.remove(os.path.realpath(path))
    open(marker, "w").close()
    print(f"[prepare] {name}: {len(samples)} images")


def download_split(out_dir, shards):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("[prepare] HF_TOKEN not available (set HF_TOKEN_SECRET in .env)")
    os.makedirs(out_dir, exist_ok=True)
    for name in shards:
        extract_shard(name, out_dir, token)
    open(os.path.join(out_dir, ".complete"), "w").close()


if ready(train_dir):
    print(f"[prepare] ImageNet train found: {train_dir} (classes, images) = {summary(train_dir)}")
else:
    n = max(1, math.ceil(1024 * float(os.environ.get("SUBSET_FRACTION", "0.1"))))
    need_gb = n * 0.15 + 2
    existing = train_dir
    while not os.path.isdir(existing):
        existing = os.path.dirname(existing)
    free_gb = shutil.disk_usage(existing).free / 1e9
    print(f"[prepare] ImageNet train missing: downloading {n}/1024 shards (~{need_gb:.0f} GB, {free_gb:.0f} GB free)")
    if free_gb < need_gb:
        raise SystemExit("[prepare] Not enough free storage; lower SUBSET_FRACTION")
    download_split(train_dir, [f"imagenet1k-train-{i:04d}.tar" for i in range(n)])
    print(f"[prepare] train ready: {summary(train_dir)}")

if ready(val_dir) or (os.path.isdir(val_dir) and summary(val_dir)[1] > 0
                      and not any(f.startswith(".done-") for f in os.listdir(val_dir))):
    print(f"[prepare] ImageNet val found: {val_dir} (classes, images) = {summary(val_dir)}")
else:
    v = int(os.environ.get("VAL_SHARDS", "8"))
    print(f"[prepare] ImageNet val missing: downloading {v}/64 shards")
    download_split(val_dir, [f"imagenet1k-validation-{i:02d}.tar" for i in range(v)])
    print(f"[prepare] val ready: {summary(val_dir)}")

for var, url in FID_URLS.items():
    path = os.environ[var]
    if os.path.exists(path):
        print(f"[prepare] {var} found: {path}")
        continue
    print(f"[prepare] {var} missing: downloading {url}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path + ".part")
    os.replace(path + ".part", path)

os.makedirs(os.environ["RESULTS_DIR"], exist_ok=True)
print("[prepare] done")
EOF

    cat > "$ctx/Dockerfile" <<EOF
FROM $BASE_IMAGE
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore PIP_BREAK_SYSTEM_PACKAGES=1 \\
    TORCH_HOME=/opt/torch
WORKDIR /workspace
COPY requirements.txt .
# torch/torchvision come from the base image; keep them, install the rest
RUN grep -vE '^(torch|torchvision)([<>=!~ ]|\$)' requirements.txt > /tmp/requirements.txt \\
 && pip install -r /tmp/requirements.txt \\
 && python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
COPY . .
# Bake the LPIPS + torchvision VGG16 weights into the image
RUN python -c "from modules.autoencoding_utils.lpips import LPIPS; LPIPS()" \\
 && chmod -R a+rX /workspace /opt/torch
EOF

    if [[ $mode == remote ]]; then build_remote "$ctx"; else build_local "$ctx"; fi
    log "Image ready: $IMAGE"
}

build_local() {  # $1 = build context
    log "Building $IMAGE locally (linux/amd64)"
    docker build --platform linux/amd64 --progress=plain -t "$IMAGE" "$1"
    if [[ -n "${REGISTRY_USER:-}" && -n "${REGISTRY_PASSWORD:-}" ]]; then
        printf '%s' "$REGISTRY_PASSWORD" | docker login -u "$REGISTRY_USER" --password-stdin "$REGISTRY"
    fi
    log "Pushing $IMAGE"
    docker push "$IMAGE"
}

build_remote() {  # $1 = build context; builds on the cluster's BuildKit servers and pushes from there
    local pod="$JOB_NAME-build" push_secret="$JOB_NAME-registry-push" t0=$SECONDS
    local certs="${BUILDKIT_CERTS_SECRET:-buildkit-client-certs}"
    : "${BUILDKIT_ADDR:?Missing BUILDKIT_ADDR in .env}" "${BUILDKIT_IMAGE:?Missing BUILDKIT_IMAGE in .env}"
    kc get secret "$certs" >/dev/null 2>&1 || {
        echo "BuildKit client certificates (secret '$certs') not found in namespace $K8S_NAMESPACE." >&2
        echo "Ask the cluster admins to enable BuildKit for the namespace, or use: ./train_cluster.sh build local" >&2
        exit 1; }
    [[ -n "${REGISTRY_USER:-}" && -n "${REGISTRY_PASSWORD:-}" ]] || {
        echo "Remote builds push from the cluster: set REGISTRY_USER / REGISTRY_PASSWORD (push-enabled) in .env." >&2
        exit 1; }

    log "[1/4] Registry push credentials -> secret $push_secret"
    local auth; auth=$(printf '%s:%s' "$REGISTRY_USER" "$REGISTRY_PASSWORD" | base64 | tr -d '\n')
    kc create secret generic "$push_secret" \
        --from-literal=config.json="{\"auths\":{\"$REGISTRY\":{\"auth\":\"$auth\"}}}" \
        --dry-run=client -o yaml | kc apply -f - >/dev/null

    log "[2/4] Starting build client pod $pod ($BUILDKIT_IMAGE)"
    kc delete pod "$pod" --ignore-not-found --wait=true >/dev/null
    CLEANUP+=("kc delete pod '$pod' --ignore-not-found --wait=false >/dev/null 2>&1")
    kc apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $pod
  labels: {app: $JOB_NAME}
spec:
  restartPolicy: Never
  activeDeadlineSeconds: 7200
  securityContext: {runAsUser: 1000, runAsGroup: 1000}
  containers:
  - name: buildctl
    image: $BUILDKIT_IMAGE
    command: ["sh", "-c", "mkdir -p /tmp/context && sleep 7200"]
    env:
    - {name: DOCKER_CONFIG, value: /docker}
    resources:
      requests: {cpu: "1", memory: 1Gi}
      limits: {cpu: "2", memory: 2Gi}
    volumeMounts:
    - {name: tmp, mountPath: /tmp}
    - {name: certs, mountPath: /certs, readOnly: true}
    - {name: docker, mountPath: /docker, readOnly: true}
  volumes:
  - {name: tmp, emptyDir: {}}
  - {name: certs, secret: {secretName: $certs}}
  - {name: docker, secret: {secretName: $push_secret}}
EOF
    kc wait --for=condition=Ready "pod/$pod" --timeout=10m >/dev/null || {
        kc describe pod "$pod" | tail -20 >&2; exit 1; }

    log "[3/4] Uploading build context ($(du -sh "$1" | cut -f1))"
    tar -C "$1" -cf - . | kc exec -i "$pod" -- tar -xf - -C /tmp/context

    log "[4/4] Building and pushing $IMAGE on $BUILDKIT_ADDR (step-by-step progress below)"
    local cache="$REGISTRY/$REGISTRY_PROJECT/$IMAGE_NAME:buildcache"
    kc exec "$pod" -- buildctl --addr "$BUILDKIT_ADDR" --tlsdir /certs build --progress=plain \
        --frontend dockerfile.v0 --local context=/tmp/context --local dockerfile=/tmp/context \
        --opt platform=linux/amd64 \
        --import-cache "type=registry,ref=$cache" \
        --export-cache "type=registry,ref=$cache,mode=max" \
        --output "type=image,name=$IMAGE,push=true"
    log "Remote build finished in $(( (SECONDS - t0) / 60 ))m $(( (SECONDS - t0) % 60 ))s"
}

# ---------------------------------------------------------------------------- manifests
yaml_labels() {  # $1 = indent
    local kv; IFS=',' read -ra kv <<< "${POD_LABELS:-}"
    for p in "${kv[@]}"; do [[ -n "$p" ]] && printf '%s%s: "%s"\n' "$1" "${p%%=*}" "${p#*=}"; done
    return 0
}

yaml_tolerations() {
    local keys; IFS=',' read -ra keys <<< "${TOLERATIONS:-}"
    for k in "${keys[@]}"; do
        [[ -n "$k" ]] && printf '      - key: %s\n        operator: Exists\n        effect: NoSchedule\n' "$k"
    done
    return 0
}

yaml_env() {  # names of variables to pass through
    for v in "$@"; do printf '        - name: %s\n          value: "%s"\n' "$v" "${!v:-}"; done
}

job_manifest() {  # $1 = job name, $2 = prepare|train
    local name=$1 mode=$2 cmd resources scheduling secret_env=""
    if [[ $mode == prepare ]]; then
        cmd="python /workspace/.cluster/prepare.py"
        resources='          requests: {cpu: "4", memory: 16Gi}
          limits: {cpu: "4", memory: 16Gi}'
        scheduling=""
        if [[ -n "${HF_TOKEN_SECRET:-}" ]]; then
            secret_env="        - name: HF_TOKEN
          valueFrom: {secretKeyRef: {name: $HF_TOKEN_SECRET, key: HF_TOKEN, optional: true}}"
        fi
    else
        cmd="bash /workspace/.cluster/train.sh"
        local cpu=$(( NUM_GPUS * CPU_PER_GPU )) mem=$(( NUM_GPUS * MEM_PER_GPU_GI ))
        resources="          requests: {cpu: \"$cpu\", memory: ${mem}Gi, nvidia.com/gpu: \"$NUM_GPUS\"}
          limits: {cpu: \"$cpu\", memory: ${mem}Gi, nvidia.com/gpu: \"$NUM_GPUS\"}"
        scheduling="      nodeSelector:
        nvidia.com/gpu.product: $GPU_PRODUCT
      tolerations:
$(yaml_tolerations)"
        if [[ -n "${WANDB_API_KEY:-}" ]]; then
            secret_env="        - name: WANDB_API_KEY
          valueFrom: {secretKeyRef: {name: $JOB_NAME-wandb, key: WANDB_API_KEY}}"
        fi
    fi

    cat <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $name
  labels:
    app: $JOB_NAME
$( [[ $mode == train ]] && echo "    kueue.x-k8s.io/queue-name: $KUEUE_QUEUE" )
spec:
  backoffLimit: 3
  template:
    metadata:
      labels:
        app: $JOB_NAME
$(yaml_labels "        ")
    spec:
      restartPolicy: Never
      securityContext: {runAsUser: 1000, runAsGroup: 1000}
$( [[ -n "${IMAGE_PULL_SECRET:-}" ]] && printf '      imagePullSecrets:\n      - name: %s\n' "$IMAGE_PULL_SECRET" )
$scheduling
      containers:
      - name: $mode
        image: $IMAGE
        imagePullPolicy: IfNotPresent
        command: ["/bin/bash", "-c", "$cmd"]
        env:
        - {name: HOME, value: /tmp}
$(yaml_env DATA_PATH INCEPTION_WEIGHTS IN256_FID_STATS RESULTS_DIR SUBSET_FRACTION VAL_SHARDS \
           EXPERIMENT_NAME CONFIG NUM_GPUS TOTAL_BATCH_SIZE PER_GPU_BATCH EPOCHS RESUME_CKPT)
$secret_env
        resources:
$resources
        volumeMounts:
        - {name: tmp, mountPath: /tmp}
        - {name: shm, mountPath: /dev/shm}
      volumes:
      - {name: tmp, emptyDir: {}}
      - {name: shm, emptyDir: {medium: Memory, sizeLimit: ${SHM_SIZE_GI}Gi}}
EOF
}

# ---------------------------------------------------------------------------- cluster steps
pod_state() {  # $1 = job name -> e.g. "Pending", "ContainerCreating", "Running", "Error"
    local st
    st=$(kc get pods -l "job-name=$1" --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{range .items[*]}{.status.phase}{" "}{.status.containerStatuses[0].state.waiting.reason}{.status.containerStatuses[0].state.terminated.reason}{"\n"}{end}' 2>/dev/null | tail -n1)
    echo "${st:-Queued}"
}

wait_job() {  # $1 = job name; returns non-zero if the job failed
    local s f st last="" t0=$SECONDS
    while true; do
        s=$(kc get job "$1" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)
        f=$(kc get job "$1" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || true)
        [[ "$s" == 1 ]] && return 0
        [[ "$f" == True ]] && return 1
        st=$(pod_state "$1")
        [[ "$st" != "$last" ]] && printf '    [%3ds] %s\n' $(( SECONDS - t0 )) "$st" && last=$st
        sleep 5
    done
}

prepare() {
    log "Checking dataset and FID weights on the cluster (job $PREPARE_JOB)"
    kc delete job "$PREPARE_JOB" --ignore-not-found --wait=true >/dev/null
    job_manifest "$PREPARE_JOB" prepare | kc apply -f -
    local ok=0; wait_job "$PREPARE_JOB" || ok=$?
    kc logs "job/$PREPARE_JOB" --tail=-1 || true
    if (( ok != 0 )); then echo "Preparation failed (kubectl describe job $PREPARE_JOB)" >&2; exit 1; fi
    kc delete job "$PREPARE_JOB" --wait=false >/dev/null
    kc delete pods -l "job-name=$PREPARE_JOB" --wait=false >/dev/null 2>&1 || true
}

train() {
    if kc get job "$TRAIN_JOB" >/dev/null 2>&1; then
        echo "Job $TRAIN_JOB already exists. './train_cluster.sh stop' first, or change EXPERIMENT_NAME." >&2
        exit 1
    fi
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        kc create secret generic "$JOB_NAME-wandb" --from-literal=WANDB_API_KEY="$WANDB_API_KEY" \
            --dry-run=client -o yaml | kc apply -f - >/dev/null
    fi
    log "Submitting $TRAIN_JOB: $NUM_GPUS x $GPU_PRODUCT, image $IMAGE"
    job_manifest "$TRAIN_JOB" train | kc apply -f -
    cat <<EOF

Training submitted. Queued by Kueue until $NUM_GPUS x $GPU_PRODUCT are free.
  ./train_cluster.sh status   # job / pod / workload state
  ./train_cluster.sh logs     # follow training logs
  ./train_cluster.sh stop     # delete the job (checkpoints in $RESULTS_DIR/$EXPERIMENT_NAME are kept)
EOF
}

case "${1:-all}" in
    all)     build "${2:-local}"; prepare; train ;;
    local|remote) build "$1"; prepare; train ;;
    build)   build "${2:-local}" ;;
    prepare) prepare ;;
    train)   train ;;
    status)  kc get jobs,pods,workloads -l "app=$JOB_NAME" 2>/dev/null || kc get jobs,pods -l "app=$JOB_NAME" ;;
    logs)    kc logs -f "job/$TRAIN_JOB" ;;
    stop)    kc delete job "$TRAIN_JOB"; kc delete pods -l "job-name=$TRAIN_JOB" --wait=false >/dev/null 2>&1 || true ;;
    manifest) if [[ "${2:-train}" == prepare ]]; then job_manifest "$PREPARE_JOB" prepare; else job_manifest "$TRAIN_JOB" train; fi ;;
    *)       sed -n '2,/^[^#]/{/^#/p;}' "$0"; exit 1 ;;
esac
