# SyneJEPA

Audio leJEPA model for self-supervised audio embeddings using SIGReg regularization.

SyneJEPA learns general-purpose audio representations from unlabeled audio by predicting
masked spectrogram patches in embedding space, regularized toward an isotropic Gaussian
distribution. The learned embeddings support downstream tasks like audio classification,
retrieval, and real-time audio visualization.

## Architecture

- **Encoder**: ViT-Small (dim=384, depth=12, heads=6, ~22M params) via [timm](https://github.com/huggingface/pytorch-image-models)
- **Input**: Mel spectrograms (128 mels × 400 frames from 4s audio at 16kHz)
- **Patches**: 16×16 → 200 patches per spectrogram
- **Masking**: Random patch masking at 40–60% (no block masking)
- **Loss**: Cosine invariance + SIGReg (isotropic Gaussian regularization)

## Quickstart

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU with CUDA support (for training)

### Install

```bash
cd synejepa/
uv sync --all-extras
```

### Run tests

```bash
uv run pytest tests/ -v
```

### Pre-commit hooks

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Training

### Local (RTX 4070 Ti / consumer GPU)

```bash
cd synejepa/

# Pre-compute spectrograms (recommended for I/O performance)
# uv run python scripts/precompute_cache.py --data-dir data/fma_small --cache-dir data/cache

# Train with default config (FMA small, batch=8, grad_accum=16)
uv run python scripts/train.py
```

Default config: `configs/pretrain/fma_small.yaml`
- Batch size: 8 (effective 128 via gradient accumulation)
- Gradient checkpointing enabled for 12GB VRAM
- bfloat16 mixed precision

Override config values via Hydra:

```bash
uv run python scripts/train.py training.lr=3e-4 training.epochs=100
```

### Cloud (GCP Vertex AI)

#### 1. Provision infrastructure

```bash
cd terraform/
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your GCP project ID

terraform init
terraform plan
terraform apply
```

This creates:
- Service account with least-privilege IAM roles
- GCS bucket for checkpoints (versioned, 30-day lifecycle)
- Artifact Registry for Docker images

#### 2. Build and push container

```bash
cd synejepa/

# Build
docker build -t $(cd ../terraform && terraform output -raw artifact_registry_url)/synejepa:latest .

# Push
docker push $(cd ../terraform && terraform output -raw artifact_registry_url)/synejepa:latest
```

#### 3. Submit training job

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="synejepa-fma-medium" \
  --worker-pool-spec=machine-type=a2-highgpu-4g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=4,replica-count=1,container-image-uri=$(cd terraform && terraform output -raw artifact_registry_url)/synejepa:latest \
  --args="--config-name=fma_medium"
```

Cloud config: `configs/pretrain/fma_medium.yaml`
- 4× A100 40GB, DDP with NCCL backend
- Batch size: 32 per GPU (128 total, no accumulation needed)
- Checkpoints saved to GCS

## Monitoring with TensorBoard

### Local training

```bash
# In a separate terminal while training runs
uv run tensorboard --logdir synejepa/tb_logs/
# Open http://localhost:6006
```

Tracks: loss curves (total, invariance, SIGReg), learning rate schedule, embedding statistics.

Structured JSON metrics are also written to `tb_logs/<run>/metrics.jsonl` for programmatic access.

### GCP training

Option 1 — Stream logs from GCS:

```bash
tensorboard --logdir gs://YOUR_PROJECT-synejepa-checkpoints/tb_logs/
```

Option 2 — Vertex AI managed TensorBoard:

```bash
# Create a Vertex AI TensorBoard instance (one-time)
gcloud ai tensorboards create --display-name="synejepa" --region=us-central1

# Upload logs after training
gcloud ai tensorboards experiments upload-tb-logs \
  --tensorboard=TENSORBOARD_ID \
  --experiment=synejepa-run-1 \
  --logdir=gs://YOUR_PROJECT-synejepa-checkpoints/tb_logs/
```

Then view at: https://console.cloud.google.com/vertex-ai/experiments/tensorboard

## Evaluation

```bash
uv run python scripts/evaluate.py eval.checkpoint_path=checkpoints/fma_small/checkpoint_epoch_0199.pt
```

Runs:
- Linear probe on ESC-50 (5-fold CV)
- k-NN classification (k=5, 20)
- Embedding quality metrics (isotropy, effective rank, retrieval recall)

## Project structure

```
SyneJEPA/
├── synejepa/           # PyTorch training package
│   ├── src/synejepa/   # Source code
│   ├── configs/        # Hydra YAML configs
│   ├── scripts/        # Training & eval entrypoints
│   ├── tests/          # Unit tests
│   └── Dockerfile
├── terraform/          # GCloud Vertex AI infrastructure
└── docs/plans/         # Design documents
```

## References

- [LeJEPA paper](https://arxiv.org/abs/2511.08544) — SIGReg regularization
- [LeJEPA repo](https://github.com/galilai-group/lejepa) — Reference implementation
- [JEPA overview](https://www.startpinch.com/research/en/jepa-encoder-translation/) — Architecture explainer
