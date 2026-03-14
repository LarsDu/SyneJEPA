# SyneJEPA: Audio leJEPA for Embeddings → Visualization & Voice Changing

## Context

We're building an audio SSL model based on leJEPA (a simplified JEPA using SIGReg regularization instead of teacher-student/EMA/stop-gradient heuristics). The model learns audio embeddings from mel spectrograms. Proximate goal: real-time audio visualizer (embeddings → colors). Downstream goal: voice changing.

leJEPA's core insight: regularize embeddings toward an isotropic Gaussian distribution via SIGReg (~50 lines of code), combined with an invariance loss between augmented views. No teacher-student, no EMA, single hyperparameter λ.

## References

- **JEPA Architecture Overview**: https://www.startpinch.com/research/en/jepa-encoder-translation/ — Context encoder, target encoder, predictor; predicts representations of masked regions not raw inputs
- **LeJEPA Paper**: https://arxiv.org/pdf/2511.08544 — SIGReg regularization, isotropic Gaussian as optimal embedding distribution, 79% ImageNet ViT-H/14, works across 60+ architectures
- **LeJEPA Repository**: https://github.com/galilai-group/lejepa — Reference implementation with SIGReg loss and statistical tests
- **LeJEPA Minimal Example**: https://raw.githubusercontent.com/galilai-group/lejepa/refs/heads/main/MINIMAL.md — ~130-line complete training loop
- **Audio-JEPA**: https://arxiv.org/abs/2507.02915 — Random patch masking on spectrograms, no extra augmentation, <1/5 training data of wav2vec 2.0
- **I-JEPA**: https://arxiv.org/abs/2301.08243 — Block masking with multi-target prediction for images

---

## Datasets

| Stage | Dataset | Size | Rationale |
|-------|---------|------|-----------|
| **Prototyping** | FMA small | 7.2 GB (8k tracks × 30s) | Fast iteration, diverse spectrograms |
| **Main training** | FMA medium + LibriSpeech-clean-100 | ~80 GB | Tonal diversity + voice quality |
| **Eval** | ESC-50 (50 classes, 2k clips) | Small | Standard audio SSL benchmark |
| **Eval** | UrbanSound8K | Small | Complementary classification eval |
| **Eval** | Speech Commands v2 | Small | Keyword spotting benchmark |
| **Eval** | GTZAN (10 genres, 1k tracks) | Small | Music genre classification |

All available via HuggingFace `datasets` library.

---

## Augmentation & Masking Strategy (Detailed)

### Masking (Primary Learning Signal)

Based on Audio-JEPA findings, **masking is the primary augmentation** — additional data augmentation is minimal or absent.

**Approach: Random Patch Masking**
- Masking ratio: **ρ ~ U(0.4, 0.6)** per sample (averaging ~50%)
- Patches are masked **randomly** (not block masking). Audio-JEPA showed random masking outperforms block masking for audio since audio events span wide frequency bands.
- **Context encoder** sees only unmasked patches
- **Predictor** predicts target encoder representations at masked positions
- Target encoder sees **all** patches (full spectrogram)

**Why not block masking (I-JEPA style)?**
I-JEPA uses 4 target blocks at 15-20% image area each with aspect ratios [0.75, 1.5]. This works for images where objects are spatially localized. Audio events span full frequency bands, making random masking more appropriate.

### Data Augmentation (Minimal)

Audio-JEPA's key finding: **best results with NO extra augmentation beyond masking**. We start with masking-only and add augmentation only if needed:

**Baseline (masking only):**
- Random 4s crop from source audio (positional diversity)
- Random patch masking at 40-60% ratio
- No SpecAugment, no noise injection, no pitch shift

**Optional augmentations (add only if baseline underperforms):**
- Mild frequency masking: F=27, 1 mask (SpecAugment-style)
- Mild time masking: T=40, 1 mask
- Light Gaussian noise: SNR 25-30dB

**Augmentations to AVOID:**
- Aggressive pitch shifting — destroys phonetic content
- Time warping — breaks temporal relationships
- Heavy mixing/background addition — confuses target semantics
- Extreme volume normalization — removes intensity cues
- Aggressive frequency masking (>27 channels) — destroys frequency structure

### View Generation for Invariance Loss

Since leJEPA uses invariance loss (not just predictive masking), we need two views:
- **View 1**: Random 4s crop, masking pattern A
- **View 2**: Overlapping 4s crop (shifted ≤0.5s), masking pattern B
- Both views are of the **same audio segment** with different random masks
- The invariance loss operates on the CLS token / global representation (not patch-level)

---

## Evaluation Strategy (Detailed)

### Comparison Models

| Model | Type | Params | Pretraining Data | Notes |
|-------|------|--------|------------------|-------|
| **Whisper (Small)** | Supervised | 244M | 680k hrs | Feature extractor (encoder_last_hidden_state, dim=512) |
| **Whisper (Tiny)** | Supervised | 39M | 680k hrs | Closer param count to our ViT-S |
| **wav2vec 2.0 Base** | SSL | 95M | LibriSpeech 960h | Standard SSL baseline |
| **HuBERT Base** | SSL | 95M | LibriSpeech 960h | Strong SSL baseline |
| **SyneJEPA (ours)** | SSL | ~22M | FMA + LibriSpeech 100h | ViT-Small |

### Evaluation Methods

We use **three complementary evaluation methods** to ensure fair comparison:

1. **Linear probe (frozen)**: Single linear layer on frozen CLS embeddings, 100 epochs, AdamW
2. **k-NN (k=5, 20)**: Cosine similarity on frozen embeddings, no training required — purest measure of representation quality
3. **Prototypical probe**: Learns class-wise prototypes, better than linear for multi-label tasks (up to 37pp improvement over linear probe per recent research)

### Eval Tasks

| Task | Dataset | Metric | What It Tests |
|------|---------|--------|---------------|
| Environmental sound classification | ESC-50 (5-fold CV) | Accuracy | General audio understanding |
| Urban sound classification | UrbanSound8K | Accuracy | Environmental sound robustness |
| Keyword spotting | Speech Commands v2 | Accuracy | Fine-grained speech discrimination |
| Music genre classification | GTZAN | Accuracy | Tonal/rhythmic understanding |
| NN audio retrieval | ESC-50 | Recall@5 | Embedding space quality |

### Embedding Quality Metrics (Internal, Not Comparative)

- **Isotropy score**: min/max eigenvalue ratio of embedding covariance (→1.0 is ideal)
- **Effective rank**: exp(entropy(normalized eigenvalues)), should approach embedding dim
- **Perceptual correlation**: Spearman correlation between PCA dims and pitch/energy/spectral centroid
- **Temporal smoothness**: embedding trajectory jitter on continuous audio (L2 between consecutive frames)
- **Collapse detection**: monitor during training, alert if effective rank < 50% of dim

### Layer-wise Analysis

Following best practices from SSL literature:
- Train separate linear probes per encoder layer
- Identify which layers capture content vs. speaker vs. acoustic features
- Compare layer information distribution to Whisper/HuBERT

### Reporting

- Report all results per-domain: speech, music, environmental
- Report parameter counts and pretraining data volume alongside accuracy
- Compute "efficiency" = accuracy / log(params × pretraining_hours) for fair scaling comparison

---

## MVP Implementation Plan

### Monorepo Structure

```
SyneJEPA/
├── synejepa/                          # PyTorch training package
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── configs/
│   │   ├── pretrain/
│   │   │   ├── fma_small.yaml         # Local 4070 Ti config
│   │   │   └── fma_medium.yaml        # Vertex AI DDP config
│   │   └── eval/
│   │       └── esc50.yaml
│   ├── src/
│   │   └── synejepa/
│   │       ├── __init__.py
│   │       ├── audio/
│   │       │   ├── __init__.py
│   │       │   ├── preprocessing.py   # wav → mel spectrogram (Single Responsibility)
│   │       │   ├── dataset.py         # HF dataset loading + view generation
│   │       │   ├── masking.py         # Random patch masking strategy
│   │       │   └── augmentations.py   # Optional SpecAugment-style augmentations
│   │       ├── model/
│   │       │   ├── __init__.py
│   │       │   ├── encoder.py         # Audio ViT encoder (Interface Segregation)
│   │       │   ├── projection.py      # MLP projection head
│   │       │   └── synejepa.py        # Composes encoder + projector (Dependency Inversion)
│   │       ├── loss/
│   │       │   ├── __init__.py
│   │       │   ├── sigreg.py          # SIGReg loss (Single Responsibility)
│   │       │   ├── invariance.py      # Invariance loss (Single Responsibility)
│   │       │   └── composite.py       # Combines losses with weighting (Open/Closed)
│   │       ├── eval/
│   │       │   ├── __init__.py
│   │       │   ├── linear_probe.py    # Linear probe evaluator
│   │       │   ├── knn.py             # k-NN evaluator
│   │       │   ├── metrics.py         # Isotropy, collapse, retrieval
│   │       │   └── perceptual.py      # Pitch/energy/timbre correlation
│   │       └── training/
│   │           ├── __init__.py
│   │           └── trainer.py         # Training loop (Dependency Inversion)
│   ├── scripts/
│   │   ├── train.py                   # Hydra entrypoint for training
│   │   └── evaluate.py               # Hydra entrypoint for evaluation
│   └── tests/
│       ├── test_preprocessing.py
│       ├── test_masking.py
│       ├── test_model.py
│       └── test_sigreg.py
├── terraform/                         # GCloud Vertex AI infrastructure
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── terraform.tfvars.example
└── docs/
    └── plans/
        └── visualizer_proposal.md
```

### SOLID Principles Applied

- **Single Responsibility**: Masking separated from augmentation; each loss in its own module; preprocessing doesn't know about datasets
- **Open/Closed**: `CompositeLoss` accepts any loss components; new augmentations can be added without modifying existing ones
- **Liskov Substitution**: Encoder interface can be swapped (ViT-Tiny, ViT-Small, ViT-Base) without changing training code
- **Interface Segregation**: Encoder only encodes; projection only projects; no god objects
- **Dependency Inversion**: Trainer depends on abstract interfaces (model, loss, dataloader), not concrete implementations; Hydra configs wire concrete classes

### Technical Decisions

- **Sample rate**: 16kHz (downmix from source)
- **Spectrogram**: 128 mels, n_fft=1024, hop_length=160 → shape (128, 400) for 4s window
- **Patches**: 16×16 → 8×25 = 200 patches per spectrogram
- **ViT**: Small (dim=384, depth=12, heads=6, ~22M params)
- **Projection**: 384 → 2048 → 256 (BN + GELU)
- **Masking**: Random 40-60% of patches, no additional augmentation by default
- **Config**: Hydra for all configuration and entrypoints

### Phase 1: Audio Pipeline + Masking + Loss

**`synejepa/audio/preprocessing.py`** — `AudioProcessor`: load via torchaudio, resample to 16kHz, mono, MelSpectrogram, log-scale, normalize per-clip

**`synejepa/audio/masking.py`** — `RandomPatchMasker`: given spectrogram shape and patch size, generates random binary mask at ratio ρ ~ U(0.4, 0.6). Returns context_indices and target_indices.

**`synejepa/audio/dataset.py`** — Load FMA small via HuggingFace, random 4s crop, generate two views with different masks

**`synejepa/loss/sigreg.py`** — Use `lejepa` package or reimplement (~50 lines): random projections → sort → compare to Gaussian quantiles → L2

**`synejepa/loss/invariance.py`** — Cosine: `2 - 2 * (normalize(z1) · normalize(z2)).mean()`

**`synejepa/loss/composite.py`** — `CompositeLoss`: `loss = invariance(z1, z2) + λ * sigreg(z1, z2)`

### Phase 2: Model

**`synejepa/model/encoder.py`** — `timm` ViT with `img_size=(128,400), patch_size=16, in_chans=1`. CLS token for global representation. Accepts mask to process only unmasked patches.

**`synejepa/model/projection.py`** — 2-layer MLP: 384→2048 (BN+GELU) →256

**`synejepa/model/synejepa.py`** — Composes encoder + projector. Forward: encode both views, project, return (z1, z2).

### Phase 3: Training

**`synejepa/training/trainer.py`** — Training loop with mixed precision (bfloat16), gradient accumulation, checkpoint saving, metric logging to stdout/file.

**`scripts/train.py`** — Hydra entrypoint. Config: AdamW, lr=1e-4, cosine schedule, 10-epoch warmup, batch_size=64, 200 epochs.

**`configs/pretrain/fma_small.yaml`** — Full Hydra config for FMA small pretraining.

### Phase 4: Evaluation

**`scripts/evaluate.py`** — Hydra entrypoint. Loads checkpoint, runs linear probe + k-NN on ESC-50, computes embedding metrics.

**`synejepa/eval/linear_probe.py`** — Freeze encoder, train linear head (5-fold CV on ESC-50)

**`synejepa/eval/knn.py`** — k-NN evaluation on frozen embeddings

**`synejepa/eval/metrics.py`** — Isotropy, effective rank, retrieval

### Dependencies

```
torch>=2.1, torchaudio, timm, datasets, lejepa, scikit-learn, hydra-core, tensorboard
```

### Logging

- **TensorBoard** for visualizations: loss curves, embedding projections, spectrogram samples
- **Structured JSON file logging** for metrics: per-epoch stats, eval results, config snapshots
- Hydra's built-in run directory structure organizes outputs per experiment
- On Vertex AI, use `google_vertex_ai_tensorboard` resource for managed TensorBoard

---

---

## Training Infrastructure

### Local Training (RTX 4070 Ti, 12GB VRAM)

**Config (`configs/pretrain/fma_small.yaml`):**
- Batch size: 8 per device
- Gradient accumulation: 16 steps → effective batch 128
- Mixed precision: bfloat16 (no loss scaling needed)
- Gradient checkpointing: enabled (~60% memory savings, ~25% slower)
- Estimated throughput: 200-400 samples/sec
- Training time: ~200 epochs on FMA small ≈ 2-3 days

**Key optimizations for 12GB:**
- Gradient checkpointing on ViT encoder
- bfloat16 for all forward/backward (optimizer states stay fp32)
- Pre-computed cached spectrograms (`.pt` files) to reduce I/O

### Cloud Training (Vertex AI DDP)

**Recommended GPU:** 4× A100 40GB (~$6-10/hr total)

**Config (`configs/pretrain/fma_medium.yaml`):**
- Batch size: 32 per device × 4 GPUs = 128 per step
- Gradient accumulation: 1 (no accumulation needed)
- No gradient checkpointing (ample VRAM)
- NCCL backend for DDP
- Estimated throughput: 800-1200 samples/sec

**DDP-aware training code:**
- `scripts/train.py` detects `WORLD_SIZE` / `LOCAL_RANK` env vars (set by Vertex AI)
- Calls `torch.distributed.init_process_group(backend="nccl")`
- Wraps model in `DistributedDataParallel`
- Uses `DistributedSampler` for data loading
- Checkpoints saved to GCS bucket (`gs://...`)
- Same training code works local (single GPU) and distributed (DDP)

**Dockerfile (`synejepa/Dockerfile`):**
- Base: `nvcr.io/nvidia/pytorch:24.01-py3` (CUDA 12, PyTorch 2.x)
- Install project dependencies
- Entrypoint: `python scripts/train.py`

### Terraform (`terraform/`)

**Resources provisioned:**

| Resource | Purpose |
|----------|---------|
| `google_service_account` | Vertex AI training SA with least-privilege |
| `google_project_iam_member` | Roles: `aiplatform.user`, `storage.admin`, `artifactregistry.writer` |
| `google_storage_bucket` | Checkpoint storage with versioning + 30-day lifecycle |
| `google_artifact_registry_repository` | Docker container registry for training images |

**Usage:**
```bash
cd terraform/
terraform init && terraform plan && terraform apply

# Build & push container
docker build -t $(terraform output -raw artifact_registry_url)/synejepa:latest ../synejepa/
docker push $(terraform output -raw artifact_registry_url)/synejepa:latest

# Submit training job via gcloud CLI
gcloud ai custom-jobs create --region=us-central1 --config=job_config.yaml
```

**`terraform.tfvars.example`:**
```hcl
project_id = "your-gcp-project"
region     = "us-central1"
gpu_type   = "NVIDIA_TESLA_A100"
gpu_count  = 4
```

### Training Scale Summary

| Setting | Local (4070 Ti) | Cloud (4×A100) |
|---------|----------------|----------------|
| VRAM | 12GB | 4×40GB |
| Effective batch | 128 (via accum) | 128 (native) |
| Throughput | 200-400 s/s | 800-1200 s/s |
| Dataset | FMA small | FMA medium + LibriSpeech |
| Training time | 2-3 days | 8-12 hours |
| Cost | Electricity | ~$60-120 |

---

## Visualizer (Separate Proposal)

The real-time audio visualizer (embeddings → colors) will be designed separately in `docs/plans/visualizer_proposal.md`. It is not part of the MVP training/eval pipeline.

---

## Verification Plan

1. **Audio pipeline**: Load batch, visualize spectrograms, confirm shapes (B, 1, 128, 400)
2. **Masking**: Visualize masked vs unmasked patches on a spectrogram, confirm ratio ~50%
3. **Model**: Forward pass on dummy input, confirm output shape (B, 256)
4. **SIGReg**: Unit test that loss decreases when embeddings approach Gaussian
5. **Training**: 5-epoch smoke test — loss decreases, embedding std ~1.0, effective rank stable
6. **Eval**: Linear probe + k-NN on ESC-50 produce above-random accuracy
7. **Comparison**: Run same eval protocol on Whisper-tiny frozen features to establish baseline

---

## Risks & Mitigations

- **Batch size for SIGReg**: needs ≥128 effective; use gradient accumulation if GPU-limited
- **FMA I/O bottleneck**: pre-compute and cache spectrograms as `.pt` files
- **Embedding collapse**: monitor effective rank; increase λ if rank drops below 50% of dim
- **leJEPA was designed for vision**: masking-only approach validated by Audio-JEPA, but we combine leJEPA's SIGReg with Audio-JEPA's masking — this combination is novel and may need tuning
