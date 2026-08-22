# Phase 11 — GPU Runtime (Development)

## Development hardware

- NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)
- AMD Ryzen 7 7445HS
- 16 GB RAM
- Windows

## Recommended development configuration

```env
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=8
```

Batch size 8 is a conservative default for 6 GB VRAM. Increase only after benchmarking.

Batch sizes 16 and 32 are **not guaranteed** to work on the RTX 3050. Use benchmark mode first.

## Commands

From project root (`D:\code\office\chatbot`):

### CUDA smoke test (100 frozen chunks)

```powershell
$env:EMBEDDING_DEVICE="cuda"
$env:EMBEDDING_BATCH_SIZE="8"
python scripts/phase11_smoke_test.py
```

The CPU smoke test result remains valid. CUDA smoke uses the same 100 deterministic chunks.

### GPU benchmark (100 chunks, no production table writes)

```powershell
$env:EMBEDDING_DEVICE="cuda"
$env:EMBEDDING_BATCH_SIZE="8"
python scripts/phase11_embed.py --benchmark
```

Reports:
- `reports/phase11_gpu_benchmark.json`
- `reports/phase11_gpu_benchmark.txt`

### Full production embedding (only after smoke + benchmark pass)

```powershell
$env:EMBEDDING_DEVICE="cuda"
$env:EMBEDDING_BATCH_SIZE="8"
python scripts/phase11_embed.py
```

## Runtime behavior

- **Device**: `EMBEDDING_DEVICE=cuda` fails clearly if CUDA is unavailable. No silent CPU fallback during production runs.
- **Batching**: Configurable via `EMBEDDING_BATCH_SIZE`. On CUDA OOM, batch size halves (32→16→8→4→2→1) and the same batch retries.
- **Checkpoint/resume**: Chunks already persisted in PGVector are skipped via idempotent `(chunk_id, chunking_algorithm_version, embedding_input_manifest, embedding_model, embedding_model_revision)` identity.
- **NUL bytes**: PostgreSQL storage sanitizes NUL bytes only at DB insert. Qwen receives the exact frozen chunk content.

## Unchanged by this optimization

- Qwen/Qwen3-Embedding-0.6B model and pinned revision
- 1024-dimensional L2-normalized embeddings
- Frozen 5,904-chunk manifest
- PGVector schema
- Retrieval evaluation logic
