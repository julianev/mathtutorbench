# Proposer sweep: reasoning models on H100 (80 GB)

Task model fixed to `Qwen/Qwen3.5-4B` across all runs. Each command launches one SLURM job that serves the proposer on GPU 1 and the task model on GPU 0.

All commands below are **full-scale** (`auto medium`, paper split ≈ 400/400/350). To turn any of them into a smoke test, append:

```
--auto light --train_size 50 --dev_size 50 --test_size 50
```

## 1. Ministral 3 — 14B Reasoning  ✅ already done (smoke test)

```bash
bash optimize/run_optimize.sh \
    --model Qwen/Qwen3.5-4B \
    --prompt_model mistralai/Ministral-3-14B-Reasoning-2512 \
    --prompt_max_tokens 8192 \
    --auto medium
```

- ~28 GB bf16, dedicated reasoning SKU, thinks by default.

## 2. Qwen 3.5 — 27B dense (FP8)

```bash
bash optimize/run_optimize.sh \
    --model Qwen/Qwen3.5-4B \
    --prompt_model Qwen/Qwen3.5-27B-FP8 \
    --prompt_max_tokens 8192 \
    --auto medium
```

- ~27 GB FP8, dense, thinks by default (Qwen3.5 27B+).

## 3. Qwen 3.5 — 35B MoE (FP8)

```bash
bash optimize/run_optimize.sh \
    --model Qwen/Qwen3.5-4B \
    --prompt_model Qwen/Qwen3.5-35B-A3B-FP8 \
    --prompt_max_tokens 8192 \
    --auto medium
```

- ~35 GB FP8, MoE (3B active / 35B total), thinks by default.

## 4. Gemma 4 — 26B MoE  ⚠ requires `enable_thinking` wiring

```bash
bash optimize/run_optimize.sh \
    --model Qwen/Qwen3.5-4B \
    --prompt_model google/gemma-4-26B-A4B-it \
    --prompt_max_tokens 8192 \
    --auto medium
```

- ~52 GB bf16, MoE (3.8B active / 26B total).
- **TODO:** Gemma 4 thinking mode is opt-in. The proposer vllm server needs one extra flag:
  `--default-chat-template-kwargs '{"enable_thinking": true}'`
  This is not currently wired through `slurm/optimize_job.sh`. Without it the proposer runs in non-reasoning mode — which violates the "only reasoning models" constraint.

## 5. Gemma 4 — 31B dense  ⚠ requires `enable_thinking` wiring + tight VRAM

```bash
bash optimize/run_optimize.sh \
    --model Qwen/Qwen3.5-4B \
    --prompt_model google/gemma-4-31B-it \
    --prompt_max_tokens 8192 \
    --prompt_max_model_len 16384 \
    --auto medium
```

- ~62 GB bf16, dense — tight on 80 GB; lower `prompt_max_model_len` leaves KV-cache headroom.
- Same `enable_thinking` TODO as #4.

## Monitor & inspect

```bash
squeue -u $USER
tail -f logs/mtb-optimize-<JOB_ID>.out
ls -lt optimize/results/*.summary.json | head
```

Each summary JSON now includes `initial_instructions`, `optimized_instructions`, top-level `auto`, `auto_settings`, and `prompt_model` for comparison across runs.
