# MathTutorBench: Benchmark Setup, Multi-Model Evaluation & Prompt Optimization

## Branch Status

| Branch | Status |
|--------|--------|
| `setup_benchmark` | Code written (Parts 1–3 scripts, env install). **SLURM smoke tests (steps 8–10) not yet run.** Part 4 (multi-model) not started. |
| `prompt_optimization` | Branched from `setup_benchmark`. Adds Part 5 (DSPy prompt optimization). |

## Context

MathTutorBench evaluates LLMs as math tutors across 9 tasks (problem solving, socratic questioning, mistake location/correction/solution correctness, scaffolding generation, pedagogy following, plus hard variants of the last two). The project has two phases:

1. **Parts 1–3:** Get the benchmark running on the SLURM cluster (h100-ferranti partition) against a small local model (Qwen2.5-1.5B-Instruct) served via vllm.
2. **Part 4:** Extend to evaluate multiple models — small open-source tutor models locally, and large commercial models via API.

---

## Part 1: Environment Setup

**1.1 Create `slurm/install_env.sh` -- Install dependencies on a GPU node**

On this cluster, CUDA-dependent packages must be installed from a GPU node with the right CUDA toolkit. This script:

- SBATCH: `--partition=h100-ferranti`, `--gres=gpu:1`, `--cpus-per-task=32`, `--mem=512G`, `--time=04:00:00`, `--exclude=mlcbm001`
- Sets up CUDA environment: `CUDA_HOME=$HOME/cuda-12.8.1`, adds to PATH and LD_LIBRARY_PATH
- Enables GCC toolset 11 via `source /opt/rh/gcc-toolset-11/enable`
- Activates `.venv`
- Runs `uv sync` to install core benchmark dependencies
- Runs `uv pip install vllm==0.10.2 --extra-index-url https://wheels.vllm.ai/0.10.2/` to install vllm (pre-built wheel compatible with RHEL 8 / glibc 2.28; newer vllm versions require glibc 2.31+)

This is a one-time setup step. After running, the `.venv` on the shared filesystem is usable by all subsequent SLURM jobs.

**1.2 Fix relative config path** in `main.py:35`

`"configs/" + config_path` breaks when invoked from outside the project root (as SLURM jobs do). Changed to use `Path(__file__).parent / "configs" / config_path` so the script works regardless of working directory.

**1.3 Cluster configuration**

All SLURM partition/node settings live in `slurm/cluster_config.sh` (gitignored). Set up once:
```bash
cp slurm/cluster_config.sh.example slurm/cluster_config.sh
# Edit PARTITION and EXCLUDE for your cluster
```

The pipeline script (`run_pipeline.sh`) sources this file and passes the settings to `sbatch` automatically. For standalone job submissions (like `install_env.sh`), pass them as flags:

**1.4 Install steps**
```bash
# Source your cluster config, then submit with those flags:
source slurm/cluster_config.sh
sbatch --partition="$PARTITION" --exclude="$EXCLUDE" slurm/install_env.sh
```
Or directly:
```bash
sbatch --partition=h100-ferranti --exclude=mlcbm001 slurm/install_env.sh
```
Wait for job to complete, check logs in `logs/` for errors. `HF_HOME` can be set in SLURM scripts to control where model weights are cached.

---

## Part 2: SLURM Integration

All scripts follow cluster conventions: `#!/bin/bash`, `#SBATCH` headers, `set -euo pipefail`, activate `.venv`, print job info and `nvidia-smi`. Scripts use `SLURM_SUBMIT_DIR` to locate the project root (must `sbatch` from the project directory).

**2.1 `slurm/run_benchmark.sh` -- Single-node benchmark job (serve + evaluate)**

Runs vllm and the benchmark on the **same node** in a single SLURM job. This avoids cross-node networking concerns.

- SBATCH: `--partition=h100-ferranti`, `--gres=gpu:1`, `--cpus-per-task=16`, `--mem=64G`, `--time=04:00:00`, `--exclude=mlcbm001`
- Configurable via environment variables with defaults:
  - `MODEL_NAME` (default: `Qwen/Qwen2.5-1.5B-Instruct`)
  - `VLLM_PORT` (default: `8000`)
  - `TENSOR_PARALLEL_SIZE` (default: `1`)
  - `MAX_MODEL_LEN` (default: `4096`)
  - `TASKS` (default: all 9 yaml files comma-separated)
  - `OUTPUT_DIR` (default: `results`)
  - `HEALTH_CHECK_TIMEOUT` (default: `600` seconds)
- Script flow:
  1. Activate `.venv`, print job info, run `nvidia-smi`
  2. `cd` to project root directory
  3. Start vllm server in the background
  4. Store the background PID; set a trap to kill it on exit/error
  5. Health check loop: poll `http://localhost:$VLLM_PORT/health` every 10 seconds until ready or timeout
  6. Run benchmark: `python main.py --tasks $TASKS --provider completion_api --model_args "base_url=http://localhost:$VLLM_PORT/v1,model=$MODEL_NAME"`
  7. Kill the vllm background process
- GPU memory guidance (in comments):
  - Qwen2.5-1.5B-Instruct:  1 GPU,  ~4GB  VRAM
  - Qwen2.5-7B-Instruct:    1 GPU,  ~16GB VRAM
  - Qwen2.5-32B-Instruct:   4 GPUs, ~80GB VRAM
  - Qwen2.5-72B-Instruct:   4+ GPUs

**2.2 `slurm/run_reward_model.sh` -- Run pedagogical reward model**

Separate job because it uses a different model (1.5B reward model) and runs after the benchmark.

- SBATCH: `--partition=h100-ferranti`, `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=32G`, `--time=01:00:00`, `--exclude=mlcbm001`
- Finds all `generations-*.json` files in the results directory
- Runs `python reward_model/compute_scaffolding_score.py --data_path <file>` for each

**2.3 `slurm/run_pipeline.sh` -- Master orchestration script (run from login node)**

- Accepts configuration as command-line arguments with sensible defaults
- Creates `logs/` directory if missing
- Exports config as environment variables
- Submits two jobs with SLURM dependency:
  1. Benchmark job (serves model + runs evaluation)
  2. Reward model job (`--dependency=afterok:<benchmark_job>`)
- Prints summary of submitted job IDs
- Includes model presets for quick config: `--preset qwen-1.5b`, `qwen-7b`, `qwen-32b`, `qwen-72b`

---

## Part 3: Verification

**3.1 Quick smoke test (single task)**
- Run: `TASKS=problem_solving.yaml bash slurm/run_pipeline.sh --preset qwen-1.5b`
- Monitor: `squeue -u $USER`
- Check `logs/` for "Uvicorn running" message and "Health check passed"
- Verify `results/results-Qwen2.5-1.5B-Instruct.yaml` exists with non-trivial accuracy

**3.2 Full benchmark run (all 9 tasks)**
- Run: `bash slurm/run_pipeline.sh --preset qwen-1.5b` (with default TASKS=all)
- Expected runtime: ~1 hour for Qwen2.5-1.5B-Instruct
- After completion, verify:
  - Results YAML has entries for all 9 tasks
  - Generation JSON files exist for scaffolding and pedagogy tasks (4 files)
  - Reward model scores appear in `scaffolding_scores/`

**3.3 Sanity checks**
- Metrics should not be all 0.0 or all 1.0 (would indicate parsing or server issues)
- Compare against README leaderboard for rough ballpark expectations
- Check logs for repeated API errors or CUDA OOM messages

**3.4 Common failure modes**
- vllm OOM: reduce `MAX_MODEL_LEN` or increase GPU count + `TENSOR_PARALLEL_SIZE`
- Health check timeout: model may still be downloading; increase `HEALTH_CHECK_TIMEOUT` or pre-download with `huggingface-cli download`
- Import errors: scripts `cd` to the project root directory to avoid path issues
- glibc incompatibility: RHEL 8 (glibc 2.28) requires vllm <= 0.11.2 for pre-built wheels; newer versions need building from source

---

## Part 4: Multi-Model Support

Extend the benchmark to evaluate a range of models — both open-source models served locally via vllm and commercial models via API.

**Target models:**

| Model | Provider | Deployment | GPU (bench) | is_chat |
|-------|----------|------------|-------------|---------|
| eth-nlped/TutorRL-7B | completion_api (vllm) | Local | 1 GPU (~16GB) | false |
| Qwen/Qwen3-8B | completion_api (vllm) | Local | 1 GPU (~16GB) | false |
| openai/gpt-oss-20b | completion_api (OpenRouter) | API | None | true |
| gpt-5 | completion_api (OpenAI) | API | None | true |
| gemini-3 | gemini | API | None | true |
| claude-opus-4-20250514 | **anthropic** (new) | API | None | true |
| kimi-k2.5 | completion_api (Moonshot) | API | None | true |

Only Anthropic needs a new provider class. OpenRouter/Moonshot/OpenAI are all OpenAI-compatible and use the existing `completion_api` provider. Local models use the existing vllm setup.

**4.1 Fix `CompletionAPI` constructor bug** in `models/completion_api.py:139-152`

When `api_key` is provided, `base_url` is currently ignored — requests always go to `api.openai.com`. This breaks OpenRouter (`https://openrouter.ai/api/v1`) and Moonshot (`https://api.moonshot.cn/v1`). Fix: always pass `base_url` to the OpenAI client when set, regardless of `api_key`. Keep auto-detect logic only for local vllm (no api_key + has base_url).

**4.2 Remove duplicate `create_llm_model`** in `models/completion_api.py:42-51`

Delete the first copy (lines 42-51), keep the second (lines 54-63).

**4.3 Add Anthropic provider** in `models/completion_api.py`

- Add `ANTHROPIC = "anthropic"` to `ProviderType` enum
- Add `import anthropic` at top of file
- Create `AnthropicAPI(BaseLLMAPI)` class:
  - Anthropic API takes `system` as a **separate parameter** (not inside messages) — extract system message from formatted messages list
  - `_make_completion_request` delegates to `_make_chat_request` (Anthropic is chat-only)
  - Handle edge case: current benchmark passes `messages=[]`, so system prompt is the only content — promote to user message when no user messages exist (Anthropic requires ≥1 user message)
  - Same retry/logging pattern as existing providers
- Update factory function: `ProviderType.ANTHROPIC → AnthropicAPI`

**4.4 Add env-var fallback for API keys** in `models/completion_api.py`, `LLMConfig.__post_init__`

When `api_key` is not passed in `--model_args`, auto-resolve from environment:
- `ANTHROPIC_API_KEY` for anthropic provider
- `GEMINI_API_KEY` for gemini provider
- `completion_api` doesn't auto-resolve (serves multiple APIs — key passed by SLURM scripts)

**4.5 Add `anthropic` dependency** in `pyproject.toml`

Add `"anthropic>=0.40.0"` and run `uv add anthropic`.

**4.6 Update `main.py` CLI** (line 44)

Add `'anthropic'` to `--provider` choices.

**4.7 Create `slurm/run_api_benchmark.sh`** (new file)

CPU-only SLURM job for API models — no vllm, no GPU needed for the benchmark itself:
- SBATCH: `--cpus-per-task=4`, `--mem=16G`, `--time=0-08:00`, no `--gres=gpu`
- Required env vars: `PROVIDER`, `MODEL_NAME`
- Optional: `BASE_URL`, `API_KEY_VAR` (name of env var holding the key, dereferenced via `${!API_KEY_VAR}`), `IS_CHAT` (default: true), `TASKS`, `OUTPUT_DIR`
- Builds `--model_args` string dynamically, only including `base_url`/`api_key` when set

**4.8 Extend pipeline presets** in `slurm/run_pipeline.sh`

Add `MODEL_TYPE` variable (`local` or `api`), set by each preset. Branch submission logic:
- `local` → `sbatch run_benchmark.sh` (with `--gres=gpu:N`)
- `api` → `sbatch run_api_benchmark.sh` (no GPU)
- Both followed by reward model job with `--dependency=afterok`

New local presets: `tutorrl-7b`, `qwen3-8b`
New API presets: `gpt-oss-20b`, `gpt-5`, `gemini-3`, `claude-opus`, `kimi-k2.5`

**4.9 Update `slurm/cluster_config.sh.example`**

Add commented-out API key placeholders:
```bash
# export OPENAI_API_KEY=""
# export ANTHROPIC_API_KEY=""
# export GEMINI_API_KEY=""
# export OPENROUTER_API_KEY=""
# export MOONSHOT_API_KEY=""
```

**4.10 Verification**

- Smoke test Anthropic provider: `ANTHROPIC_API_KEY=... uv run python main.py --tasks problem_solving.yaml --provider anthropic --model_args "model=claude-opus-4-20250514,is_chat=true"`
- Smoke test OpenRouter: `uv run python main.py --tasks problem_solving.yaml --provider completion_api --model_args "base_url=https://openrouter.ai/api/v1,model=openai/gpt-oss-20b,api_key=$OPENROUTER_API_KEY,is_chat=true"`
- SLURM presets: `bash slurm/run_pipeline.sh --preset tutorrl-7b` and `bash slurm/run_pipeline.sh --preset claude-opus`
- Verify results YAML and generations JSON created for each model

---

## Part 5: Prompt Optimization (branch: `prompt_optimization`)

Optimize the scaffolding generation system prompt using DSPy + MIPROv2, with the pedagogical reward model as the scoring signal.

**5.1 Add `dspy` dependency**

```bash
uv add dspy
```

**5.2 Create `optimize/metric.py`**

DSPy metric wrapper around `RewardModel` from `reward_model/compute_scaffolding_score.py`:
- Accepts a DSPy `Example` (with `problem`, `reference_solution`, `dialog_history`, `ground_truth_response`) and a `Prediction` (with `teacher_utterance`)
- Formats a single conversation using the same format as `PreferenceDataLoader._format_conversation`
- Returns the raw reward model score (float) for the generated utterance
- `RewardModel` is loaded once and reused across all metric calls

**5.3 Create `optimize/module.py`**

DSPy Signature + Module:
```python
class ScaffoldingSignature(dspy.Signature):
    """You are an experienced math teacher..."""  # ← MIPRO optimizes this docstring
    problem: str = dspy.InputField()
    reference_solution: str = dspy.InputField()
    dialog_history: str = dspy.InputField()
    teacher_utterance: str = dspy.OutputField(desc="maximum two sentences")

class ScaffoldingModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(ScaffoldingSignature)
    def forward(self, problem, reference_solution, dialog_history):
        return self.predict(problem=problem, reference_solution=reference_solution, dialog_history=dialog_history)
```

**5.4 Create `optimize/run_optimize.py`**

Entry point:
- Args: `--model`, `--base_url`, `--train_size` (default 200), `--dev_size` (default 100), `--trials` (default 20), `--output` (default `optimize/optimized_prompt.json`)
- Loads data from `datasets/mathdial_bridge.json` via `dataloaders/mathbridge.MathBridge`
- Splits into train/dev (random, seeded)
- Wraps data as `dspy.Example` objects with `input_keys=("problem", "reference_solution", "dialog_history")`
- Configures `dspy.LM("openai/<model>", base_url=..., api_key="EMPTY")`
- Runs `dspy.MIPROv2` with the reward model metric
- Saves the best optimized program to `--output` (JSON via `program.save()`)
- Prints baseline vs. optimized score comparison

**5.5 Verification**

```bash
# Pilot run (cheap: small sets, few trials)
uv run python optimize/run_optimize.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --base_url http://localhost:8000/v1 \
  --train_size 50 --dev_size 20 --trials 10

# Full run
uv run python optimize/run_optimize.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --base_url http://localhost:8000/v1 \
  --train_size 200 --dev_size 100 --trials 20
```

**Key concerns:**
- Reward model runs on GPU — keep dev set small to control cost per trial
- MIPRO needs 40+ trials for best results; start with 20 to calibrate
- Output constraint (max 2 sentences) must be communicated in the Signature `desc` and enforced in post-processing if needed

---

## Implementation Order

**Parts 1–3 (SLURM setup):**
1. ~~Fix the config path in `main.py` (required for SLURM execution)~~ DONE
2. ~~Create `.gitignore`~~ DONE
3. ~~Create `slurm/install_env.sh` (one-time environment setup)~~ DONE
4. ~~Create `slurm/run_benchmark.sh` (serve model + run evaluation, single node)~~ DONE
5. ~~Create `slurm/run_reward_model.sh`~~ DONE
6. ~~Create `slurm/run_pipeline.sh`~~ DONE
7. ~~Create `logs/.gitkeep`~~ DONE
8. ~~Submit `slurm/install_env.sh` to install dependencies~~ DONE
9. ~~Run smoke test with single task~~ DONE — Job 321219, 2026-02-19, `problem_solving` accuracy = 0.5497 (Qwen2.5-1.5B-Instruct on mlcbm008, ~10 min)
10. Run full benchmark (all 9 tasks)

**Part 5 (prompt optimization, branch: `prompt_optimization`):**
11. ~~Add `dspy` dependency (5.1)~~ DONE
12. ~~Implement `optimize/metric.py` (5.2)~~ DONE
13. ~~Implement `optimize/module.py` (5.3)~~ DONE
14. ~~Implement `optimize/run_optimize.py` (5.4)~~ DONE
15. ~~Create `slurm/optimize_job.sh` + `optimize/run_optimize.sh` (SLURM submission scripts)~~ DONE
16. Pilot run (small train/dev, few trials) to verify pipeline end-to-end (5.5)
17. Full optimization run and baseline comparison

**Part 4 (multi-model support):**
18. Fix `CompletionAPI` constructor + remove duplicate `create_llm_model` (4.1, 4.2)
19. Add Anthropic provider class + env-var fallback for API keys (4.3, 4.4)
20. Add `anthropic` dependency (4.5)
21. Update `main.py` CLI with `anthropic` provider choice (4.6)
22. Create `slurm/run_api_benchmark.sh` (4.7)
23. Extend pipeline presets in `slurm/run_pipeline.sh` (4.8)
24. Update `slurm/cluster_config.sh.example` with API key placeholders (4.9)
25. Smoke test: Anthropic provider locally
26. Smoke test: OpenRouter provider locally
27. SLURM test: local preset (tutorrl-7b)
28. SLURM test: API preset (claude-opus)
29. Full benchmark runs across all target models

---

## Files Modified

| File | Change | Part |
|------|--------|------|
| `main.py:35` | Fixed config path to use `Path(__file__).parent` | 1 |
| `models/completion_api.py` | Fix CompletionAPI constructor, rm duplicate factory, add AnthropicAPI, env-var fallback | 4 |
| `main.py:44` | Add `'anthropic'` to `--provider` choices | 4 |
| `pyproject.toml` | Add `anthropic>=0.40.0` dependency | 4 |
| `slurm/run_pipeline.sh` | Add 7 new presets, MODEL_TYPE branching (local vs api) | 4 |
| `slurm/cluster_config.sh.example` | Add API key placeholder comments | 4 |

## Files Created

| File | Purpose | Part |
|------|---------|------|
| `slurm/install_env.sh` | SLURM job: one-time env setup (install deps + vllm on GPU node) | 1 |
| `slurm/run_benchmark.sh` | SLURM job: serve model + run benchmark (single node) | 2 |
| `slurm/run_reward_model.sh` | SLURM job: run pedagogical reward model | 2 |
| `slurm/run_pipeline.sh` | Master orchestration (run from login node) | 2 |
| `logs/.gitkeep` | Log directory placeholder | 2 |
| `.gitignore` | Ignore logs, results, pycache, .venv | 1 |
| `slurm/cluster_config.sh.example` | Template for cluster-specific SLURM settings | 1 |
| `slurm/cluster_config.sh` | Actual cluster config (gitignored) | 1 |
| `slurm/run_api_benchmark.sh` | SLURM job: run benchmark against API models (CPU-only) | 4 |
| `optimize/metric.py` | DSPy metric wrapper around RewardModel | 5 |
| `optimize/module.py` | DSPy Signature + ScaffoldingModule | 5 |
| `optimize/run_optimize.py` | MIPROv2 optimization entry point | 5 |
| `optimize/run_optimize.sh` | Login-node wrapper: parses CLI args, submits SLURM job | 5 |
| `slurm/optimize_job.sh` | SLURM job: starts vllm + runs optimization (2 GPUs) | 5 |

---

## Future TODOs (not in scope now)

These are issues found during codebase exploration to address later:

**Bugs:**
- ~~Duplicate `create_llm_model` function in `models/completion_api.py:42-63` (identical definition twice)~~ Fixed in Part 4 (step 4.2)
- Misleading `parse_response` docstrings in scaffolding/pedagogy tasks (say "boolean" but return raw string)
- Reward model references global `args.data_path` instead of function parameter in `reward_model/compute_scaffolding_score.py:210`
- Test import error: `from extraction import ...` should be `from tasks.extraction import ...` in `tests/test_socratic_questioning.py:3`
- Duplicate test method name `test_extract_questions` in same file (lines 8 and 19)
- Class name typo: `ScaffoldingGeneretionHard` should be `ScaffoldingGenerationHard` in `tasks/scaffolding_generation_hard.py` and `tasks/__init__.py:8`

**Code quality:**
- Add docstrings to all modules, classes, and public functions
- Replace `print()` with `logging` in `main.py` and `models/completion_api.py`
- Extract duplicated `_is_question` helper (in 4 task files) to `tasks/extraction.py`
- Fix inaccurate type hints (e.g., `GSM8K.parse_response` declares `-> float` but returns `str`)
