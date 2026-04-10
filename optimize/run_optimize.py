"""
MIPROv2 prompt optimization entry point.

Optimizes the scaffolding generation system prompt (instruction + few-shot
demos, jointly) using the pedagogical reward model as the scoring signal.

Usage:
    uv run python optimize/run_optimize.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --base_url http://localhost:8000/v1 \
        --prompt_model Qwen/Qwen2.5-7B-Instruct \
        --prompt_base_url http://localhost:8001/v1 \
        --auto medium
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy

from dataloaders.mathbridge import MathBridge
from optimize.metric import reward_metric
from optimize.module import ScaffoldingModule


def load_dspy_examples(data_path: str) -> list[dspy.Example]:
    loader = MathBridge(data_path)
    raw = loader.load()
    examples = []
    for item in raw:
        ex = dspy.Example(
            problem=item["question"],
            reference_solution=item["reference_solution"],
            dialog_history=item["dialog_history"],
            conversation_json=item["conversation_json"],
        ).with_inputs("problem", "reference_solution", "dialog_history")
        examples.append(ex)
    return examples


def diagnose_outputs(program: dspy.Module, examples: list[dspy.Example], n: int = 3):
    """Run a few examples and print full predictions to diagnose truncation."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: Running {n} examples to inspect raw outputs")
    print(f"{'='*60}")
    for i, ex in enumerate(examples[:n]):
        pred = program(
            problem=ex.problem,
            reference_solution=ex.reference_solution,
            dialog_history=ex.dialog_history,
        )
        reasoning = getattr(pred, "reasoning", "")
        utterance = getattr(pred, "teacher_utterance", "")
        reasoning_tokens = len(reasoning) // 4  # rough estimate
        utterance_tokens = len(utterance) // 4
        total_chars = len(reasoning) + len(utterance)
        total_tokens_est = total_chars // 4

        print(f"\n--- Example {i+1} ---")
        print(f"Problem: {ex.problem[:100]}...")
        print(f"Reasoning ({reasoning_tokens}~tok, {len(reasoning)} chars):")
        print(f"  {reasoning[:500]}{'...[TRUNCATED]' if len(reasoning) > 500 else ''}")
        print(f"Teacher utterance ({utterance_tokens}~tok, {len(utterance)} chars):")
        print(f"  {utterance}")
        print(f"Total output: ~{total_tokens_est} tokens ({total_chars} chars)")
        if not utterance:
            print("  *** WARNING: teacher_utterance is EMPTY — likely truncated before reaching this field ***")
    print(f"{'='*60}\n")


def evaluate_program(program: dspy.Module, examples: list[dspy.Example]) -> float:
    scores = []
    for ex in examples:
        pred = program(
            problem=ex.problem,
            reference_solution=ex.reference_solution,
            dialog_history=ex.dialog_history,
        )
        score = reward_metric(ex, pred)
        scores.append(score)
    return sum(scores) / len(scores)


# Split presets. Both modes hold out the same test set for fair comparison
# and use the same total optimization budget (train + dev), differing only
# in how examples are allocated between train and dev.
SPLIT_MODES = {
    # MIPROv2 paper style: balanced train/dev.
    "paper": {"train_size": 400, "dev_size": 400, "test_size": 350},
    # DSPy recommendation: 20% train, 80% validation — "stable validation"
    # to reduce prompt-optimizer overfitting.
    "dspy":  {"train_size": 160, "dev_size": 640, "test_size": 350},
}


def main():
    parser = argparse.ArgumentParser(description="Optimize scaffolding prompt with DSPy MIPROv2")
    parser.add_argument("--model", required=True, help="Model name (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--base_url", required=True, help="vllm base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max completion tokens for the task LM (avoids silent truncation of ChainOfThought reasoning + output)")
    parser.add_argument(
        "--prompt_model",
        default=None,
        help="Separate (stronger) model used by MIPROv2 to propose candidate instructions. "
             "If unset, MIPROv2 falls back to the task model — which for small models (e.g. 1.5B) "
             "produces poor, example-regurgitating instructions.",
    )
    parser.add_argument("--prompt_base_url", default=None, help="Base URL for --prompt_model (OpenAI-compatible endpoint)")
    parser.add_argument("--prompt_api_key", default="EMPTY", help="API key for --prompt_model (default: 'EMPTY' for local vllm)")
    parser.add_argument("--prompt_max_tokens", type=int, default=4096, help="Max completion tokens for the prompt proposer model")
    parser.add_argument(
        "--mode",
        choices=list(SPLIT_MODES.keys()),
        default="paper",
        help="Split preset: 'paper' (balanced train/dev, MIPROv2-paper-style) or 'dspy' (20/80 train/val, DSPy-recommended)",
    )
    parser.add_argument("--train_size", type=int, default=None, help="Override --mode preset: number of training examples")
    parser.add_argument("--dev_size", type=int, default=None, help="Override --mode preset: number of dev examples (used by optimizer for validation)")
    parser.add_argument("--test_size", type=int, default=None, help="Override --mode preset: number of held-out test examples (not seen during optimization)")
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPROv2 auto setting controlling trial count (default: light)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_path",
        default="datasets/mathdial_bridge.json",
        help="Path to mathdial_bridge.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the optimized DSPy program (default: auto-generated with timestamp)",
    )
    args = parser.parse_args()

    # Resolve split sizes: --mode preset provides defaults, explicit flags override.
    preset = SPLIT_MODES[args.mode]
    if args.train_size is None:
        args.train_size = preset["train_size"]
    if args.dev_size is None:
        args.dev_size = preset["dev_size"]
    if args.test_size is None:
        args.test_size = preset["test_size"]
    print(f"Split mode: {args.mode}  (train={args.train_size}, dev={args.dev_size}, test={args.test_size})")

    # Generate default output path with timestamp — include mode for easy comparison
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"optimize/results/optimized_program_{args.mode}_{timestamp}.json"

    random.seed(args.seed)

    # Configure DSPy LM (OpenAI-compatible endpoint)
    lm = dspy.LM(
        f"openai/{args.model}",
        api_base=args.base_url,
        api_key="EMPTY",
        model_type="chat",
        temperature=0.1,
        max_tokens=args.max_tokens,
        # Prevent the model from continuing a hallucinated dialog past its own turn.
        # "\n\n" is deliberately NOT included — it would cut DSPy's structured output
        # format between the reasoning and teacher_utterance fields.
        stop=["Student:", "Teacher:"],
    )
    dspy.configure(lm=lm)

    # Configure prompt proposer LM for MIPROv2 (optional).
    # Higher temperature encourages instruction diversity during proposal.
    prompt_lm = None
    if args.prompt_model is not None:
        prompt_lm = dspy.LM(
            f"openai/{args.prompt_model}",
            api_base=args.prompt_base_url,
            api_key=args.prompt_api_key,
            model_type="chat",
            temperature=0.7,
            max_tokens=args.prompt_max_tokens,
        )
        print(f"Using prompt proposer model: {args.prompt_model} @ {args.prompt_base_url}")
    else:
        print(
            "WARNING: No --prompt_model set. MIPROv2 will use the task model to propose instructions. "
            "Small task models (e.g. 1.5B) tend to regurgitate training examples verbatim."
        )

    # Load data
    print(f"Loading data from {args.data_path} ...")
    examples = load_dspy_examples(args.data_path)
    random.shuffle(examples)

    total_needed = args.train_size + args.dev_size + args.test_size
    if total_needed > len(examples):
        raise ValueError(
            f"Requested split ({args.train_size} train + {args.dev_size} dev + "
            f"{args.test_size} test = {total_needed}) exceeds dataset size ({len(examples)})."
        )
    train = examples[: args.train_size]
    dev = examples[args.train_size : args.train_size + args.dev_size]
    test = examples[args.train_size + args.dev_size : args.train_size + args.dev_size + args.test_size]
    print(f"Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")

    # Diagnostic: inspect a few raw outputs before committing to full evaluation
    baseline_program = ScaffoldingModule()
    diagnose_outputs(baseline_program, train, n=3)

    # Baseline scores (train included for overfitting diagnosis)
    print("Computing baseline score on train set ...")
    baseline_train_score = evaluate_program(baseline_program, train)
    print(f"Baseline train score: {baseline_train_score:.4f}")
    print("Computing baseline score on dev set ...")
    baseline_dev_score = evaluate_program(baseline_program, dev)
    print(f"Baseline dev score: {baseline_dev_score:.4f}")
    print("Computing baseline score on test set ...")
    baseline_test_score = evaluate_program(baseline_program, test)
    print(f"Baseline test score: {baseline_test_score:.4f}")

    # Create MIPROv2 optimizer
    mipro_kwargs = {"metric": reward_metric, "auto": args.auto}
    if prompt_lm is not None:
        mipro_kwargs["prompt_model"] = prompt_lm
    optimizer = dspy.MIPROv2(**mipro_kwargs)

    print(f"Running MIPROv2 (auto='{args.auto}') ...")
    optimized_program = optimizer.compile(ScaffoldingModule(), trainset=train, valset=dev)

    # Evaluate optimized program on train, dev, and held-out test
    print("Computing optimized score on train set ...")
    optimized_train_score = evaluate_program(optimized_program, train)
    print(f"Optimized train score: {optimized_train_score:.4f}  (baseline: {baseline_train_score:.4f}, improvement: {optimized_train_score - baseline_train_score:+.4f})")
    print("Computing optimized score on dev set ...")
    optimized_dev_score = evaluate_program(optimized_program, dev)
    print(f"Optimized dev score: {optimized_dev_score:.4f}  (baseline: {baseline_dev_score:.4f}, improvement: {optimized_dev_score - baseline_dev_score:+.4f})")
    print("Computing optimized score on test set ...")
    optimized_test_score = evaluate_program(optimized_program, test)
    print(f"Optimized test score: {optimized_test_score:.4f}  (baseline: {baseline_test_score:.4f}, improvement: {optimized_test_score - baseline_test_score:+.4f})")

    # Save optimized program
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(output_path))
    print(f"Saved optimized program to {output_path}")

    # Save summary with all args for reproducibility
    summary = {
        "model": args.model,
        "optimizer": "mipro",
        "baseline_train_score": baseline_train_score,
        "baseline_dev_score": baseline_dev_score,
        "baseline_test_score": baseline_test_score,
        "optimized_train_score": optimized_train_score,
        "optimized_dev_score": optimized_dev_score,
        "optimized_test_score": optimized_test_score,
        "train_improvement": optimized_train_score - baseline_train_score,
        "dev_improvement": optimized_dev_score - baseline_dev_score,
        "test_improvement": optimized_test_score - baseline_test_score,
        "args": vars(args),
        "train_size_actual": len(train),
        "dev_size_actual": len(dev),
        "test_size_actual": len(test),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
