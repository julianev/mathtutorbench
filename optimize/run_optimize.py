"""
Prompt optimization entry point using DSPy optimizers.

Optimizes the scaffolding generation system prompt using the pedagogical
reward model as the scoring signal.

Usage:
    # MIPROv2 (default — optimizes instruction + few-shot jointly):
    uv run python optimize/run_optimize.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --base_url http://localhost:8000/v1 \
        --optimizer mipro --trials 20

    # BootstrapFewShotWithRandomSearch (cheaper — few-shot only):
    uv run python optimize/run_optimize.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --base_url http://localhost:8000/v1 \
        --optimizer bootstrap_rs --trials 10

    # BootstrapFewShot (cheapest — single pass):
    uv run python optimize/run_optimize.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --base_url http://localhost:8000/v1 \
        --optimizer bootstrap
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


def main():
    parser = argparse.ArgumentParser(description="Optimize scaffolding prompt with DSPy MIPROv2")
    parser.add_argument("--model", required=True, help="Model name (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--base_url", required=True, help="vllm base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--train_size", type=int, default=200, help="Number of training examples")
    parser.add_argument("--dev_size", type=int, default=100, help="Number of dev examples for metric evaluation")
    parser.add_argument("--trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument(
        "--optimizer",
        choices=["mipro", "bootstrap", "bootstrap_rs"],
        default="mipro",
        help="DSPy optimizer: mipro (MIPROv2), bootstrap (BootstrapFewShot), bootstrap_rs (BootstrapFewShotWithRandomSearch)",
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

    # Generate default output path with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"optimize/results/optimized_program_{timestamp}.json"

    random.seed(args.seed)

    # Configure DSPy LM (OpenAI-compatible endpoint)
    lm = dspy.LM(
        f"openai/{args.model}",
        api_base=args.base_url,
        api_key="EMPTY",
        model_type="chat",
        temperature=0.0,
        max_tokens=256,
        stop=["Student:", "\n\n", "Teacher:"],
    )
    dspy.configure(lm=lm)

    # Load data
    print(f"Loading data from {args.data_path} ...")
    examples = load_dspy_examples(args.data_path)
    random.shuffle(examples)

    train = examples[: args.train_size]
    dev = examples[args.train_size : args.train_size + args.dev_size]
    print(f"Train: {len(train)}, Dev: {len(dev)}, Remaining: {len(examples) - len(train) - len(dev)}")

    # Baseline score
    print("Computing baseline score on dev set ...")
    baseline_program = ScaffoldingModule()
    baseline_score = evaluate_program(baseline_program, dev)
    print(f"Baseline mean reward score: {baseline_score:.4f}")

    # Create optimizer
    if args.optimizer == "mipro":
        optimizer = dspy.MIPROv2(
            metric=reward_metric,
            auto="light",
        )
    elif args.optimizer == "bootstrap":
        optimizer = dspy.BootstrapFewShot(
            metric=reward_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
        )
    elif args.optimizer == "bootstrap_rs":
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=reward_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_candidate_programs=args.trials,
        )

    if args.optimizer == "mipro":
        print(f"Running {args.optimizer} (auto='light') ...")
    else:
        print(f"Running {args.optimizer} with {args.trials} trials ...")
    optimized_program = optimizer.compile(
        ScaffoldingModule(),
        trainset=train,
        valset=dev,
    )

    # Evaluate optimized program
    print("Computing optimized score on dev set ...")
    optimized_score = evaluate_program(optimized_program, dev)
    print(f"Optimized mean reward score: {optimized_score:.4f}")
    print(f"Improvement: {optimized_score - baseline_score:+.4f}")

    # Save optimized program
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(output_path))
    print(f"Saved optimized program to {output_path}")

    # Save summary
    summary = {
        "model": args.model,
        "optimizer": args.optimizer,
        "train_size": len(train),
        "dev_size": len(dev),
        "trials": args.trials,
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": optimized_score - baseline_score,
    }
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
