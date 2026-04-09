"""
DSPy metric wrapping the pedagogical reward model.

Scores a generated teacher utterance by running it through the reward model.
Returns the raw score (float) — higher is better.
"""

import sys
from pathlib import Path

import dspy

# Make reward_model importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from reward_model.compute_scaffolding_score import RewardModel, SYSTEM_PROMPT

_reward_model: RewardModel | None = None
REWARD_MODEL_NAME = "eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel"


def _get_reward_model() -> RewardModel:
    global _reward_model
    if _reward_model is None:
        _reward_model = RewardModel(REWARD_MODEL_NAME)
    return _reward_model


def _format_conversation(
    problem: str,
    reference_solution: str,
    dialog_history: list[dict],
    response: str,
) -> list[dict]:
    """Format a single conversation for the reward model (mirrors PreferenceDataLoader._format_conversation)."""
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Problem: {problem}\nReference Solution: {reference_solution}",
        },
    ]
    for entry in dialog_history:
        role = "assistant" if entry["user"] in ("Teacher", "Tutor") else "user"
        conversation.append({"role": role, "content": entry["text"]})
    conversation.append({"role": "assistant", "content": response})
    return conversation


def reward_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    DSPy metric function.

    Args:
        example: Must have fields `problem`, `reference_solution`, `conversation_json` (list of dialog turns).
        prediction: Must have field `teacher_utterance`.
        trace: Unused (required by DSPy metric signature).

    Returns:
        Reward model score for the generated teacher utterance (float).
    """
    rm = _get_reward_model()
    conversation = _format_conversation(
        problem=example.problem,
        reference_solution=example.reference_solution,
        dialog_history=example.conversation_json,
        response=prediction.teacher_utterance,
    )
    scores = rm.get_scores([conversation])
    return scores[0]
