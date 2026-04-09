"""
DSPy Signature and Module for scaffolding generation.

The Signature docstring is what MIPROv2 optimizes.
The initial docstring mirrors the current system prompt in configs/scaffolding_generation.yaml.
"""

import dspy


class ScaffoldingSignature(dspy.Signature):
    """You are an experienced math teacher and you are going to respond to a student in a useful and caring way. The student is trying to solve the following problem."""

    problem: str = dspy.InputField(desc="The math problem the student is working on")
    reference_solution: str = dspy.InputField(desc="The correct reference solution to the problem")
    dialog_history: str = dspy.InputField(desc="The conversation so far between teacher and student")
    teacher_utterance: str = dspy.OutputField(desc="Your response as the teacher (maximum two sentences)")


class ScaffoldingModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought(ScaffoldingSignature)

    def forward(self, problem: str, reference_solution: str, dialog_history: str) -> dspy.Prediction:
        return self.predict(
            problem=problem,
            reference_solution=reference_solution,
            dialog_history=dialog_history,
        )
