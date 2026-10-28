"""Symbolic specification of the training loss. No model and no tuned numbers.

The step reward used in training is the rubric change. Token KL against the
teacher is a gated auxiliary term. These helpers only decide the gate, the
optional non-negative agreement bonus, and how returns are cut at a handoff.
"""

from __future__ import annotations

from dataclasses import dataclass


# Design defaults, not measured hyperparameters.
READ_WEIGHT = 0.2
COMMIT_WEIGHT = 1.0


def rubric_delta(verified: dict[str, float], broken: dict[str, float]) -> float:
    """Reward for one step: new credit minus credit that was undone."""

    return sum(verified.values()) - sum(broken.values())


def kl_coefficient(label: str, committing: bool, neutral: float = 0.0) -> float:
    """How hard this student span is pulled toward the teacher.

    neutral is an ablation. The first training run uses 0. A small positive
    value is a separate run, not a default that has been tuned.
    """

    if neutral < 0:
        raise ValueError("neutral KL weight must be non-negative")
    if label == "progress":
        return 0.0
    if label == "invalidation":
        return 1.0
    if label == "dependency_block":
        return 1.0 if committing else neutral
    if label in {"neutral_unknown", "neutral_repeat", "neutral_no_effect"}:
        return neutral
    raise ValueError(f"unknown label: {label}")


def agreement_bonus(
    label: str,
    committing: bool,
    matched: bool,
    bonus: float = 0.0,
) -> float:
    """Black-box stand-in for KL. Matching can add reward; mismatch cannot subtract.

    Progress steps get nothing here: the rubric delta is already their signal.
    """

    if bonus < 0:
        raise ValueError("agreement bonus must be non-negative")
    if not matched or label == "progress":
        return 0.0
    if label in {"neutral_unknown", "neutral_repeat", "neutral_no_effect"}:
        return bonus
    if label == "invalidation" or (label == "dependency_block" and committing):
        return bonus
    if label == "dependency_block":
        return bonus
    raise ValueError(f"unknown label: {label}")


@dataclass(frozen=True)
class StudentStep:
    reward: float
    decision: str


def student_returns(
    steps: list[StudentStep],
    outcome: float,
    gamma: float = 1.0,
) -> list[float]:
    """Returns for student-attempt steps only. Teacher progress is not an input.

    No handoff: ordinary discounted return, with the outcome attached to the
    last student step.
    One intervene: steps before it do not see that step, later student steps,
    or the outcome. The replaced step keeps only its own reward. Later student
    steps use the ordinary return, including the outcome.
    Stop: the trajectory ends there. Earlier steps do see the stop step,
    because no teacher produced the ending.
    """

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    cuts = [index for index, step in enumerate(steps) if step.decision == "intervene"]
    if len(cuts) > 1:
        raise ValueError("return cutting is specified for one intervention")
    if any(step.decision == "stop" for step in steps[:-1]):
        raise ValueError("stop must be the last generated student step")

    if cuts:
        cut = cuts[0]
        prefix = _discount([step.reward for step in steps[:cut]], 0.0, gamma)
        suffix = _discount([step.reward for step in steps[cut + 1 :]], outcome, gamma)
        return prefix + [steps[cut].reward] + suffix

    end = len(steps)
    if steps and steps[-1].decision == "stop":
        end = len(steps)
    return _discount([step.reward for step in steps[:end]], outcome, gamma)


def _discount(rewards: list[float], outcome: float, gamma: float) -> list[float]:
    """Outcome is part of the last step, then discounted one step at a time."""

    if not rewards:
        return []
    out = [0.0] * len(rewards)
    tail = 0.0
    last = len(rewards) - 1
    for index in range(last, -1, -1):
        extra = outcome if index == last else 0.0
        tail = rewards[index] + extra + (0.0 if index == last else gamma * tail)
        out[index] = tail
    return out
