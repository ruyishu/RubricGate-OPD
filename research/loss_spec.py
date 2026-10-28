"""Symbolic specification of the training loss. No model and no tuned numbers.

The step reward used in training is the rubric change. The teacher enters the
gradient only at consulted states: confirmed errors, and stalls where the
rubric has not moved for several steps. White-box teachers enter through token
KL on the student's own span. Black-box teachers enter through a few student
candidates resampled at that state.
"""

from __future__ import annotations

from dataclasses import dataclass


# Design defaults, not measured hyperparameters.
READ_WEIGHT = 0.2
COMMIT_WEIGHT = 1.0
STALL_STEPS = 3

NEUTRAL_LABELS = frozenset({"neutral_unknown", "neutral_repeat", "neutral_no_effect"})
LABELS = NEUTRAL_LABELS | {"progress", "invalidation", "dependency_block"}


def rubric_delta(verified: dict[str, float], broken: dict[str, float]) -> float:
    """Reward for one step: new credit minus credit that was undone."""

    return sum(verified.values()) - sum(broken.values())


def stall_flags(changed: list[bool], k: int = STALL_STEPS) -> list[bool]:
    """A step is stalled when it ends a run of at least k steps with no rubric change."""

    if k < 1:
        raise ValueError("stall length must be at least 1")
    flags: list[bool] = []
    run = 0
    for moved in changed:
        run = 0 if moved else run + 1
        flags.append(run >= k)
    return flags


def is_error(label: str, committing: bool) -> bool:
    _check(label)
    return label == "invalidation" or (label == "dependency_block" and committing)


def consults_teacher(label: str, committing: bool, stalled: bool) -> bool:
    """States where the teacher is asked at all."""

    if is_error(label, committing):
        return True
    return label != "progress" and stalled


def kl_coefficient(
    label: str,
    committing: bool,
    stalled: bool = False,
    neutral: float = 0.0,
    stall: float = 1.0,
) -> float:
    """How hard this student span is pulled toward the teacher.

    neutral is an ablation for isolated no-change steps; the first run uses 0.
    stall applies once the rubric has not moved for STALL_STEPS steps.
    """

    if neutral < 0 or stall < 0:
        raise ValueError("KL weights must be non-negative")
    if label == "progress":
        return 0.0
    if is_error(label, committing):
        return 1.0
    return stall if stalled else neutral


def candidate_reward(progressed: bool, broke: bool, agreement: float) -> float:
    """Reward for one resampled student candidate at a consulted state.

    The candidate runs on a copy of the environment. Rubric progress earns full
    credit whatever the teacher would do, so a valid alternative is never ranked
    below a teacher match. Breaking an established checkpoint earns 0. Otherwise
    the reward is the fraction of teacher samples the candidate matches.
    """

    if not 0 <= agreement <= 1:
        raise ValueError("agreement must lie in [0, 1]")
    if broke:
        return 0.0
    if progressed:
        return 1.0
    return agreement


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
    return _discount([step.reward for step in steps], outcome, gamma)


def _check(label: str) -> None:
    if label not in LABELS:
        raise ValueError(f"unknown label: {label}")


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
