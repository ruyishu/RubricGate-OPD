"""Scorers compared by the mechanism fixture.

action_match imitates same-state teacher-action agreement.
flat_checklist ignores dependencies and environment failure.
dars_like is a tiny potential difference, not a reimplementation of DARS.
rubricgate_v1 is the score-only rule. rubricgate_v2 additionally chooses
continue / intervene / stop and keeps teacher progress off the student return.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from research.toyenv import Dag, Episode, Step, classify, transition, unlocked


PROGRESS_REWARD = 1.0
INVALIDATION_PENALTY = -1.0


def reward_v1(label: str) -> float:
    """Positive only for a newly verified subgoal. Unknown work stays at zero."""

    if label == "progress":
        return PROGRESS_REWARD
    if label == "invalidation":
        return INVALIDATION_PENALTY
    return 0.0


def decide(
    label: str,
    committing: bool,
    interventions_used: int,
    max_interventions: int,
) -> str:
    """Phase-2 router. Missing checklist items do not trigger intervention."""

    if label in {"progress", "neutral_unknown", "neutral_repeat", "neutral_no_effect"}:
        return "continue"
    if label == "dependency_block" and not committing:
        return "continue"
    if interventions_used >= max_interventions:
        return "stop"
    return "intervene"


def teacher_probs(ready: tuple[str, ...], prefer_b: float) -> dict[str, float]:
    if not ready:
        return {}
    if set(ready) == {"b", "c"}:
        return {"b": prefer_b, "c": 1.0 - prefer_b}
    share = 1.0 / len(ready)
    return {name: share for name in ready}


def sample_teacher(
    ready: tuple[str, ...], prefer_b: float, k: int, rng: random.Random
) -> list[str]:
    probs = teacher_probs(ready, prefer_b)
    if not probs or k <= 0:
        return []
    names = list(probs)
    weights = [probs[name] for name in names]
    return [rng.choices(names, weights=weights, k=1)[0] for _ in range(k)]


def match_score(target: str | None, samples: list[str]) -> float:
    """Fraction of same-state teacher samples equal to the student target."""

    if target is None or not samples:
        return 0.0
    return sum(sample == target for sample in samples) / len(samples)


def flat_reward(dag: Dag, done: frozenset[str], step: Step) -> float:
    """Name is on the checklist and not yet checked. Preconditions are ignored."""

    if step.target is None or step.target not in dag.nodes or step.target in done:
        return 0.0
    return PROGRESS_REWARD


@dataclass
class GateReplay:
    student_return: float
    teacher_progress: float
    decisions: list[str] = field(default_factory=list)
    student_step_rewards: list[float] = field(default_factory=list)
    stopped_early: bool = False


def replay_score(dag: Dag, episode: Episode) -> list[float]:
    done: frozenset[str] = frozenset()
    rewards: list[float] = []
    for step in episode.steps:
        label = classify(dag, done, step)
        rewards.append(reward_v1(label))
        done, _ = transition(dag, done, step)
    return rewards


def replay_potential(dag: Dag, episode: Episode) -> list[float]:
    """Signed change in the fraction of verified nodes.

    This is only the potential difference. It does not implement DARS's
    downstream attenuation, repair rules, or annotator.
    """

    done: frozenset[str] = frozenset()
    width = float(len(dag.nodes))
    rewards: list[float] = []
    for step in episode.steps:
        updated, _ = transition(dag, done, step)
        rewards.append((len(updated) - len(done)) / width)
        done = updated
    return rewards


def replay_gate(
    dag: Dag,
    episode: Episode,
    max_interventions: int = 1,
) -> GateReplay:
    """Student return excludes any subgoal completed by the teacher handoff."""

    done: frozenset[str] = frozenset()
    student_return = 0.0
    teacher_progress = 0.0
    decisions: list[str] = []
    student_step_rewards: list[float] = []
    used = 0
    for step in episode.steps:
        label = classify(dag, done, step)
        student_reward = reward_v1(label)
        student_step_rewards.append(student_reward)
        student_return += student_reward
        node = dag.nodes.get(step.target) if step.target else None
        committing = bool(node and node.committing and label == "dependency_block")
        if label == "invalidation":
            committing = True
        decision = decide(label, committing, used, max_interventions)
        decisions.append(decision)
        if decision == "stop":
            return GateReplay(
                student_return, teacher_progress, decisions, student_step_rewards, True
            )
        if decision == "intervene":
            used += 1
            ready = unlocked(dag, done)
            if ready:
                teacher_step = Step(ready[0], ready[0], env_ok=True)
                done, teacher_label = transition(dag, done, teacher_step)
                if teacher_label == "progress":
                    teacher_progress += PROGRESS_REWARD
            continue
        done, _ = transition(dag, done, step)
    return GateReplay(
        student_return, teacher_progress, decisions, student_step_rewards, False
    )

