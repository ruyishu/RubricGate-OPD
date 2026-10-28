"""Symbolic four-step tasks with a known dependency DAG.

The environment is a mechanism fixture. It does not estimate how often a real
teacher model penalizes legal reorderings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    name: str
    parents: tuple[str, ...]
    committing: bool = False


@dataclass(frozen=True)
class Dag:
    nodes: dict[str, Node]

    def names(self) -> tuple[str, ...]:
        return tuple(self.nodes)


@dataclass(frozen=True)
class Step:
    action: str
    target: str | None
    env_ok: bool = True
    invalidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class Episode:
    episode_id: str
    category: str
    steps: tuple[Step, ...]


def make_dag(extra_parents: dict[str, tuple[str, ...]] | None = None) -> Dag:
    """Task DAG: a, then b and c in either order, then committing d.

    extra_parents replaces the parent tuple of the named node. Used to feed the
    scorer a deliberately wrong annotation.
    """

    spec = {
        "a": Node("a", ()),
        "b": Node("b", ("a",)),
        "c": Node("c", ("a",)),
        "d": Node("d", ("b", "c"), committing=True),
    }
    if extra_parents:
        for name, parents in extra_parents.items():
            node = spec[name]
            spec[name] = Node(name, parents, node.committing)
    return Dag(spec)


TRUE_DAG = make_dag()
# Destroys the legal swap: c is annotated as requiring b.
WRONG_DAG = make_dag({"c": ("a", "b")})


def unlocked(dag: Dag, done: frozenset[str]) -> tuple[str, ...]:
    ready = [
        node.name
        for node in dag.nodes.values()
        if node.name not in done and set(node.parents) <= done
    ]
    return tuple(sorted(ready))


def classify(dag: Dag, done: frozenset[str], step: Step) -> str:
    """Legality of one executed step. Unknown work is not an error."""

    damaged = tuple(name for name in step.invalidates if name in done)
    if damaged:
        return "invalidation"
    if step.target is None or step.target not in dag.nodes:
        return "neutral_unknown"
    if step.target in done:
        return "neutral_repeat"
    node = dag.nodes[step.target]
    if not set(node.parents) <= done:
        return "dependency_block"
    if not step.env_ok:
        return "neutral_no_effect"
    return "progress"


def transition(
    dag: Dag, done: frozenset[str], step: Step
) -> tuple[frozenset[str], str]:
    label = classify(dag, done, step)
    updated = set(done)
    if label == "invalidation":
        updated.difference_update(name for name in step.invalidates if name in done)
    elif label == "progress" and step.target is not None:
        updated.add(step.target)
    return frozenset(updated), label


def _ep(category: str, index: int, steps: tuple[Step, ...]) -> Episode:
    return Episode(f"{category}-{index:03d}", category, steps)


def generate_episodes() -> list[Episode]:
    """100 scripted episodes. Counts are fixed so the table does not depend on luck."""

    episodes: list[Episode] = []
    plan: list[tuple[str, int, tuple[Step, ...]]] = [
        (
            "canonical",
            15,
            (
                Step("a", "a"),
                Step("b", "b"),
                Step("c", "c"),
                Step("d", "d"),
            ),
        ),
        (
            "legal_swap",
            20,
            (
                Step("a", "a"),
                Step("c", "c"),
                Step("b", "b"),
                Step("d", "d"),
            ),
        ),
        (
            "committing_block",
            15,
            (
                Step("a", "a"),
                Step("b", "b"),
                Step("d", "d"),
                Step("c", "c"),
            ),
        ),
        (
            "unknown",
            15,
            (
                Step("a", "a"),
                Step("look", None),
                Step("b", "b"),
                Step("c", "c"),
            ),
        ),
        (
            "repeat",
            10,
            (
                Step("a", "a"),
                Step("b", "b"),
                Step("b", "b"),
                Step("c", "c"),
            ),
        ),
        (
            "env_fail",
            10,
            (
                Step("a", "a"),
                Step("b", "b", env_ok=False),
                Step("b", "b"),
                Step("c", "c"),
            ),
        ),
        (
            "noncommit_block",
            10,
            (
                Step("b", "b"),
                Step("a", "a"),
                Step("b", "b"),
                Step("c", "c"),
            ),
        ),
        (
            "invalidation",
            5,
            (
                Step("a", "a"),
                Step("b", "b"),
                Step("drop_b", None, invalidates=("b",)),
                Step("c", "c"),
            ),
        ),
    ]
    for category, count, steps in plan:
        if len(steps) != 4:
            raise ValueError(f"{category} must have four steps")
        for index in range(count):
            episodes.append(_ep(category, index, steps))
    if len(episodes) != 100:
        raise ValueError(f"expected 100 episodes, got {len(episodes)}")
    return episodes
