"""Run the scripted mechanism battery and write a small result table.

A high action-match downscore under a peaked scripted teacher is produced by
that script. It is not an estimate of a real on-policy teacher.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from research.scorers import (
    flat_reward,
    replay_gate,
    replay_potential,
    replay_score,
    teacher_probs,
)
from research.toyenv import TRUE_DAG, WRONG_DAG, classify, generate_episodes, transition, unlocked


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m0-mechanism-001"
FULL_CREDIT = {
    "rubricgate_v1": 1.0,
    "potential_delta": 0.25,
    "flat_checklist": 1.0,
    "action_match": 1.0,
}


def _situation(done: frozenset[str]) -> str:
    if set(unlocked(TRUE_DAG, done)) == {"b", "c"}:
        return "open_bc"
    return "other"


def _add(grouped: dict[tuple[str, str, str], list[float]], episode, oracle: str, situation: str, reward: float) -> None:
    grouped[(episode.category, oracle, situation)].append(reward)


def _graph_rows(episodes, dag, setting: str, scorer: str, replay) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for episode in episodes:
        done = frozenset()
        for step, reward in zip(episode.steps, replay(dag, episode)):
            _add(grouped, episode, classify(TRUE_DAG, done, step), _situation(done), reward)
            done, _ = transition(TRUE_DAG, done, step)
    return [_summarize(setting, scorer, key, values) for key, values in sorted(grouped.items())]


def _flat_rows(episodes) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for episode in episodes:
        done = frozenset()
        for step in episode.steps:
            _add(grouped, episode, classify(TRUE_DAG, done, step), _situation(done), flat_reward(TRUE_DAG, done, step))
            done, _ = transition(TRUE_DAG, done, step)
    return [_summarize("true_dag", "flat_checklist", key, values) for key, values in sorted(grouped.items())]


def _match_rows(episodes, prefer_b: float, k: int) -> list[dict[str, str]]:
    """Expected frequency reward. k does not change the mean; it changes P(reward=0)."""

    grouped: dict[tuple[str, str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for episode in episodes:
        done = frozenset()
        for step in episode.steps:
            probs = teacher_probs(unlocked(TRUE_DAG, done), prefer_b)
            probability = 0.0 if step.target is None else probs.get(step.target, 0.0)
            zero_rate = (1.0 - probability) ** k
            below_full = 1.0 - (probability ** k)
            grouped[(episode.category, classify(TRUE_DAG, done, step), _situation(done))].append(
                (probability, zero_rate, below_full)
            )
            done, _ = transition(TRUE_DAG, done, step)
    setting = f"expected_p{prefer_b:.2f}_k{k}"
    rows = []
    for key, values in sorted(grouped.items()):
        category, label, situation = key
        n = len(values)
        rows.append(
            {
                "setting": setting,
                "scorer": "action_match",
                "category": category,
                "oracle_label": label,
                "situation": situation,
                "n_steps": str(n),
                "mean_reward": f"{sum(item[0] for item in values) / n:.4f}",
                "negative_rate": "0.0000",
                "positive_rate": f"{sum(item[0] > 0 for item in values) / n:.4f}",
                "below_full_credit_rate": f"{sum(item[2] for item in values) / n:.4f}",
                "zero_rate": f"{sum(item[1] for item in values) / n:.4f}",
            }
        )
    return rows


def _summarize(setting: str, scorer: str, key: tuple[str, str, str], values: list[float]) -> dict[str, str]:
    category, label, situation = key
    n = len(values)
    full = FULL_CREDIT[scorer]
    mean = sum(values) / n
    return {
        "setting": setting,
        "scorer": scorer,
        "category": category,
        "oracle_label": label,
        "situation": situation,
        "n_steps": str(n),
        "mean_reward": f"{mean:.4f}",
        "negative_rate": f"{sum(v < 0 for v in values) / n:.4f}",
        "positive_rate": f"{sum(v > 0 for v in values) / n:.4f}",
        "below_full_credit_rate": f"{sum(v < full - 1e-9 for v in values) / n:.4f}",
        "zero_rate": f"{sum(abs(v) < 1e-9 for v in values) / n:.4f}",
    }


def _same_sign(left: list[dict[str, str]], right: list[dict[str, str]]) -> bool:
    def sign(row: dict[str, str]) -> int:
        value = float(row["mean_reward"])
        if value > 1e-9:
            return 1
        if value < -1e-9:
            return -1
        return 0

    index = {
        (row["category"], row["oracle_label"], row["situation"]): sign(row)
        for row in right
    }
    return all(
        sign(row) == index[(row["category"], row["oracle_label"], row["situation"])]
        for row in left
    )


def _gate_rows(episodes, dag, max_interventions: int):
    counts: dict[tuple[str, str], int] = defaultdict(int)
    leaked = 0
    for episode in episodes:
        replay = replay_gate(dag, episode, max_interventions=max_interventions)
        if abs(replay.student_return - sum(replay.student_step_rewards)) > 1e-9:
            leaked += 1
        for decision, reward in zip(replay.decisions, replay.student_step_rewards):
            counts[(episode.category, decision)] += 1
            if decision == "intervene" and reward > 0:
                leaked += 1
    rows = [
        {"category": category, "decision": decision, "count": str(count)}
        for (category, decision), count in sorted(counts.items())
    ]
    return rows, leaked, len(episodes)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    headers = list(rows[0])
    lines = [",".join(headers)]
    lines.extend(",".join(row[header] for header in headers) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find(rows: list[dict[str, str]], **wanted: str) -> dict[str, str]:
    matches = [row for row in rows if all(row[key] == value for key, value in wanted.items())]
    if len(matches) != 1:
        raise KeyError((wanted, len(matches)))
    return matches[0]


def build_tables():
    episodes = generate_episodes()
    v1 = _graph_rows(episodes, TRUE_DAG, "true_dag", "rubricgate_v1", replay_score)
    potential = _graph_rows(episodes, TRUE_DAG, "true_dag", "potential_delta", replay_potential)
    if not _same_sign(v1, potential):
        raise RuntimeError("v1 and potential delta disagree in sign on the true DAG")
    rows = [
        *v1,
        *potential,
        *_graph_rows(episodes, WRONG_DAG, "wrong_dag_c_requires_b", "rubricgate_v1", replay_score),
        *_flat_rows(episodes),
    ]
    for prefer_b, k in ((0.5, 1), (0.95, 1), (0.5, 3), (0.95, 3)):
        rows.extend(_match_rows(episodes, prefer_b, k))
    gate_rows, leaked, checked = _gate_rows(episodes, TRUE_DAG, max_interventions=1)
    return rows, gate_rows, leaked, checked


def _notes(rows: list[dict[str, str]], leaked: int, checked: int) -> str:
    peaked = _find(
        rows,
        setting="expected_p0.95_k1",
        scorer="action_match",
        category="legal_swap",
        situation="open_bc",
    )
    peaked_k3 = _find(
        rows,
        setting="expected_p0.95_k3",
        scorer="action_match",
        category="legal_swap",
        situation="open_bc",
    )
    neutral = _find(
        rows,
        setting="expected_p0.50_k3",
        scorer="action_match",
        category="legal_swap",
        situation="open_bc",
    )
    preferred = _find(
        rows,
        setting="expected_p0.95_k1",
        scorer="action_match",
        category="canonical",
        situation="open_bc",
    )
    unknown = _find(
        rows,
        setting="true_dag",
        scorer="rubricgate_v1",
        category="unknown",
        oracle_label="neutral_unknown",
    )
    blocked = _find(
        rows,
        setting="true_dag",
        scorer="rubricgate_v1",
        category="committing_block",
        oracle_label="dependency_block",
    )
    no_effect = _find(
        rows,
        setting="true_dag",
        scorer="rubricgate_v1",
        category="env_fail",
        oracle_label="neutral_no_effect",
    )
    flat_block = _find(
        rows,
        setting="true_dag",
        scorer="flat_checklist",
        category="committing_block",
        oracle_label="dependency_block",
    )
    wrong = _find(
        rows,
        setting="wrong_dag_c_requires_b",
        scorer="rubricgate_v1",
        category="legal_swap",
        situation="open_bc",
    )
    return f"""# M0 机制检查

命令：`python -m research.run_m0`

100 条手写模板轨迹，每条 4 步。`open_bc` 表示这一步开始时 b 和 c 都已解锁。标签来自正确 DAG 和环境结果。动作匹配行是脚本分布下的期望，不是抽样结果。

## 这张表能说明什么

- 正确 DAG 上，RubricGate v1 和势能差的奖励符号一致：新完成的子目标为正，破坏已完成子目标为负，其余为 0。势能差没有实现 DARS 的下游衰减，不能当成 DARS 复现。
- 清单外动作平均奖励 {unknown["mean_reward"]}，负奖励比例 {unknown["negative_rate"]}。未解锁的付款步骤平均奖励 {blocked["mean_reward"]}，负奖励比例 {blocked["negative_rate"]}。工具名对了但环境没生效时平均奖励 {no_effect["mean_reward"]}。
- 平铺清单给未解锁付款步骤正奖励的比例是 {flat_block["positive_rate"]}。
- 把 c 错标成必须在 b 之后后，合法换序那一步的平均奖励是 {wrong["mean_reward"]}。图错了，打分就跟着错。
- 动作匹配的期望奖励等于学生动作在教师分布中的概率，候选数 k 不改变这个期望。b/c 都可做且教师以 0.95 偏好 b 时，老师那一步的期望奖励是 {preferred["mean_reward"]}，合法换序那一步是 {peaked["mean_reward"]}。k=1 时这步奖励为 0 的概率是 {peaked["zero_rate"]}；k=3 时期望仍是 {peaked_k3["mean_reward"]}，奖励为 0 的概率是 {peaked_k3["zero_rate"]}。偏好降到 0.50 且 k=3 时，期望奖励是 {neutral["mean_reward"]}，奖励为 0 的概率是 {neutral["zero_rate"]}。
- 接管记账：{checked} 条轨迹里，学生回报与逐步学生奖励不一致、或接管步被记成正奖励的次数是 {leaked}。教师代打完成的子目标记在 `teacher_progress`，没有加进学生回报。

## 这张表不能说明什么

脚本偏好把「合法换序匹配分低」写进了分布，所以那个差距不能证明真实 OPD 存在同样严重的误罚。本表也没有训练，不能证明少误罚或多出来的接管能提高成功率。
"""


def main() -> None:
    rows, gate_rows, leaked, checked = build_tables()
    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT / "metrics.csv", rows)
    _write_csv(OUT / "gate_counts.csv", gate_rows)
    (OUT / "notes.md").write_text(_notes(rows, leaked, checked), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
