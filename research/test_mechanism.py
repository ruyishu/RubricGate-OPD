"""Invariant tests. Passing them does not validate the research claim."""

from __future__ import annotations

import unittest

from research.run_m0 import build_tables
from research.scorers import match_score, replay_gate, replay_score
from research.toyenv import TRUE_DAG, WRONG_DAG, Episode, Step, generate_episodes


def _one(category: str) -> Episode:
    for episode in generate_episodes():
        if episode.category == category:
            return episode
    raise AssertionError(category)


class MechanismTest(unittest.TestCase):
    def test_episode_count(self) -> None:
        episodes = generate_episodes()
        self.assertEqual(len(episodes), 100)
        self.assertTrue(all(len(episode.steps) == 4 for episode in episodes))

    def test_legal_swap_is_full_credit(self) -> None:
        rewards = replay_score(TRUE_DAG, _one("legal_swap"))
        self.assertEqual(rewards, [1.0, 1.0, 1.0, 1.0])

    def test_unknown_repeat_and_env_fail_are_neutral(self) -> None:
        self.assertEqual(replay_score(TRUE_DAG, _one("unknown"))[1], 0.0)
        self.assertEqual(replay_score(TRUE_DAG, _one("repeat"))[2], 0.0)
        self.assertEqual(replay_score(TRUE_DAG, _one("env_fail"))[1], 0.0)

    def test_agreement_does_not_override_env_failure(self) -> None:
        # Teacher's only legal action is b, so a same-state match would score 1.
        self.assertEqual(match_score("b", ["b"]), 1.0)
        self.assertEqual(replay_score(TRUE_DAG, _one("env_fail"))[1], 0.0)

    def test_missing_checklist_item_is_not_an_intervention(self) -> None:
        replay = replay_gate(TRUE_DAG, _one("unknown"))
        self.assertEqual(replay.decisions[1], "continue")
        self.assertEqual(replay.teacher_progress, 0.0)

    def test_noncommitting_block_continues(self) -> None:
        replay = replay_gate(TRUE_DAG, _one("noncommit_block"))
        self.assertEqual(replay.decisions[0], "continue")
        self.assertEqual(replay.student_return, 3.0)

    def test_committing_block_intervenes_without_crediting_teacher(self) -> None:
        episode = _one("committing_block")
        replay = replay_gate(TRUE_DAG, episode, max_interventions=1)
        self.assertEqual(replay.decisions, ["continue", "continue", "intervene", "continue"])
        self.assertEqual(replay.teacher_progress, 1.0)
        self.assertEqual(replay.student_return, 2.0)
        self.assertEqual(replay.student_return, sum(replay.student_step_rewards))
        self.assertEqual(replay.student_step_rewards[2], 0.0)

    def test_invalidation_is_the_only_negative_v1_signal(self) -> None:
        rewards = replay_score(TRUE_DAG, _one("invalidation"))
        self.assertEqual(rewards, [1.0, 1.0, -1.0, 1.0])

    def test_wrong_dag_downscores_a_legal_swap(self) -> None:
        episode = _one("legal_swap")
        self.assertEqual(replay_score(WRONG_DAG, episode)[1], 0.0)
        self.assertEqual(replay_score(TRUE_DAG, episode)[1], 1.0)

    def test_stop_does_not_keep_generating(self) -> None:
        replay = replay_gate(TRUE_DAG, _one("committing_block"), max_interventions=0)
        self.assertEqual(replay.decisions, ["continue", "continue", "stop"])
        self.assertTrue(replay.stopped_early)
        self.assertEqual(replay.teacher_progress, 0.0)

    def test_teacher_progress_is_not_inside_student_return(self) -> None:
        _, _, leaked, checked = build_tables()
        self.assertEqual(checked, 100)
        self.assertEqual(leaked, 0)

    def test_handmade_episode_rejects_pay_before_ready(self) -> None:
        episode = Episode(
            "pay",
            "manual",
            (Step("a", "a"), Step("pay", "d"), Step("b", "b"), Step("c", "c")),
        )
        replay = replay_gate(TRUE_DAG, episode)
        self.assertEqual(replay.decisions[1], "intervene")
        self.assertEqual(replay_score(TRUE_DAG, episode)[1], 0.0)


if __name__ == "__main__":
    unittest.main()
