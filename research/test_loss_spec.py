"""Checks the loss rules. Passing them does not show a training gain."""

from __future__ import annotations

import unittest

from research.loss_spec import (
    COMMIT_WEIGHT,
    READ_WEIGHT,
    StudentStep,
    candidate_reward,
    consults_teacher,
    kl_coefficient,
    rubric_delta,
    stall_flags,
    student_returns,
)


class LossSpecTest(unittest.TestCase):
    def test_read_items_weigh_less_than_commit_items(self) -> None:
        reward = rubric_delta({"looked_up": READ_WEIGHT}, {})
        commit = rubric_delta({"paid": COMMIT_WEIGHT}, {})
        undone = rubric_delta({}, {"looked_up": READ_WEIGHT})
        self.assertEqual(reward, 0.2)
        self.assertEqual(commit, 1.0)
        self.assertEqual(undone, -0.2)

    def test_stall_needs_a_run_of_unchanged_steps(self) -> None:
        changed = [True, False, False, False, True, False]
        self.assertEqual(stall_flags(changed, k=3), [False, False, False, True, False, False])

    def test_progress_turns_teacher_kl_off(self) -> None:
        self.assertEqual(kl_coefficient("progress", committing=False), 0.0)
        self.assertEqual(kl_coefficient("progress", committing=True, stalled=True), 0.0)

    def test_isolated_no_change_steps_use_the_ablation_weight(self) -> None:
        for label in ("neutral_unknown", "neutral_repeat", "neutral_no_effect"):
            self.assertEqual(kl_coefficient(label, committing=False), 0.0)
            self.assertEqual(kl_coefficient(label, committing=False, neutral=0.1), 0.1)
        self.assertEqual(kl_coefficient("dependency_block", committing=False, neutral=0.1), 0.1)

    def test_stalled_steps_pull_toward_the_teacher(self) -> None:
        self.assertEqual(kl_coefficient("neutral_unknown", committing=False, stalled=True), 1.0)
        self.assertEqual(kl_coefficient("dependency_block", committing=False, stalled=True), 1.0)

    def test_confirmed_errors_use_full_kl(self) -> None:
        self.assertEqual(kl_coefficient("invalidation", committing=False), 1.0)
        self.assertEqual(kl_coefficient("dependency_block", committing=True), 1.0)

    def test_teacher_is_consulted_only_on_errors_and_stalls(self) -> None:
        self.assertFalse(consults_teacher("progress", False, stalled=True))
        self.assertFalse(consults_teacher("neutral_unknown", False, stalled=False))
        self.assertFalse(consults_teacher("dependency_block", False, stalled=False))
        self.assertTrue(consults_teacher("neutral_unknown", False, stalled=True))
        self.assertTrue(consults_teacher("invalidation", False, stalled=False))
        self.assertTrue(consults_teacher("dependency_block", True, stalled=False))

    def test_valid_alternative_is_not_ranked_below_a_teacher_match(self) -> None:
        alternative = candidate_reward(progressed=True, broke=False, agreement=0.0)
        teacher_like = candidate_reward(progressed=False, broke=False, agreement=1.0)
        self.assertGreaterEqual(alternative, teacher_like)
        self.assertEqual(candidate_reward(progressed=False, broke=False, agreement=0.25), 0.25)
        self.assertEqual(candidate_reward(progressed=True, broke=True, agreement=1.0), 0.0)

    def test_handoff_hides_later_return_from_the_replaced_action(self) -> None:
        steps = [
            StudentStep(1.0, "continue"),
            StudentStep(1.0, "continue"),
            StudentStep(-1.0, "intervene"),
            StudentStep(1.0, "continue"),
        ]
        returns = student_returns(steps, outcome=1.0, gamma=1.0)
        self.assertEqual(returns, [2.0, 1.0, -1.0, 2.0])

    def test_outcome_is_not_discounted_an_extra_step(self) -> None:
        steps = [StudentStep(0.0, "continue"), StudentStep(0.0, "continue")]
        self.assertEqual(student_returns(steps, outcome=1.0, gamma=0.5), [0.5, 1.0])

    def test_stop_keeps_the_prefix_on_the_student_trajectory(self) -> None:
        steps = [
            StudentStep(1.0, "continue"),
            StudentStep(-1.0, "stop"),
        ]
        self.assertEqual(student_returns(steps, outcome=0.0), [0.0, -1.0])


if __name__ == "__main__":
    unittest.main()
