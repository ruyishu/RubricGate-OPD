"""Checks the loss rules. Passing them does not show a training gain."""

from __future__ import annotations

import unittest

from research.loss_spec import (
    COMMIT_WEIGHT,
    READ_WEIGHT,
    StudentStep,
    agreement_bonus,
    kl_coefficient,
    rubric_delta,
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

    def test_progress_turns_teacher_kl_off(self) -> None:
        self.assertEqual(kl_coefficient("progress", committing=False), 0.0)
        self.assertEqual(kl_coefficient("progress", committing=True), 0.0)

    def test_neutral_steps_use_the_ablation_weight(self) -> None:
        for label in ("neutral_unknown", "neutral_repeat", "neutral_no_effect"):
            self.assertEqual(kl_coefficient(label, committing=False), 0.0)
            self.assertEqual(kl_coefficient(label, committing=False, neutral=0.1), 0.1)
        self.assertEqual(kl_coefficient("dependency_block", committing=False, neutral=0.1), 0.1)

    def test_confirmed_errors_use_full_kl(self) -> None:
        self.assertEqual(kl_coefficient("invalidation", committing=False), 1.0)
        self.assertEqual(kl_coefficient("dependency_block", committing=True), 1.0)

    def test_agreement_bonus_never_penalizes_a_mismatch_or_a_success(self) -> None:
        self.assertEqual(agreement_bonus("progress", False, matched=True, bonus=0.1), 0.0)
        self.assertEqual(agreement_bonus("neutral_unknown", False, matched=False, bonus=0.1), 0.0)
        self.assertEqual(agreement_bonus("invalidation", False, matched=False, bonus=0.1), 0.0)
        self.assertEqual(agreement_bonus("neutral_unknown", False, matched=True, bonus=0.1), 0.1)
        self.assertEqual(agreement_bonus("dependency_block", True, matched=True, bonus=0.1), 0.1)

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
