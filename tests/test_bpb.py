import math
import unittest

import prepare


class BpbDecisionTests(unittest.TestCase):
    def test_keeps_only_a_lower_val_bpb(self):
        for candidate, incumbent, kept in [
            (0.97, 0.98, True),
            (0.974082, 0.974083, True),
            (0.974083, 0.974083, False),
            (0.99, 0.98, False),
        ]:
            with self.subTest(candidate=candidate, incumbent=incumbent):
                self.assertEqual(prepare.bpb_improves(candidate, incumbent), kept)

    def test_requires_finite_values_for_a_decision(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    prepare.bpb_improves(value, 0.98)
                with self.assertRaises(ValueError):
                    prepare.bpb_improves(0.98, value)


class SuiteDecisionTests(unittest.TestCase):
    incumbent = {"qa": 1.0, "math": 2.0, "code": 1.5}

    def test_keeps_a_candidate_whose_average_rate_of_improvement_is_positive(self):
        for candidate, kept in [
            ({"qa": 0.99, "math": 1.98, "code": 1.485}, True),
            ({"qa": 1.01, "math": 2.0, "code": 1.425}, True),
            ({"qa": 0.98, "math": 2.1, "code": 1.5}, False),
            ({"qa": 1.0, "math": 2.0, "code": 1.5}, False),
        ]:
            with self.subTest(candidate=candidate):
                self.assertEqual(prepare.suites_improve(candidate, self.incumbent), kept)

    def test_requires_the_same_suites_with_finite_positive_values(self):
        for candidate in (
            {"qa": 0.9, "math": 1.9},
            {"qa": 0.9, "math": 1.9, "code": 1.4, "extra": 1.0},
            {"qa": math.nan, "math": 1.9, "code": 1.4},
            {"qa": 0.0, "math": 1.9, "code": 1.4},
        ):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    prepare.suites_improve(candidate, self.incumbent)


if __name__ == "__main__":
    unittest.main()
