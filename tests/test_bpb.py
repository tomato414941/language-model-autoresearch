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


if __name__ == "__main__":
    unittest.main()
