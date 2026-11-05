import unittest

import numpy as np
import torch
import torch.nn.functional as F

from tools.softmax_candidates import build_candidates


class SoftmaxCandidateTests(unittest.TestCase):
    def test_includes_every_true_target_and_preserves_order_duplicates_and_ignored_positions(self):
        targets = np.array([[1, 7, -1], [3, 1, 7]], dtype=np.int64)
        result = build_candidates(targets, 16, 8, offset=5)
        self.assertEqual(result.token_ids.shape, (8,))
        self.assertEqual(len(np.unique(result.token_ids)), 8)
        self.assertEqual(result.distinct_targets, 3)
        self.assertFalse(result.fallback)
        np.testing.assert_array_equal(result.token_ids, np.sort(result.token_ids))
        valid = targets != -1
        np.testing.assert_array_equal(result.token_ids[result.target_positions[valid]], targets[valid])
        np.testing.assert_array_equal(result.target_positions[~valid], [-1])
        self.assertTrue(np.all((result.token_ids >= 0) & (result.token_ids < 16)))

    def test_uses_the_full_vocabulary_when_the_width_cannot_cover_all_true_targets(self):
        targets = np.array([0, 2, 4, 6, -1])
        for width in (2, 8):
            with self.subTest(width=width):
                result = build_candidates(targets, 8, width)
                np.testing.assert_array_equal(result.token_ids, np.arange(8))
                np.testing.assert_array_equal(result.target_positions, targets)
                self.assertEqual(result.fallback, width == 2)

    def test_rotates_negatives_reproducibly_and_returns_independent_arrays(self):
        targets = np.array([0, 0])
        first = build_candidates(targets, 8, 2, offset=3)
        second = build_candidates(targets, 8, 2, offset=3)
        np.testing.assert_array_equal(first.token_ids, second.token_ids)
        first.token_ids[0] = 999
        np.testing.assert_array_equal(second.token_ids[second.target_positions], targets)
        offset = 0
        visited = []
        for _ in range(7):
            result = build_candidates(targets, 8, 2, offset=offset)
            visited.extend(token for token in result.token_ids if token != 0)
            offset = result.next_offset
        self.assertEqual(sorted(visited), list(range(1, 8)))

    def test_matches_the_declared_subset_normalizer_loss_and_gradients(self):
        torch.manual_seed(42)
        inputs = torch.randn(5, 6, requires_grad=True)
        weights = torch.randn(16, 6, requires_grad=True)
        targets = np.array([1, 3, -1, 7, 1])
        valid = torch.from_numpy(targets != -1)
        labels = torch.from_numpy(targets[targets != -1])
        for width in (8, 16):
            with self.subTest(width=width):
                candidate = build_candidates(targets, 16, width)
                indices = torch.from_numpy(candidate.token_ids)
                mapped = torch.from_numpy(candidate.target_positions)
                projected = F.linear(inputs, weights.index_select(0, indices))
                subset_logits = 15 * torch.tanh(projected.float() / 15)
                actual = F.cross_entropy(subset_logits, mapped, ignore_index=-1)
                dense_logits = 15 * torch.tanh(F.linear(inputs, weights).float() / 15)
                expected = (torch.logsumexp(dense_logits[valid][:, indices], dim=1)
                            - dense_logits[valid][torch.arange(len(labels)), labels]).mean()
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                actual_grad = torch.autograd.grad(actual, (inputs, weights), retain_graph=True)
                expected_grad = torch.autograd.grad(expected, (inputs, weights), retain_graph=True)
                for observed, reference in zip(actual_grad, expected_grad, strict=True):
                    torch.testing.assert_close(observed, reference, rtol=1e-5, atol=1e-6)
                full = F.cross_entropy(dense_logits[valid], labels)
                if width == 16:
                    torch.testing.assert_close(actual, full, rtol=1e-6, atol=1e-6)
                else:
                    self.assertLessEqual(actual.item(), full.item())

    def test_preserves_all_ignored_targets_and_requires_valid_vocabulary_inputs(self):
        result = build_candidates(np.array([-1, -1]), 8, 4)
        np.testing.assert_array_equal(result.target_positions, [-1, -1])
        self.assertEqual(result.distinct_targets, 0)
        self.assertEqual(len(result.token_ids), 4)
        for targets, vocabulary, width, stride in (
                ([-2], 8, 4, 20011), ([8], 8, 4, 20011),
                ([1.5], 8, 4, 20011), ([1], 8, 0, 20011),
                ([1], 8, 9, 20011), ([1], 8, 4, 2)):
            with self.subTest(targets=targets, vocabulary=vocabulary, width=width, stride=stride):
                with self.assertRaises(ValueError):
                    build_candidates(targets, vocabulary, width, stride=stride)


if __name__ == "__main__":
    unittest.main()
