from gpu_config import active_gpu
from collections import Counter
import math
from pathlib import Path
import unittest

import torch
from oe_eval.components.requests import LoglikelihoodRequest, LoglikelihoodRollingRequest
from oe_eval.models.eleuther_huggingface import HFLM_Verbose

from evaluation.count_tokens import TokenObserver, TokensObserved
from evaluation.tokenizer_adapter import TokenizerAdapter


class ForwardProbe(TokenObserver):
    """Exercise the native inference path with synthetic zero logits on the CPU."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.shapes = []
        self._shown_model_input = True

    def _loglikelihood_tokens_verbose(self, requests, **kwargs):
        return HFLM_Verbose._loglikelihood_tokens_verbose(self, requests, **kwargs)

    def _model_call(self, inputs, **kwargs):
        self.shapes.append(tuple(inputs.shape))
        return torch.zeros(*inputs.shape, self.vocab_size, dtype=torch.float32)


class NativeTokenCountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = active_gpu().cache_directory("tokenizer") / "tokenizer.pkl"
        cls.tokenizer = TokenizerAdapter(path)
        torch.set_num_threads(2)

    def compare_native_shapes(self, requests, rolling=False):
        observer = TokenObserver(self.tokenizer, max_length=64, batch_size=2)
        probe = ForwardProbe(self.tokenizer, max_length=64, batch_size=2)
        method = "loglikelihood_rolling_verbose" if rolling else "loglikelihood_verbose"
        with self.assertRaises(TokensObserved):
            getattr(observer, method)(requests, disable_tqdm=True)
        native_results = getattr(probe, method)(requests, disable_tqdm=True)
        self.assertEqual(len(native_results), len(requests))
        expected = Counter(probe.shapes)
        actual = Counter({(item["batch"], item["length"]): item["count"]
                          for item in observer.statistics()["forward_shapes"]})
        self.assertEqual(actual, expected)
        self.assertAlmostEqual(observer.scored_tokens,
                               -sum(result["sum_logits"] for result in native_results) / math.log(len(self.tokenizer)),
                               places=4)
        return observer, probe

    def test_counts_native_pair_encoding_cache_groups_and_right_padded_batches(self):
        self.compare_native_shapes([
            LoglikelihoodRequest("A question: ", "one"),
            LoglikelihoodRequest("A question: ", "two"),
            LoglikelihoodRequest("A question: ", "one"),
            LoglikelihoodRequest("", "An answer."),
            LoglikelihoodRequest("日本語の質問: ", "答えです。"),
            LoglikelihoodRequest("long context " * 70, "final answer"),
        ])

    def test_counts_native_rolling_windows_and_masked_targets(self):
        self.compare_native_shapes([
            LoglikelihoodRollingRequest("long rolling text " * 70),
            LoglikelihoodRollingRequest("masked prompt and scored answer", [(0, 17)]),
            LoglikelihoodRollingRequest("日本語の問題と答え。🙂", [(0, 6)]),
        ], rolling=True)

    def test_maps_unicode_token_bytes_to_the_original_character_spans(self):
        text = "a日本語🙂é\n"
        output = self.tokenizer.encode_plus(text, return_offsets_mapping=True)
        self.assertEqual(self.tokenizer.decode(output["input_ids"]), text)
        byte_position = 0
        boundaries = [0]
        for character in text:
            boundaries.append(boundaries[-1] + len(character.encode("utf-8")))
        for token, (start, end) in zip(output["input_ids"], output["offset_mapping"], strict=True):
            raw = self.tokenizer.encoding.decode_single_token_bytes(token)
            self.assertLessEqual(boundaries[start], byte_position)
            self.assertGreater(boundaries[start + 1], byte_position)
            self.assertGreaterEqual(boundaries[end], byte_position + len(raw))
            self.assertLess(boundaries[end - 1], byte_position + len(raw))
            byte_position += len(raw)

    def test_encodes_special_looking_text_literally_and_adds_bos_only_when_requested(self):
        text = "literal <|reserved_0|> text"
        encoded = self.tokenizer.encode(text)
        self.assertEqual(self.tokenizer.decode(encoded), text)
        self.assertEqual(self.tokenizer.encode(text, add_special_tokens=True),
                         [self.tokenizer.bos_token_id] + encoded)


if __name__ == "__main__":
    unittest.main()
