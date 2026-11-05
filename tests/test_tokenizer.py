import ast
from pathlib import Path
import unittest

import tiktoken
import torch

import prepare
from tools.cpu_smoke import load_model_namespace
from tests.test_training_loader import cpu_allocations


SOURCE = Path(__file__).resolve().parents[1] / "train.py"


def tokenizer_type():
    node = next(node for node in ast.parse(SOURCE.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "Tokenizer")
    namespace = {"BaseTokenizer": prepare.Tokenizer}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["Tokenizer"]


def encoding():
    return tiktoken.Encoding(
        name="byte-test", pat_str=prepare.SPLIT_PATTERN,
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={name: 256 + index for index, name in enumerate(prepare.SPECIAL_TOKENS)},
    )


class SmallBatchTokenizerTests(unittest.TestCase):
    def setUp(self):
        enc = encoding()
        self.reference = prepare.Tokenizer(enc)
        self.candidate = tokenizer_type()(enc)

    def test_returns_native_ids_for_text_batches_unicode_and_each_bos_form(self):
        samples = ["Hello world!", "你好 🌍\n", "", prepare.BOS_TOKEN]
        inputs = [samples[0], [], samples[:1], samples[:2], samples, samples + ["extra"], samples * 32]
        for text in inputs:
            for prepend in (None, 0, self.reference.get_bos_token_id(), prepare.BOS_TOKEN):
                with self.subTest(batch=text if isinstance(text, str) else len(text), prepend=prepend):
                    expected = self.reference.encode(text, prepend=prepend)
                    self.assertEqual(self.candidate.encode(text, prepend=prepend), expected)
                    self.assertEqual(self.candidate(text, prepend=prepend), expected)
        self.assertEqual(self.candidate.get_vocab_size(), self.reference.get_vocab_size())
        self.assertEqual(self.candidate.get_bos_token_id(), self.reference.get_bos_token_id())
        self.assertEqual(self.candidate.decode(self.candidate.encode(samples[1])), samples[1])

    def test_respects_requested_thread_counts_and_validates_native_input_types(self):
        text = ["Alpha", "你好"]
        for threads in (None, 1, 2, 8):
            with self.subTest(threads=threads):
                self.assertEqual(self.candidate.encode(text, num_threads=threads),
                                 self.reference.encode(text, num_threads=threads))
        for threads in (0, -1):
            for tokenizer in (self.reference, self.candidate):
                with self.subTest(threads=threads, tokenizer=type(tokenizer).__name__):
                    with self.assertRaises(ValueError):
                        tokenizer.encode(text, num_threads=threads)
        for invalid in (123, ("hello",)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.candidate.encode(invalid)
        with self.assertRaises(TypeError):
            self.candidate.encode([123])

    def test_returns_independent_token_rows_and_batches_with_prepended_bos(self):
        expected = self.reference.encode(["same", "same"], prepend=prepare.BOS_TOKEN)
        actual = self.candidate.encode(["same", "same"], prepend=prepare.BOS_TOKEN)
        actual[0][0] = 999
        actual[0].append(1000)
        self.assertEqual(actual[1], expected[1])
        self.assertEqual(self.candidate.encode(["same", "same"], prepend=prepare.BOS_TOKEN), expected)

    def test_preserves_packed_training_inputs_targets_and_epochs(self):
        def documents(split):
            self.assertEqual(split, "train")
            text = ["", "a", "bb", "same", "same", "Hello world!", "你好 🌍"]
            for step in range(1000):
                yield (text[:2] if step % 2 == 0 else (text * 19)[:128]), step // 4 + 1

        namespace = load_model_namespace(SOURCE)
        namespace["_document_batches"] = documents
        loader = cpu_allocations("make_training_dataloader", SOURCE.read_text(), namespace)
        reference = loader(self.reference, 2, 16, buffer_size=8)
        candidate = loader(self.candidate, 2, 16, buffer_size=8)
        for _ in range(32):
            expected_x, expected_y, expected_epoch = next(reference)
            x, y, epoch = next(candidate)
            torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
            torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
            self.assertEqual(epoch, expected_epoch)


if __name__ == "__main__":
    unittest.main()
