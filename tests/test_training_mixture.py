from gpu_config import active_gpu
from contextlib import ExitStack, closing
import gzip
from itertools import count, islice
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

import prepare
from tests.test_training_loader import cpu_allocations, ordered_batches, ToyTokenizer
from tools.cpu_smoke import load_model_namespace


class TrainingMixtureTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_model_namespace()

    def test_reads_every_math_and_code_document_including_empty_duplicates_and_unicode(self):
        rows = ["one", "", "日本語🙂", "one", "last"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subset in ("finemath-3plus", "finemath-4plus"):
                math = root/subset
                math.mkdir()
                pq.write_table(pa.table({"text":rows}), math/"train.parquet", row_group_size=2)
            code = root/"code"
            code.mkdir()
            with gzip.open(code/"original.json.gz", "wt", encoding="utf-8") as output:
                for text in rows:
                    output.write(json.dumps(dict(text=text, license="MIT"))+"\n")
            with patch.object(prepare, "FINEMATH_DIR", str(root)), \
                    patch.object(prepare, "COMMON_PILE_DIR", str(code)):
                for source in ("finemath-3plus", "finemath-4plus", "common-pile"):
                    with self.subTest(source=source), closing(
                            self.namespace['_extra_training_document_batches'](source, batch_size=2)) as stream:
                        first = list(islice(stream, 3))
                        self.assertEqual(first, [(rows[:2],1), (rows[2:4],1), (rows[4:],1)])
                        self.assertEqual(next(stream), (rows[:2],2))

    def test_cycles_requested_sources_and_preserves_each_readers_order_and_epoch(self):
        def stream(name):
            index = 0
            while True:
                yield [f"{name}-{index}", ""], index//2+1
                index += 1
        self.namespace['_document_batches'] = lambda split: stream("fineweb")
        self.namespace['_extra_training_document_batches'] = stream
        pattern = ("fineweb",)*8+("finemath-4plus", "common-pile")
        batches = list(islice(self.namespace['_training_document_batches'](pattern), 20))
        self.assertEqual([batch[0][0].rsplit('-',1)[0] for batch in batches], list(pattern)*2)
        self.assertEqual([batch[0][0] for batch in batches if batch[0][0].startswith("fineweb-")],
                         [f"fineweb-{index}" for index in range(16)])
        self.assertEqual(batches[0][1]['fineweb'], 1)
        self.assertEqual(batches[-1][1], dict(fineweb=8, **{'finemath-4plus':1,'common-pile':1}))

    def test_preserves_the_native_fineweb_only_document_interface(self):
        expected = [(["a", ""],1), (["b"],2)]
        self.namespace['_document_batches'] = lambda split: iter(expected)
        for pattern in (None, ("fineweb",), ("fineweb",)*8):
            self.assertEqual(list(self.namespace['_training_document_batches'](pattern)), expected)

    def test_trains_from_contiguous_source_rows_with_their_original_tokens_and_epochs(self):
        batches = {"fineweb":[[1]*7, [2]*2], "finemath-4plus":[[3]*10, []],
                   "common-pile":[[4]*12, [5]*3]}
        def stream(name):
            for epoch in count(1):
                yield batches[name], epoch
        self.namespace['_document_batches'] = lambda split: stream("fineweb")
        self.namespace['_extra_training_document_batches'] = stream
        source = active_gpu().train_source().read_text()
        loader_fn = cpu_allocations('make_training_dataloader', source, self.namespace)
        pattern = ("fineweb", "finemath-4plus", "common-pile")
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](sequence_len=17, vocab_size=32, n_layer=4,
                                            n_head=2, n_kv_head=2, n_embd=64)
        model = self.namespace['GPT'](config)
        model.init_weights()
        optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02)
        with ExitStack() as stack:
            loader = stack.enter_context(closing(loader_fn(ToyTokenizer(), 2, 17, source_pattern=pattern)))
            references = {name: stack.enter_context(closing(ordered_batches(
                1, 17, lambda split, name=name: stream(name)))) for name in pattern}
            expected_epochs = dict.fromkeys(pattern, 1)
            position = 0
            for _ in range(3):
                inputs, targets, epochs = next(loader)
                for row in range(2):
                    name = pattern[position % len(pattern)]
                    expected_inputs, expected_targets, epoch = next(references[name])
                    torch.testing.assert_close(inputs[row], expected_inputs[0], rtol=0, atol=0)
                    torch.testing.assert_close(targets[row], expected_targets[0], rtol=0, atol=0)
                    expected_epochs[name] = epoch
                    position += 1
                self.assertEqual(epochs, expected_epochs)
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    loss = model(inputs,targets)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in model.parameters()))
                optimizer.step()
                model.zero_grad(set_to_none=True)


if __name__ == '__main__':
    unittest.main()
