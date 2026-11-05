import gzip
from itertools import islice
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

import prepare
from tests.test_training_loader import cpu_allocations, ToyTokenizer
from tools.cpu_smoke import load_model_namespace


class TrainingMixtureTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_model_namespace()

    def test_reads_every_math_and_code_document_including_empty_duplicates_and_unicode(self):
        rows = ["one", "", "日本語🙂", "one", "last"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            math = root/"finemath-4plus"
            math.mkdir()
            pq.write_table(pa.table({"text":rows}), math/"train.parquet", row_group_size=2)
            code = root/"code"
            code.mkdir()
            with gzip.open(code/"original.json.gz", "wt", encoding="utf-8") as output:
                for text in rows:
                    output.write(json.dumps(dict(text=text, license="MIT"))+"\n")
            with patch.object(prepare, "FINEMATH_DIR", str(root)), \
                    patch.object(prepare, "COMMON_PILE_DIR", str(code)):
                for source in ("finemath-4plus", "common-pile"):
                    stream = self.namespace['_extra_training_document_batches'](source, batch_size=2)
                    first = list(islice(stream, 3))
                    self.assertEqual(first, [(rows[:2],1), (rows[2:4],1), (rows[4:],1)])
                    self.assertEqual(next(stream), (rows[:2],2))

    def test_follows_the_predeclared_cycle_and_preserves_each_sources_order_and_epoch(self):
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

    def test_packs_bos_prefixed_documents_from_every_source_and_trains_the_current_model(self):
        batches = {"fineweb":[[1]*7, [2]*2], "finemath-4plus":[[3]*10, []],
                   "common-pile":[[4]*12, [5]*3]}
        def stream(name):
            while True:
                yield batches[name], 1
        self.namespace['_document_batches'] = lambda split: stream("fineweb")
        self.namespace['_extra_training_document_batches'] = stream
        source = (Path(__file__).resolve().parents[1]/"train.py").read_text()
        loader_fn = cpu_allocations('make_training_dataloader', source, self.namespace)
        loader = loader_fn(ToyTokenizer(), 2, 17, buffer_size=5,
                           source_pattern=("fineweb","finemath-4plus","common-pile"))
        torch.manual_seed(42)
        config = self.namespace['GPTConfig'](sequence_len=17, vocab_size=32, n_layer=4,
                                            n_head=2, n_kv_head=2, n_embd=64)
        model = self.namespace['GPT'](config)
        model.init_weights()
        optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02)
        seen = set()
        for _ in range(3):
            inputs, targets, epochs = next(loader)
            self.assertTrue(torch.equal(inputs[:,1:],targets[:,:-1]))
            self.assertTrue(torch.equal(inputs[:,0],torch.zeros(2,dtype=torch.long)))
            seen.update(inputs.reshape(-1).tolist())
            self.assertEqual(set(epochs),set(batches))
            with torch.autocast('cpu',dtype=torch.bfloat16):
                loss = model(inputs,targets)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()))
            optimizer.step()
            model.zero_grad(set_to_none=True)
        self.assertTrue({2,3,4} <= seen)


if __name__ == '__main__':
    unittest.main()
