import ast
import inspect
from pathlib import Path
import unittest

import torch

import prepare
from tools.cpu_smoke import load_model_namespace


def cpu_allocations(name, source, namespace):
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    tree = ast.Module(body=[function], type_ignores=[])
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == 'pin_memory':
            node.value = ast.Constant(False)
        elif (isinstance(node, ast.keyword) and node.arg == 'device'
              and isinstance(node.value, ast.Constant) and node.value.value == 'cuda'):
            node.value = ast.Constant('cpu')
    exec(compile(ast.fix_missing_locations(tree), '<cpu_loader_allocations>', 'exec'), namespace)
    return namespace[name]


class ToyTokenizer:
    def __init__(self):
        self.calls = []

    def get_bos_token_id(self):
        return 0

    def encode(self, documents, prepend):
        self.calls.append(tuple(tuple(document) for document in documents))
        return [[prepend] + list(document) for document in documents]


def documents(split):
    assert split == 'train'
    sizes = (0, 1, 2, 14, 15, 16, 17, 30, 31, 63, 64, 65, 130, 15, 31)
    for batch in range(10000):
        rows = [[1 + (batch * 19 + index) % 63] * size for index, size in enumerate(sizes)]
        yield rows, batch // 4 + 1


class TrainingLoaderTests(unittest.TestCase):
    def test_selects_the_largest_fit_with_fifo_ties_and_the_shortest_overflow(self):
        namespace = load_model_namespace()
        index = namespace['LengthIndexedDocuments']()
        first = [10] * 5
        second = [11] * 5
        shortest = [12] * 2
        longest = [13] * 9
        index.extend([first, shortest, second, longest])
        self.assertIs(index.pop_best(7), first)
        later = [14] * 5
        index.extend([later])
        self.assertIs(index.pop_best(5), second)
        self.assertIs(index.pop_best(1), shortest)
        self.assertIs(index.pop_best(8), later)
        self.assertIs(index.pop_best(2), longest)
        self.assertEqual(len(index), 0)

    def test_matches_native_inputs_targets_and_epochs_across_refills_crops_and_empty_texts(self):
        for batch, length, buffer in ((2, 15, 8), (4, 16, 23), (3, 63, 17)):
            with self.subTest(batch=batch, length=length, buffer=buffer):
                namespace = load_model_namespace()
                namespace['_document_batches'] = documents
                candidate = cpu_allocations('make_training_dataloader',
                                            (Path(__file__).resolve().parents[1] / 'train.py').read_text(),
                                            namespace)
                reference_namespace = dict(torch=torch, _document_batches=documents)
                reference = cpu_allocations('make_dataloader', inspect.getsource(prepare.make_dataloader),
                                            reference_namespace)
                actual_tokenizer, reference_tokenizer = ToyTokenizer(), ToyTokenizer()
                actual = candidate(actual_tokenizer, batch, length, buffer_size=buffer)
                expected = reference(reference_tokenizer, batch, length, 'train', buffer_size=buffer)
                for _ in range(12):
                    x, y, epoch = next(actual)
                    expected_x, expected_y, expected_epoch = next(expected)
                    torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
                    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
                    self.assertEqual(epoch, expected_epoch)
                    self.assertTrue(torch.all(x[:, 0] == 0))
                self.assertEqual(actual_tokenizer.calls, reference_tokenizer.calls)

    def test_trains_the_current_model_from_packed_batches_and_returns_the_declared_average(self):
        namespace = load_model_namespace()
        namespace['_document_batches'] = documents
        candidate = cpu_allocations('make_training_dataloader',
                                    (Path(__file__).resolve().parents[1] / 'train.py').read_text(),
                                    namespace)
        loader = candidate(ToyTokenizer(), 2, 17, buffer_size=23)
        torch.manual_seed(42)
        config = namespace['GPTConfig'](
            sequence_len=16, vocab_size=64, n_layer=2, n_head=5, n_kv_head=1, n_embd=80)
        with torch.device('meta'):
            model = namespace['GPT'](config)
        model.to_empty(device='cpu')
        model.init_weights()
        optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02)
        average = namespace['TailEMA'](model, start=0.8, tau=0.075, blend=0.6)
        compiled = torch.compile(model, backend='aot_eager', fullgraph=True, dynamic=False)
        for step in range(2):
            for _ in range(2):
                inputs, targets, epoch = next(loader)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = compiled(inputs, targets) / 2
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
            for parameter in model.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
            model.zero_grad(set_to_none=True)
            average.update(0.8 + step * 0.1)
        average.apply()
        compiled.eval()
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            logits = compiled(inputs)
            losses = compiled(inputs, targets, reduction='none')
        self.assertEqual(logits.shape, (2, 17, 64))
        self.assertEqual(losses.numel(), targets.numel())
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(losses).all())


if __name__ == '__main__':
    unittest.main()
