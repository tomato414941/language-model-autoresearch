from gpu_config import active_gpu
import ast
from contextlib import closing
from itertools import islice
import types
import unittest

import torch

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


def ordered_batches(batch, length, reader=documents):
    """Construct shifted targets independently from the original document tokens."""
    def tokens():
        for rows, epoch in reader('train'):
            for row in rows:
                for token in (0, *row):
                    yield token, epoch

    stream = tokens()
    previous, _ = next(stream)
    try:
        while True:
            tagged = list(islice(stream, batch * length))
            if len(tagged) < batch * length:
                return
            values = [token for token, _ in tagged]
            inputs = torch.tensor([previous, *values[:-1]], dtype=torch.long).view(batch, length)
            targets = torch.tensor(values, dtype=torch.long).view(batch, length)
            previous = values[-1]
            yield inputs, targets, tagged[-1][1]
    finally:
        stream.close()


class DeferredStream:
    """Model FIFO device work while host writes remain immediate."""

    def __init__(self):
        self.pending = []
        self.submitted = 0
        self.observed = []

    def submit(self, function):
        self.submitted += 1
        self.pending.append((self.submitted, function))
        return self.submitted

    def synchronize(self, boundary=None):
        while self.pending and (boundary is None or self.pending[0][0] <= boundary):
            _, function = self.pending.pop(0)
            function()

    def consume(self, inputs, targets, epoch):
        self.submit(lambda: self.observed.append((inputs.clone(), targets.clone(), epoch)))

    def event(self):
        stream = self

        class Event:
            def __init__(self):
                self.boundary = None

            def record(self):
                self.boundary = stream.submit(lambda: None)

            def synchronize(self):
                stream.synchronize(self.boundary)

        return Event()

    def torch_proxy(self):
        stream = self

        class DeviceBuffer:
            is_cuda = True

            def __init__(self, *args, **kwargs):
                self.values = torch.empty(*args, **kwargs)

            def __getitem__(self, item):
                return self.values[item]

            def copy_(self, source, non_blocking=False):
                stream.submit(lambda: self.values.copy_(source))
                if not non_blocking:
                    stream.synchronize()
                return self

        def empty(*args, **kwargs):
            device = kwargs.pop('device', None)
            kwargs.pop('pin_memory', None)
            if device == 'cuda':
                return DeviceBuffer(*args, **kwargs)
            return torch.empty(*args, **kwargs)

        return types.SimpleNamespace(empty=empty, tensor=torch.tensor, long=torch.long,
                                     cuda=types.SimpleNamespace(Event=self.event, synchronize=self.synchronize))


def deferred_training_batches(source):
    namespace = load_model_namespace()
    stream = DeferredStream()
    namespace.update(torch=stream.torch_proxy(), _document_batches=documents)
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'make_training_dataloader')
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<deferred_training_loader>', 'exec'), namespace)
    with closing(namespace['make_training_dataloader'](ToyTokenizer(), 2, 17)) as loader:
        inputs, targets, epoch = next(loader)
        # Match the training loop's initial synchronization and two accumulation passes.
        for _ in range(2):
            stream.synchronize()
            for _ in range(2):
                stream.consume(inputs, targets, epoch)
                inputs, targets, epoch = next(loader)
            stream.synchronize()
    return stream.observed


class TrainingLoaderTests(unittest.TestCase):
    def test_preserves_each_training_batch_when_nonblocking_transfers_are_deferred(self):
        with closing(ordered_batches(2, 17)) as reference:
            expected = list(islice(reference, 4))
        source = active_gpu().train_source().read_text()
        actual = deferred_training_batches(source)
        self.assertEqual(len(actual), len(expected))
        for (inputs, targets, epoch), (expected_inputs, expected_targets, expected_epoch) in zip(
                actual, expected, strict=True):
            torch.testing.assert_close(inputs, expected_inputs, rtol=0, atol=0)
            torch.testing.assert_close(targets, expected_targets, rtol=0, atol=0)
            self.assertEqual(epoch, expected_epoch)

    def test_closes_the_original_document_reader_after_prefetch_finishes(self):
        namespace = load_model_namespace()
        closed = []

        def reader(split):
            try:
                yield from documents(split)
            finally:
                closed.append(True)

        namespace['_document_batches'] = reader
        candidate = cpu_allocations('make_training_dataloader',
                                    active_gpu().train_source().read_text(), namespace)
        with closing(candidate(ToyTokenizer(), 2, 17)) as loader:
            next(loader)
        self.assertEqual(closed, [True])

    def test_preserves_original_tokens_bos_and_epochs_across_row_and_document_boundaries(self):
        for batch, length in ((2, 15), (4, 16), (3, 63)):
            with self.subTest(batch=batch, length=length):
                namespace = load_model_namespace()
                namespace['_document_batches'] = documents
                candidate = cpu_allocations('make_training_dataloader',
                                            active_gpu().train_source().read_text(),
                                            namespace)
                with closing(candidate(ToyTokenizer(), batch, length)) as actual, \
                        closing(ordered_batches(batch, length)) as expected:
                    for _ in range(12):
                        x, y, epoch = next(actual)
                        expected_x, expected_y, expected_epoch = next(expected)
                        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
                        torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
                        self.assertEqual(epoch, expected_epoch)
                        torch.testing.assert_close(x[:, 1:], y[:, :-1], rtol=0, atol=0)

    def test_preserves_wide_token_ids_in_the_original_order_across_batches(self):
        wide_ids = (65535, 65536, 2**24 + 1, 2**31, 2**32 - 1)

        def wide_documents(split):
            self.assertEqual(split, 'train')
            for batch in range(100):
                rows = [[wide_ids[(batch + offset + index) % len(wide_ids)] for index in range(size)]
                        for offset, size in enumerate((0, 1, 2, 14, 15, 16, 17, 31, 65, 130))]
                yield rows, batch // 3 + 1

        source = active_gpu().train_source().read_text()
        for batch, length in ((2, 17), (3, 63)):
            with self.subTest(batch=batch, length=length):
                namespace = load_model_namespace()
                namespace['_document_batches'] = wide_documents
                candidate = cpu_allocations('make_training_dataloader', source, namespace)
                seen = set()
                with closing(candidate(ToyTokenizer(), batch, length)) as actual, \
                        closing(ordered_batches(batch, length, wide_documents)) as expected:
                    for _ in range(24):
                        inputs, targets, epoch = next(actual)
                        expected_inputs, expected_targets, expected_epoch = next(expected)
                        torch.testing.assert_close(inputs, expected_inputs, rtol=0, atol=0)
                        torch.testing.assert_close(targets, expected_targets, rtol=0, atol=0)
                        self.assertEqual(epoch, expected_epoch)
                        seen.update(targets.reshape(-1).tolist())
                self.assertEqual(seen, {0, *wide_ids})

    def test_trains_the_model_from_contiguous_batches_and_evaluates_the_tail_average(self):
        namespace = load_model_namespace()
        namespace['_document_batches'] = documents
        candidate = cpu_allocations('make_training_dataloader',
                                    active_gpu().train_source().read_text(),
                                    namespace)
        loader = candidate(ToyTokenizer(), 2, 17)
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
        with closing(loader):
            for step in range(2):
                for _ in range(2):
                    inputs, targets, epoch = next(loader)
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        loss = compiled(inputs, targets) / 2
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                optimizer.step()
                model.zero_grad(set_to_none=True)
                average.update(0.8 + step * 0.1)
            average.apply()
            compiled.eval()
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                logits = compiled(inputs)
                losses = compiled(inputs, targets, reduction='none')
            self.assertEqual(logits.shape, (*inputs.shape, config.vocab_size))
            self.assertEqual(losses.numel(), targets.numel())
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(losses).all())


if __name__ == '__main__':
    unittest.main()
