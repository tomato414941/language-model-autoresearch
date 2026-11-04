"""Measure projection rounding and its optimizer amplification on synthetic CPU data."""

import ast
import contextlib
import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.cpu_smoke import load_model_namespace


def separate_projection(self, x):
    return self.c_q(x), self.c_k(x), self.c_v(x)


def relative_difference(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    difference = (actual - expected).norm().item()
    norm = expected.norm().item()
    return dict(difference_l2=difference, reference_l2=norm,
                relative_l2=difference / max(norm, 1e-30))


def float32_muon_namespace():
    namespace = load_model_namespace()
    tree = ast.parse((ROOT / 'train.py').read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'muon_step_fused')
    node.decorator_list = []

    class FullPrecisionMuon(ast.NodeTransformer):
        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if node.attr == 'bfloat16':
                node.attr = 'float'
            return node

    node = FullPrecisionMuon().visit(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 'float32_muon_diagnostic', 'exec'), namespace)
    return namespace


def inspect_updates(namespace, precision, muon_precision='bfloat16'):
    torch.manual_seed(42)
    config = namespace['GPTConfig'](sequence_len=32, vocab_size=256, n_layer=4,
                                   n_head=2, n_kv_head=2, n_embd=64)
    packed = namespace['GPT'](config)
    packed.init_weights()
    if precision == 'float32':
        packed.float()
    # Exercise Q/K/V gradients immediately instead of the zero-output initialization.
    with torch.no_grad():
        for block in packed.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.02)
    reference = copy.deepcopy(packed)
    for block in reference.transformer.h:
        block.attn.project_qkv = types.MethodType(separate_projection, block.attn)
    with contextlib.redirect_stdout(io.StringIO()):
        packed_optimizer = packed.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02, weight_decay=0.2)
        reference_optimizer = reference.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02, weight_decay=0.2)
    before = {name: value.detach().clone() for name, value in packed.named_parameters()}
    losses = []
    for _ in range(2):
        inputs = torch.randint(0, config.vocab_size, (2, 32))
        targets = torch.randint(0, config.vocab_size, (2, 32))
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=precision == 'bfloat16'):
            loss = packed(inputs, targets) / 2
            reference_loss = reference(inputs, targets) / 2
        if not torch.isfinite(loss) or not torch.isfinite(reference_loss):
            raise RuntimeError('The synthetic CPU loss is non-finite.')
        losses.append(dict(packed=float(loss.detach()), separate=float(reference_loss.detach())))
        loss.backward()
        reference_loss.backward()
    gradients = []
    for (name, parameter), (reference_name, reference_parameter) in zip(
            packed.named_parameters(), reference.named_parameters(), strict=True):
        if name != reference_name or parameter.grad is None or reference_parameter.grad is None:
            raise RuntimeError('The projection changed parameter coverage.')
        if not torch.isfinite(parameter.grad).all() or not torch.isfinite(reference_parameter.grad).all():
            raise RuntimeError('The accumulated gradients are non-finite.')
        gradients.append(dict(parameter=name, **relative_difference(parameter.grad, reference_parameter.grad)))
    packed_optimizer.step()
    reference_optimizer.step()
    updates = []
    for (name, parameter), (_, reference_parameter) in zip(
            packed.named_parameters(), reference.named_parameters(), strict=True):
        if not torch.isfinite(parameter).all() or not torch.isfinite(reference_parameter).all():
            raise RuntimeError('The optimizer produced non-finite weights.')
        update = parameter.detach() - before[name]
        reference_update = reference_parameter.detach() - before[name]
        values = relative_difference(update, reference_update)
        # Report the numerical-preservation hypothesis without treating it as a quality gate.
        within_epsilon = values['difference_l2'] <= torch.finfo(torch.bfloat16).eps * values['reference_l2'] + 1e-7
        updates.append(dict(parameter=name, **values, within_one_bf16_epsilon=within_epsilon))
    return dict(precision=precision, muon_precision=muon_precision, config=asdict(config), accumulation_steps=2,
                losses=losses, gradients=gradients, updates=updates,
                worst_gradient=max(gradients, key=lambda row: row['relative_l2']),
                worst_update=max(updates, key=lambda row: row['relative_l2']),
                updates_outside_one_bf16_epsilon=sum(not row['within_one_bf16_epsilon'] for row in updates))


def main():
    namespace = load_model_namespace()
    reports = [inspect_updates(namespace, precision) for precision in ('float32', 'bfloat16')]
    reports.append(inspect_updates(float32_muon_namespace(), 'float32', muon_precision='float32'))
    report = dict(
        diagnostic_only=True, training_source_sha256=hashlib.sha256((ROOT / 'train.py').read_bytes()).hexdigest(),
        seed=42, torch_version=torch.__version__, cases=reports,
        limitations=[
            'Synthetic CPU inputs and small random model states do not measure CORE or GPU throughput.',
            'Default Muon uses BF16 orthogonalization even with float32 forward/backward.',
            'The explicit FP32-Muon ablation changes optimizer arithmetic only in this diagnostic process.',
            'Similar forward values and gradients do not guarantee matching low-precision optimizer updates.',
            'GPU kernels and torch.compile may produce different numerical behavior from this CPU diagnostic.',
        ],
    )
    destination = ROOT / 'results/projection-training-diagnostics.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}, indent=2))
    for case in reports:
        print(json.dumps({key: case[key] for key in (
            'precision', 'muon_precision', 'worst_gradient', 'worst_update',
            'updates_outside_one_bf16_epsilon')}, indent=2))


if __name__ == '__main__':
    main()
