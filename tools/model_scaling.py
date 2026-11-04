"""Inspect model sizes and training shapes on CPU without training or scoring a model."""

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prepare import MAX_SEQ_LEN, Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'train.py')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/model-scaling.json')
    args = parser.parse_args()
    source = args.source.read_text()
    tree = ast.parse(source)
    names = {'GPTConfig', 'CausalSelfAttention', 'MLP', 'Block', 'GPT',
             'norm', 'has_ve', 'apply_rotary_emb', 'build_model_config'}
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = dict(__name__='__main__', torch=torch, nn=nn, F=F, math=math, dataclass=dataclass,
                     MAX_SEQ_LEN=MAX_SEQ_LEN, vocab_size=Tokenizer.from_directory().get_vocab_size())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'ASPECT_RATIO', 'HEAD_DIM', 'WINDOW_PATTERN'}:
                    namespace[target.id] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'train.py', 'exec'), namespace)
    rows = []
    for depth in (8, 9, 10, 12, 14):
        for divisor in (2, 4):
            config = namespace['build_model_config'](depth)
            with torch.device('meta'):
                model = namespace['GPT'](config)
            model.window_sizes = [(MAX_SEQ_LEN // divisor, right) if 0 <= left < MAX_SEQ_LEN else (left, right)
                                  for left, right in model.window_sizes]
            counts = model.num_scaling_params()
            for device_batch, total_batch in ((128, 524288), (64, 262144)):
                rows.append(dict(
                    depth=depth, width=config.n_embd, heads=config.n_head, short_window=MAX_SEQ_LEN // divisor,
                    device_batch=device_batch, total_batch_tokens=total_batch,
                    accumulation_steps=total_batch // (device_batch * MAX_SEQ_LEN),
                    parameters=counts, scaling_parameters=counts['transformer_matrices'] + counts['lm_head'],
                    token_embedding_width=model.transformer.wte.embedding_dim,
                    layer_head_dimensions=[block.attn.head_dim for block in model.transformer.h],
                    flops_per_token=model.estimate_flops(),
                    window_sizes=model.window_sizes,
                ))
    report = dict(training_source_sha256=hashlib.sha256(source.encode()).hexdigest(), models=rows)
    destination = args.output
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        if row['device_batch'] == 64:
            print(f"depth={row['depth']:2d} width={row['width']:3d} short={row['short_window']:4d} "
                  f"params={row['parameters']['total'] / 1e6:6.1f}M "
                  f"scaling_params={row['scaling_parameters'] / 1e6:6.1f}M "
                  f"FLOPs/token={row['flops_per_token'] / 1e6:6.1f}M")


if __name__ == '__main__':
    main()
