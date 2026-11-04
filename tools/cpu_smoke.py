import ast
import dataclasses
import math
import pathlib
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)


def load_model_namespace(path=None):
    if path is None:
        path = pathlib.Path(__file__).resolve().parent.parent / 'train.py'
    source = pathlib.Path(path).read_text()
    fragment = source[source.index('@dataclass'):source.index('# Hyperparameters')]
    tree = ast.parse(fragment)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = [d for d in node.decorator_list if not (
                isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == 'compile')]
    namespace = {'__name__': '__main__', 'torch': torch, 'nn': nn, 'F': F, 'math': math,
                 'dataclass': dataclasses.dataclass, 'fa3': types.SimpleNamespace(flash_attn_func=attention)}
    exec(compile(ast.fix_missing_locations(tree), 'train.py', 'exec'), namespace)
    return namespace

def attention(q, k, v, causal, window_size):
    length = q.shape[1]
    positions = torch.arange(length)
    distance = positions[:, None] - positions[None, :]
    allowed = distance >= 0
    if window_size[0] >= 0:
        allowed &= distance <= window_size[0]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed,
        enable_gqa=q.shape[2] != k.shape[2],
    ).transpose(1, 2)


def main():
    namespace = load_model_namespace()
    torch.manual_seed(42)
    config = namespace['GPTConfig'](sequence_len=32, vocab_size=256, n_layer=4, n_head=2, n_kv_head=2, n_embd=64, window_pattern='SSSL')
    model = namespace['GPT'](config)
    model.init_weights()
    optimizer = model.setup_optimizer(embedding_lr=0.03, matrix_lr=0.02)
    torch.manual_seed(42)
    x = torch.randint(0, config.vocab_size, (2, 32))
    y = torch.randint(0, config.vocab_size, (2, 32))
    before = model.lm_head.weight.detach().clone()
    for _ in range(3):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = model(x, y)
        assert torch.isfinite(loss), loss
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        model.zero_grad(set_to_none=True)
    assert not torch.equal(before, model.lm_head.weight)
    model.eval()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        losses = model(x, y, reduction='none')
        assert losses.numel() == y.numel() and torch.isfinite(losses).all()
        logits = model(x)
        changed = x.clone()
        changed[:, 16:] = (changed[:, 16:] + 1) % config.vocab_size
        changed_logits = model(changed)
        torch.testing.assert_close(logits[:, :16], changed_logits[:, :16], rtol=0, atol=0)
    print(f'PASS: small-model forward/backward, all optimizer groups, per-token loss, and causal predictions; loss={loss.item():.6f}')


if __name__ == '__main__':
    main()
