"""Account for CPU autograd tensor shapes without allocating model activations."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import types

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from cpu_smoke import load_model_namespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from prepare import MAX_SEQ_LEN, Tokenizer
from package_runtime import KERNEL_BUILD, KERNEL_REVISION

INTERFACE_SHA256 = "b7b5db2071bdce8f31160bb92d8d1765bf696d0dee3bd580a37efdc847c17ae5"


def attention_forward_shapes(q, k, v, *args, **kwargs):
    out = torch.empty_like(q)
    lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32, device=q.device)
    return out, lse


def attention_backward_shapes(*args, **kwargs):
    raise RuntimeError("Shape accounting does not compute attention values or gradients.")


def load_attention_shapes():
    snapshot = ROOT / "artifacts/gpu-assets/hf-cache/hub/models--varunneal--flash-attention-3/snapshots"
    paths = list((snapshot / KERNEL_REVISION / "build" / KERNEL_BUILD).rglob("flash_attn_interface.py"))
    if len(paths) != 1:
        raise RuntimeError("The prepared, pinned FlashAttention interface is required.")
    source = paths[0].read_bytes()
    if hashlib.sha256(source).hexdigest() != INTERFACE_SHA256:
        raise RuntimeError("The pinned attention interface differs from the inspected tensor layout.")
    definition = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.ClassDef) and node.name == "FlashAttnFunc")
    namespace = {"__name__": "attention_shapes", "torch": torch,
                 "_flash_attn_forward": attention_forward_shapes,
                 "_flash_attn_backward": attention_backward_shapes}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "<pinned_attention>", "exec"), namespace)

    def attention(q, k, v, causal, window_size):
        return namespace["FlashAttnFunc"].apply(q, k, v, None, causal, None, None, None, None, window_size)

    return attention


def constants(source):
    names = {"DEPTH", "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN", "TOTAL_BATCH_SIZE", "DEVICE_BATCH_SIZE"}
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = eval(compile(ast.Expression(node.value), "<constant>", "eval"),
                                             {"__builtins__": {}})
    return result


def profile(path, batch, vocab_size, attention):
    source = path.read_text()
    settings = constants(source)
    namespace = load_model_namespace(path)
    namespace["fa3"] = types.SimpleNamespace(flash_attn_func=attention)
    width = ((settings["DEPTH"] * settings["ASPECT_RATIO"] + settings["HEAD_DIM"] - 1)
             // settings["HEAD_DIM"]) * settings["HEAD_DIM"]
    config = namespace["GPTConfig"](
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size, n_layer=settings["DEPTH"],
        n_head=width // settings["HEAD_DIM"], n_kv_head=width // settings["HEAD_DIM"],
        n_embd=width, window_pattern=settings["WINDOW_PATTERN"],
    )
    with FakeTensorMode():
        model = namespace["GPT"](config)
        model.init_weights()
        persistent = {t.untyped_storage()._cdata: t.untyped_storage()
                      for t in (*model.parameters(), *model.buffers())}
        saved = {}
        section = ["embedding_and_residual"]
        hooks = []
        for block in model.transformer.h:
            hooks.append(block.register_forward_pre_hook(
                lambda *args: section.__setitem__(0, "transformer_blocks")))
            hooks.append(block.register_forward_hook(
                lambda *args: section.__setitem__(0, "embedding_and_residual")))
        hooks.append(model.lm_head.register_forward_pre_hook(
            lambda *args: section.__setitem__(0, "head_and_loss")))

        def pack(tensor):
            storage = tensor.untyped_storage()
            saved.setdefault(storage._cdata, {
                "storage": storage, "bytes": storage.nbytes(), "section": section[0],
                "dtype": str(tensor.dtype), "shape": list(tensor.shape),
            })
            return tensor.detach()

        inputs = torch.zeros((batch, MAX_SEQ_LEN), dtype=torch.long)
        targets = torch.zeros_like(inputs)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(inputs, targets)
        assert loss.ndim == 0 and saved
        activation = [row for key, row in saved.items() if key not in persistent]
        sections = Counter()
        dtypes = Counter()
        for row in activation:
            sections[row["section"]] += row["bytes"]
            dtypes[row["dtype"]] += row["bytes"]
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        state_bytes = 0
        for name, parameter in model.named_parameters():
            if name.startswith("transformer.h."):
                state_bytes += parameter.numel() * parameter.element_size()
                state_bytes += max(parameter.shape) * parameter.element_size()
            else:
                state_bytes += 2 * parameter.numel() * parameter.element_size()
        for hook in hooks:
            hook.remove()
        return {
            "batch": batch, "sequence_length": MAX_SEQ_LEN, "width": width,
            "depth": config.n_layer, "vocabulary_size": vocab_size,
            "token_embedding_width": model.transformer.wte.embedding_dim,
            "layer_head_dimensions": [block.attn.head_dim for block in model.transformer.h],
            "configured_device_batch": settings["DEVICE_BATCH_SIZE"],
            "total_batch_tokens": settings["TOTAL_BATCH_SIZE"],
            "saved_unique_nonpersistent_bytes": sum(row["bytes"] for row in activation),
            "saved_nonpersistent_by_section": dict(sections),
            "saved_nonpersistent_by_dtype": dict(dtypes),
            "persistent_model_bytes": sum(storage.nbytes() for storage in persistent.values()),
            "all_parameter_gradient_bytes": parameter_bytes,
            "steady_optimizer_state_bytes": state_bytes,
            "largest_saved_nonpersistent_storages": [
                {key: value for key, value in row.items() if key != "storage"}
                for row in sorted(activation, key=lambda row: row["bytes"], reverse=True)[:8]
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(batch <= 0 for batch in args.batches):
        parser.error("Batch sizes must be positive.")
    attention = load_attention_shapes()
    vocab_size = Tokenizer.from_directory().get_vocab_size()
    rows = [profile(args.source, batch, vocab_size, attention) for batch in args.batches]
    report = {
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "attention_interface_sha256": INTERFACE_SHA256,
        "torch_version": torch.__version__, "profiles": rows,
        "sources": [
            "https://docs.pytorch.org/docs/2.9/autograd.html#torch.autograd.graph.saved_tensors_hooks",
            f"https://huggingface.co/varunneal/flash-attention-3/tree/{KERNEL_REVISION}/build/{KERNEL_BUILD}",
        ],
        "limitations": [
            "CPU autocast and eager autograd shape accounting; no predictions or scores are computed.",
            "The unchanged pinned FA3 Python class uses a shape-only forward kernel; no CUDA kernel or workspace.",
            "Saved aliases share one storage count; persistent model storage is counted separately.",
            "Compiler fusion, recomputation, CUDA autocast, allocator reuse, and backward temporaries can differ.",
            "The reported components are not a measured or guaranteed GPU memory peak or a throughput estimate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(f"depth={row['depth']} width={row['width']} batch={row['batch']} "
              f"saved_nonpersistent_GiB={row['saved_unique_nonpersistent_bytes'] / 2**30:.3f} "
              f"persistent_GiB={row['persistent_model_bytes'] / 2**30:.3f} "
              f"gradient_GiB={row['all_parameter_gradient_bytes'] / 2**30:.3f} "
              f"optimizer_state_GiB={row['steady_optimizer_state_bytes'] / 2**30:.3f}")


if __name__ == "__main__":
    main()
