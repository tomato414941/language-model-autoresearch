"""Transfer an experiment's own model definition and parameters to the evaluator."""

import ast
from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

KERNEL_REPOSITORY = "varunneal/flash-attention-3"
KERNEL_REVISION = "de87b9b5af06dd9984df595bef90b2eba44b181a"


def sdpa_attention(q, k, v, causal=True, window_size=(-1, 0)):
    """Use the RTX track's full causal attention through the existing Torch backend."""
    if window_size != (-1, 0):
        raise ValueError("The SDPA model uses full causal attention.")
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, enable_gqa=q.shape[2] != k.shape[2],
    ).transpose(1, 2)


def export_checkpoint(model, destination, source=None):
    model = getattr(model, "_orig_mod", model)
    if source is None:
        source = Path(inspect.getsourcefile(type(model))).read_text()
    payload = {
        "format_version": 1, "class_name": type(model).__name__,
        "source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "config": asdict(model.config),
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
    }
    torch.save(payload, destination)
    return payload["source_sha256"]


def load_definition(source, attention, compile_functions=True):
    tree = ast.parse(source)
    start = next(node.lineno for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GPTConfig")
    end = next(node.end_lineno for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GPT")
    nodes = [node for node in tree.body if start <= node.lineno and node.end_lineno <= end]
    if not compile_functions:
        for node in nodes:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = [item for item in node.decorator_list
                                       if not (isinstance(item, ast.Call) and
                                               isinstance(item.func, ast.Attribute) and item.func.attr == "compile")]
    name = "autoresearch_model_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    module = types.ModuleType(name)
    module.__file__ = "train.py"
    module.__dict__.update(torch=torch, nn=nn, F=F, math=math, dataclass=dataclass,
                           fa3=types.SimpleNamespace(flash_attn_func=attention))
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "train.py", "exec"),
         module.__dict__)
    return module


def load_checkpoint(path, device="cuda", attention=None, compile_functions=True):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["format_version"] != 1 or payload["class_name"] != "GPT":
        raise ValueError("Unsupported experiment checkpoint.")
    if hashlib.sha256(payload["source"].encode()).hexdigest() != payload["source_sha256"]:
        raise ValueError("The checkpoint's model definition is incomplete.")
    if attention is None:
        if payload["config"].get("attention_backend") == "sdpa":
            attention = sdpa_attention
        else:
            from kernels import get_kernel
            attention = get_kernel(KERNEL_REPOSITORY, revision=KERNEL_REVISION).flash_attn_interface.flash_attn_func
    module = load_definition(payload["source"], attention, compile_functions)
    config = module.GPTConfig(**payload["config"])
    with torch.device("meta"):
        model = module.GPT(config)
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    head_dim = config.n_embd // config.n_head
    model.cos, model.sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    model.eval()
    return model, payload["source_sha256"]
