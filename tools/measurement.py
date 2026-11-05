"""Identify measurement conditions from an experiment's immutable source snapshot."""

import ast
import hashlib
import json
import math
from pathlib import Path
import tarfile


RUNTIME_FIELDS = (
    "python_version", "kernel_repo", "kernel_revision", "container_image",
    "gpu_env_sha256",
)

MEASUREMENT_ROOTS = (
    "evaluate_bpb", "download_data", "train_tokenizer", "MAX_SEQ_LEN", "TIME_BUDGET",
    "BASE_URL", "MAX_SHARD", "VAL_SHARD", "VAL_FILENAME", "DATA_DIR",
)
OLMES_METRIC = "olmes_bpb"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def prepare_semantics(source, roots=MEASUREMENT_ROOTS):
    """Hash BPB, allowed data and budgets with their transitive module dependencies."""
    tree = ast.parse(source)
    bindings = {}
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.asname or alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [value.id for target in targets for value in ast.walk(target)
                     if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Store)]
        for name in names:
            bindings.setdefault(name, []).append(node)
    if not {"evaluate_bpb", "TIME_BUDGET"} <= bindings.keys():
        raise ValueError("The archived source must define BPB evaluation and its training budget.")
    pending = list(roots)
    selected = set()
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in bindings.get(name, ()):
            selected.add(node)
            pending.extend(value.id for value in ast.walk(node)
                           if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load))
    tree.body = [node for node in tree.body if node in selected]
    return digest(ast.dump(tree, include_attributes=False).encode())


def from_snapshot(job, directory):
    directory = Path(directory)
    archive_file = directory / "source.tar.gz"
    if digest(archive_file.read_bytes()) != job["source_archive_sha256"]:
        raise ValueError("The archived source differs from the prepared experiment.")
    with tarfile.open(archive_file, "r:gz") as archive:
        member = archive.getmember("prepare.py")
        if not member.isfile() or member.size > 2_000_000:
            raise ValueError("The archived measurement source is invalid.")
        source = archive.extractfile(member).read()
    if digest(source) != job["source_sha256"]["prepare.py"]:
        raise ValueError("The measurement source differs from its recorded hash.")
    runtime_bytes = (directory / "runtime.json").read_bytes()
    if digest(runtime_bytes) != job["runtime_manifest_sha256"]:
        raise ValueError("The archived runtime differs from its recorded hash.")
    runtime = json.loads(runtime_bytes)
    definition = dict(
        version=2, metric="val_bpb", prepare_semantics_sha256=prepare_semantics(source),
        dependencies={name: job["source_sha256"][name]
                      for name in ("pyproject.toml", "uv.lock")},
        gpu_name=job["gpu_name"], num_gpus=1,
        runtime={name: runtime[name] for name in RUNTIME_FIELDS},
    )
    if job.get("metric") != OLMES_METRIC:
        definition["runtime"]["gpu_env_sha256"] = runtime.get(
            "training_gpu_env_sha256", runtime["gpu_env_sha256"])
    if job.get("metric") == OLMES_METRIC:
        evaluation = runtime["olmes"]
        definition.update(version=3, metric=OLMES_METRIC,
                          prepare_semantics_sha256=prepare_semantics(source, (
                              "TIME_BUDGET", "download_data", "BASE_URL", "MAX_SHARD",
                              "VAL_SHARD", "VAL_FILENAME", "DATA_DIR")),
                          olmes={name: evaluation[name] for name in
                                 ("revision", "suites", "approved_config_fix", "native_datasets_sha256")})
        definition["dependencies"].update({name: job["source_sha256"][name] for name in
                                           ("evaluation/pyproject.toml", "evaluation/uv.lock")})
    # The source archive retains the rules and model; tokenizer assets are a model choice.
    encoded = json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    return dict(id=digest(encoded), definition=definition)


def for_metric(job, directory, metric):
    """Identify a shared evaluation, including the unchanged reference BPB runtime."""
    return from_snapshot(dict(job, metric=metric), directory)


def decision_helper(job, directory, name):
    """Use the user's decision function saved before this experiment was measured."""
    with tarfile.open(Path(directory) / "source.tar.gz", "r:gz") as archive:
        source = archive.extractfile("prepare.py").read()
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {"math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "prepare.py", "exec"), namespace)
    return namespace[name]
