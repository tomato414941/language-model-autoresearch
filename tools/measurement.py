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
# The bindings of prepare.py that decide what a run may train on and for how long. The score itself is the OLMES
# evaluation's, which the definition names by its revision, suites and datasets.
MEASUREMENT_ROOTS = ("TIME_BUDGET", "download_data", "BASE_URL", "MAX_SHARD", "VAL_SHARD", "VAL_FILENAME", "DATA_DIR")
OLMES_METRIC = "olmes_bpb"
# How prepare.py reads the training budget at run time; the job's frozen budget takes its place in the hash.
BUDGET = ('active_gpu().settings()["training_seconds"]', "TIME_BUDGET")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def native_datasets_identity(tasks):
    """Identify the complete native task data independently of execution order and tokenizer."""
    datasets = {entry["alias"]: {name: entry[name] for name in
                               ("task_hash", "dataset_fingerprints", "examples")}
                for entry in tasks}
    if len(datasets) != len(tasks):
        raise ValueError("Every native task alias must occur exactly once.")
    return digest(json.dumps(datasets, sort_keys=True, separators=(",", ":")).encode())


def prepare_semantics(source, settings):
    """Hash the data a run may use and its frozen training budget, with their transitive module dependencies."""
    tree = ast.parse(source)
    expression = ast.dump(ast.parse(BUDGET[0], mode="eval").body)
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == BUDGET[1] and ast.dump(node.value) == expression):
            node.value = ast.parse(repr(settings["training_seconds"]), mode="eval").body
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
    pending = list(MEASUREMENT_ROOTS)
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
    """The measurement a job was prepared under, from its own snapshot: the data it may train on and its frozen budget,
    its dependencies, runtime and GPU, and the OLMES evaluation."""
    if job["metric"] != OLMES_METRIC:
        raise ValueError("A measurement is an OLMES evaluation.")
    directory = Path(directory)
    archive_file = directory / "source.tar.gz"
    if digest(archive_file.read_bytes()) != job["source_archive_sha256"]:
        raise ValueError("The archived source differs from the prepared experiment.")
    with tarfile.open(archive_file, "r:gz") as archive:
        member = archive.getmember("prepare.py")
        if not member.isfile() or member.size > 2_000_000:
            raise ValueError("The archived measurement source is invalid.")
        source = archive.extractfile(member).read()
        config = archive.extractfile("gpus.json").read()
    if digest(config) != job["source_sha256"]["gpus.json"]:
        raise ValueError("The archived GPU settings differ from their recorded hash.")
    profile = json.loads(config)["gpus"][job["gpu_id"]]
    settings = {name: profile[name] for name in ("training_seconds", "rental_seconds", "max_hourly_cost")}
    if settings != job["gpu_settings"] or profile["gpu_name"] != job["gpu_name"]:
        raise ValueError("The frozen GPU settings differ from the prepared job.")
    if digest(source) != job["source_sha256"]["prepare.py"]:
        raise ValueError("The measurement source differs from its recorded hash.")
    runtime_bytes = (directory / "runtime.json").read_bytes()
    if digest(runtime_bytes) != job["runtime_manifest_sha256"]:
        raise ValueError("The archived runtime differs from its recorded hash.")
    runtime = json.loads(runtime_bytes)
    definition = dict(
        version=3, metric=OLMES_METRIC, prepare_semantics_sha256=prepare_semantics(source, settings),
        dependencies={name: job["source_sha256"][name] for name in
                      ("pyproject.toml", "uv.lock", "evaluation/pyproject.toml", "evaluation/uv.lock")},
        gpu_name=job["gpu_name"], num_gpus=1,
        runtime={name: runtime[name] for name in RUNTIME_FIELDS},
        olmes={name: runtime["olmes"][name] for name in
               ("revision", "suites", "approved_config_fix", "native_datasets_sha256")},
    )
    # The source archive retains the rules and model; tokenizer assets are a model choice.
    encoded = json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    return dict(id=digest(encoded), definition=definition)


def decision_helper(job, directory, name):
    """Use the user's decision function saved before this experiment was measured."""
    with tarfile.open(Path(directory) / "source.tar.gz", "r:gz") as archive:
        source = archive.extractfile("prepare.py").read()
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {"math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "prepare.py", "exec"), namespace)
    return namespace[name]
