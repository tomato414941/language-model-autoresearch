"""Identify measurement conditions from an experiment's immutable source snapshot."""

import ast
import hashlib
import json
from pathlib import Path
import tarfile


RUNTIME_FIELDS = (
    "python_version", "kernel_repo", "kernel_revision", "container_image",
    "nanochat_revision", "core_bundle_sha256", "gpu_env_sha256",
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def prepare_semantics(source):
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "core_improves":
            continue
        targets = (node.targets if isinstance(node, ast.Assign) else
                   [node.target] if isinstance(node, ast.AnnAssign) else [])
        if targets and all(isinstance(target, ast.Name) and target.id == "CORE_MIN_DELTA"
                           for target in targets):
            continue
        body.append(node)
    tree.body = body
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
        version=1, prepare_semantics_sha256=prepare_semantics(source),
        dependencies={name: job["source_sha256"][name]
                      for name in ("pyproject.toml", "uv.lock")},
        gpu_name=job["gpu_name"], num_gpus=1,
        runtime={name: runtime[name] for name in RUNTIME_FIELDS},
    )
    # The source archive retains the rules and model; tokenizer assets are a model choice.
    encoded = json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    return dict(id=digest(encoded), definition=definition)
