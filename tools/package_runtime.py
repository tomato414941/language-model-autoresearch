"""Prepare and package the GPU runtime and fixed evaluation assets on the processing server."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import prepare

PYTHON_VERSION = "3.12.16"
KERNEL_REPO = "varunneal/flash-attention-3"
KERNEL_REVISION = "de87b9b5af06dd9984df595bef90b2eba44b181a"
KERNEL_BUILD = "torch29-cxx11-cu128-x86_64-linux"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66"
SOURCES = ("train.py", "prepare.py", "program.md", "pyproject.toml", "uv.lock")


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_tokenizer():
    tokenizer = prepare.Tokenizer.from_directory()
    token_bytes = prepare.get_token_bytes("cpu")
    if token_bytes.ndim != 1 or token_bytes.numel() != tokenizer.get_vocab_size():
        raise RuntimeError("The tokenizer and BPB byte table disagree.")
    if token_bytes.is_floating_point() or bool((token_bytes < 0).any()):
        raise RuntimeError("The tokenizer byte table must contain nonnegative integers.")
    return {name: digest(Path(prepare.TOKENIZER_DIR) / name)
            for name in ("tokenizer.pkl", "token_bytes.pt")}


def pack(directory, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    run(["tar", "-C", str(directory), "-I", "gzip -1", "-cf", str(temporary), "."])
    temporary.replace(destination)
    return digest(destination)


def pack_runtime_assets(artifacts):
    previous = artifacts / "gpu-assets.tar.gz"
    if previous.exists():
        history = artifacts / "history"
        history.mkdir(exist_ok=True)
        preserved = history / ("gpu-assets-" + digest(previous) + ".tar.gz")
        if not preserved.exists():
            os.link(previous, preserved)
    with tempfile.TemporaryDirectory(dir=artifacts, prefix=".bpb-assets-") as temporary:
        assets = Path(temporary)
        shutil.copytree(Path(prepare.TOKENIZER_DIR), assets / "autoresearch" / "tokenizer",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(artifacts / "gpu-assets" / "hf-cache", assets / "hf-cache",
                        symlinks=True, copy_function=os.link)
        return pack(assets, artifacts / "gpu-assets.tar.gz")


def refresh_assets():
    """Repackage current tokenizer assets using the unchanged locked runtime and kernel cache."""
    artifacts = ROOT / "artifacts"
    path = artifacts / "runtime.json"
    original = path.read_bytes()
    manifest = json.loads(original)
    for name in ("pyproject.toml", "uv.lock"):
        if digest(ROOT / name) != manifest["sources"][name]:
            raise RuntimeError("The dependencies changed; rebuild the locked runtime first.")
    for name, field in (("gpu-env.tar.gz", "gpu_env_sha256"), ("gpu-assets.tar.gz", "gpu_assets_sha256")):
        if digest(artifacts / name) != manifest[field]:
            raise RuntimeError("A prepared runtime archive changed; retain its original provenance.")
    tokenizer = verify_tokenizer()
    history = artifacts / "history"
    history.mkdir(exist_ok=True)
    (history / ("runtime-" + hashlib.sha256(original).hexdigest() + ".json")).write_bytes(original)
    manifest["gpu_assets_sha256"] = pack_runtime_assets(artifacts)
    manifest.pop("nanochat_revision", None)
    manifest.pop("core_bundle_sha256", None)
    manifest.update(evaluation_metric="val_bpb", tokenizer=tokenizer,
                    sources={name: digest(ROOT / name) for name in SOURCES})
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    print("Repackaged tokenizer and offline kernel assets; the locked runtime is unchanged.", flush=True)


def refresh_source_hashes():
    """Reuse verified runtime archives after changing only training or repository rules."""
    artifacts = ROOT / "artifacts"
    path = artifacts / "runtime.json"
    manifest = json.loads(path.read_text())
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    for name in ("prepare.py", "pyproject.toml", "uv.lock"):
        if hashes[name] != manifest["sources"][name]:
            raise RuntimeError("The runtime or evaluation changed; rebuild its archives first.")
    for name, field in (("gpu-env.tar.gz", "gpu_env_sha256"), ("gpu-assets.tar.gz", "gpu_assets_sha256")):
        if digest(artifacts / name) != manifest[field]:
            raise RuntimeError("A prepared runtime archive changed; rebuild it first.")
    manifest["sources"] = hashes
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    print("Refreshed source hashes using the unchanged, verified runtime archives.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refresh-source-hashes", action="store_true")
    mode.add_argument("--refresh-assets", action="store_true")
    args = parser.parse_args()
    if args.refresh_source_hashes:
        refresh_source_hashes()
        return
    if args.refresh_assets:
        refresh_assets()
        return
    tokenizer = verify_tokenizer()
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv must be available on the processing server.")
    artifacts = ROOT / "artifacts"
    runtime = artifacts / "gpu-env"
    assets = artifacts / "gpu-assets"
    runtime.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    run([uv, "python", "install", PYTHON_VERSION, "--install-dir", str(runtime / ".python"),
         "--no-bin", "--no-registry"])
    interpreter = runtime / ".python" / f"cpython-{PYTHON_VERSION}-linux-x86_64-gnu" / "bin" / "python3.12"
    environment = runtime / ".venv"
    if not environment.exists():
        run([uv, "venv", str(environment), "--python", str(interpreter), "--relocatable"])
    run([uv, "sync", "--frozen", "--no-dev", "--project", str(ROOT)],
        env=dict(os.environ, UV_PROJECT_ENVIRONMENT=str(environment)))
    executable = environment / "bin" / "python"
    executable.unlink()
    executable.symlink_to(os.path.relpath(interpreter, executable.parent))
    configuration = environment / "pyvenv.cfg"
    configuration.write_text("\n".join(
        "home = " + os.path.relpath(interpreter.parent, environment) if line.startswith("home =") else line
        for line in configuration.read_text().splitlines()
    ) + "\n")
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, runtime / name)
    shutil.copy2(uv, runtime / "uv")

    # Exercise a different path, using only the installed interpreter and locked packages.
    with tempfile.TemporaryDirectory(dir=artifacts, prefix=".runtime-check-") as temporary:
        relocated = Path(temporary) / "runtime"
        shutil.copytree(runtime, relocated, symlinks=True, copy_function=os.link)
        run([str(relocated / "uv"), "run", "--offline", "--frozen", "python", "-c",
             "import torch, kernels; print('Relocated runtime:', torch.__version__)"], cwd=relocated)

    os.environ["HF_HOME"] = str(assets / "hf-cache")
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(KERNEL_REPO, revision="main", cache_dir=assets / "hf-cache" / "hub",
                                 allow_patterns=[f"build/{KERNEL_BUILD}/*"])
    if Path(snapshot).name != KERNEL_REVISION:
        raise RuntimeError("The upstream kernel changed; retain the prepared frame instead of silently updating it.")
    run([str(environment / "bin" / "python"), "-c",
         "import kernels; install=kernels.get_kernel.__globals__['install_kernel']; "
         "print('Offline kernel assets:', install('varunneal/flash-attention-3', revision='main', local_files_only=True)[0])"],
        env=dict(os.environ, HF_HUB_OFFLINE="1"))

    manifest = dict(
        python_version=PYTHON_VERSION, kernel_repo=KERNEL_REPO, kernel_revision=KERNEL_REVISION,
        container_image=CUDA_IMAGE, evaluation_metric="val_bpb", tokenizer=tokenizer,
        sources={name: digest(ROOT / name) for name in SOURCES},
    )
    manifest["gpu_env_sha256"] = pack(runtime, artifacts / "gpu-env.tar.gz")
    manifest["gpu_assets_sha256"] = pack_runtime_assets(artifacts)
    (artifacts / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("GPU runtime, tokenizer and offline kernel assets packaged.", flush=True)


if __name__ == "__main__":
    main()
