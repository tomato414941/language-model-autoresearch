"""Prepare and package the GPU runtime and fixed evaluation assets on the processing server."""

import argparse
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


def pack(directory, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    run(["tar", "-C", str(directory), "-I", "gzip -1", "-cf", str(temporary), "."])
    temporary.replace(destination)
    return prepare._file_sha256(destination)


def refresh_source_hashes():
    """Reuse verified runtime archives after changing only training or repository rules."""
    prepare.verify_core_assets()
    artifacts = ROOT / "artifacts"
    path = artifacts / "runtime.json"
    manifest = json.loads(path.read_text())
    hashes = {name: prepare._file_sha256(ROOT / name) for name in SOURCES}
    for name in ("prepare.py", "pyproject.toml", "uv.lock"):
        if hashes[name] != manifest["sources"][name]:
            raise RuntimeError("The runtime or evaluation changed; rebuild its archives first.")
    for name, field in (("gpu-env.tar.gz", "gpu_env_sha256"), ("gpu-assets.tar.gz", "gpu_assets_sha256")):
        if prepare._file_sha256(artifacts / name) != manifest[field]:
            raise RuntimeError("A prepared runtime archive changed; rebuild it first.")
    manifest["sources"] = hashes
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    print("Refreshed source hashes using the unchanged, verified runtime archives.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-source-hashes", action="store_true")
    args = parser.parse_args()
    if args.refresh_source_hashes:
        refresh_source_hashes()
        return
    prepare.verify_core_assets()
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

    for name, source in (("tokenizer", Path(prepare.TOKENIZER_DIR)), ("core", Path(prepare.CORE_DIR))):
        shutil.copytree(source, assets / "autoresearch" / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))

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
        container_image=CUDA_IMAGE, nanochat_revision=prepare.CORE_NANOCHAT_REVISION,
        core_bundle_sha256=prepare.CORE_BUNDLE_SHA256,
        sources={name: prepare._file_sha256(ROOT / name) for name in SOURCES},
    )
    manifest["gpu_env_sha256"] = pack(runtime, artifacts / "gpu-env.tar.gz")
    manifest["gpu_assets_sha256"] = pack(assets, artifacts / "gpu-assets.tar.gz")
    (artifacts / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("GPU runtime and complete evaluation assets packaged.", flush=True)


if __name__ == "__main__":
    main()
