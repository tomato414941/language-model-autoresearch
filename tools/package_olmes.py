"""Build a separate relocatable OLMES runtime without changing a prepared rental."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare
from tools.package_runtime import PYTHON_VERSION, digest, pack, verify_tokenizer
from tools.measurement import native_datasets_identity
from gpu_config import active_gpu, source_files


def run(arguments, **kwargs):
    subprocess.run(arguments, check=True, **kwargs)


def package_assets():
    from huggingface_hub import snapshot_download
    from evaluation.arc_macro import APPROVED_CONFIG_FIX

    profile = active_gpu()
    artifacts = profile.artifacts_directory(ROOT)
    requests = Path(prepare.OLMES_REQUEST_DIR)
    manifest = json.loads((requests / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Prepare every native request before packaging the evaluation.")
    for entry in manifest["tasks"]:
        if digest(requests / entry["requests_file"]) != entry["requests_sha256"]:
            raise ValueError("A prepared native request file is incomplete.")
    runtime = json.loads((artifacts / "runtime.json").read_text())
    staged = json.loads((profile.results_directory(ROOT) / "olmes-relocatable-runtime.json").read_text())
    if digest(artifacts / "gpu-env-olmes.tar.gz") != staged["gpu_env_sha256"]:
        raise ValueError("The staged OLMES runtime is incomplete.")
    with tempfile.TemporaryDirectory(dir=artifacts, prefix=".olmes-assets-") as temporary:
        assets = Path(temporary)
        cache = artifacts / "gpu-assets/hf-cache"
        if cache.exists():
            shutil.copytree(cache, assets / "hf-cache", symlinks=True, copy_function=os.link)
        cache_root = assets / "autoresearch" / "gpus" / profile.identifier
        shutil.copytree(requests, cache_root / "olmes", copy_function=os.link)
        shutil.copytree(prepare.TOKENIZER_DIR, cache_root / "tokenizer",
                        ignore=shutil.ignore_patterns("__pycache__"))
        if profile.values["attention_backend"] == "flash_attention_3":
            for build in ("torch28-cxx11-cu128-x86_64-linux", "torch29-cxx11-cu128-x86_64-linux"):
                snapshot_download(profile.values["kernel_repo"], revision=profile.values["kernel_revision"],
                                  cache_dir=assets / "hf-cache/hub", allow_patterns=[f"build/{build}/*"])
        assets_sha256 = pack(assets, artifacts / "gpu-assets-olmes.tar.gz")
    datasets_sha256 = native_datasets_identity(manifest["tasks"])
    runtime.pop("transport", None)
    runtime.update(
        gpu_id=profile.identifier, sources={name: digest(ROOT / name) for name in source_files()},
        evaluation_metric="olmes_bpb", gpu_env_sha256=staged["gpu_env_sha256"],
        gpu_assets_sha256=assets_sha256, tokenizer=verify_tokenizer(),
        olmes=dict(revision=manifest["olmes_revision"], suites=manifest["suites"],
                   approved_config_fix=APPROVED_CONFIG_FIX, native_datasets_sha256=datasets_sha256,
                   request_manifest_sha256=digest(requests / "manifest.json"),
                   leaf_tasks=len(manifest["tasks"]), examples=manifest["total_examples"],
                   requests=manifest["total_requests"]),
    )
    (artifacts / "runtime-olmes.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(dict(gpu_assets_sha256=assets_sha256,
                          bytes=(artifacts / "gpu-assets-olmes.tar.gz").stat().st_size,
                          olmes=runtime["olmes"], prepared_rental_archives_preserved=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    if args.assets_only:
        package_assets()
        return
    profile = active_gpu()
    artifacts = profile.artifacts_directory(ROOT)
    profile.results_directory(ROOT).mkdir(parents=True, exist_ok=True)
    runtime = artifacts / "gpu-env-olmes"
    base = artifacts / "gpu-env"
    if not runtime.exists():
        shutil.copytree(base, runtime, symlinks=True, copy_function=os.link)
    interpreter = runtime / ".python" / f"cpython-{PYTHON_VERSION}-linux-x86_64-gnu/bin/python3.12"
    environment_dir = runtime / "evaluation/.venv"
    if not environment_dir.exists():
        environment_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["uv", "venv", str(environment_dir), "--python", str(interpreter), "--relocatable"])
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("VAST_", "PROCESSING_"))}
    environment["UV_PROJECT_ENVIRONMENT"] = str(environment_dir)
    environment["UV_LINK_MODE"] = "hardlink"
    run(["uv", "sync", "--project", str(ROOT / "evaluation"), "--frozen", "--no-dev", "--no-editable"],
        env=environment)
    executable = environment_dir / "bin/python"
    executable.unlink()
    executable.symlink_to(os.path.relpath(interpreter, executable.parent))
    configuration = environment_dir / "pyvenv.cfg"
    lines = configuration.read_text().splitlines()
    text = "\n".join("home = " + os.path.relpath(interpreter.parent, environment_dir)
                     if line.startswith("home =") else line for line in lines) + "\n"
    temporary = configuration.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(configuration)
    for name in ("pyproject.toml", "uv.lock"):
        target = runtime / "evaluation" / name
        target.unlink(missing_ok=True)
        shutil.copy2(ROOT / "evaluation" / name, target)
    log_path = profile.results_directory(ROOT) / "olmes-relocated-runtime-import.log"
    with log_path.open("w") as log:
        run([str(executable), "-c",
             "import torch,numpy,pyarrow,tiktoken,kernels;from oe_eval import run_eval;"
             "print({'python':__import__('sys').version.split()[0],'torch':torch.__version__,"
             "'numpy':numpy.__version__,'pyarrow':pyarrow.__version__,'native_olmes_imported':True})"],
            cwd=runtime, env=dict(environment, PYTHONWARNINGS="ignore", HF_HUB_OFFLINE="1"),
            stdout=log, stderr=subprocess.STDOUT)
    sha256 = pack(runtime, artifacts / "gpu-env-olmes.tar.gz")
    report = {
        "gpu_env_sha256": sha256, "bytes": (artifacts / "gpu-env-olmes.tar.gz").stat().st_size,
        "python_version": PYTHON_VERSION, "separate_evaluation_environment": True,
        "training_runtime_preserved": True, "prepared_rental_archives_preserved": True,
        "evaluation_lock_sha256": hashlib.sha256((ROOT / "evaluation/uv.lock").read_bytes()).hexdigest(),
    }
    (profile.results_directory(ROOT) / "olmes-relocatable-runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
