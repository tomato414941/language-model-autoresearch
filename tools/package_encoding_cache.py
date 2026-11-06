"""Create a separate deployment variant containing exact native OLMES encodings."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_config import active_gpu, get_profile, read_config
from tools.runtime_archives import archive_specs, encoding_cache_asset_path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


class HashingReader:
    def __init__(self, source):
        self.source, self.digest = source, hashlib.sha256()

    def read(self, size=-1):
        value = self.source.read(size)
        self.digest.update(value)
        return value


def member_identity(member, payload_sha256=None):
    result = dict(name=member.name, type=member.type.decode("ascii"), size=member.size,
                  mode=member.mode, uid=member.uid, gid=member.gid, uname=member.uname,
                  gname=member.gname, mtime=member.mtime, linkname=member.linkname,
                  pax_headers={key: value for key, value in member.pax_headers.items()
                               if key not in {"path", "linkpath", "size", "uid", "gid", "uname", "gname", "mtime"}})
    if payload_sha256 is not None:
        result["sha256"] = payload_sha256
    return result


def add_cache(source, cache, destination, member_name):
    """Preserve every original member and verify the new archive in a second pass."""
    expected = []
    with tarfile.open(source, "r|gz") as original, \
            destination.open("xb") as output, \
            tarfile.open(fileobj=output, mode="w|gz", compresslevel=1) as target:
        for member in original:
            if str(PurePosixPath(member.name)) == member_name:
                raise ValueError("The original assets already contain an encoding cache.")
            if member.isfile():
                with original.extractfile(member) as payload:
                    reader = HashingReader(payload)
                    target.addfile(member, reader)
                expected.append(member_identity(member, reader.digest.hexdigest()))
            else:
                target.addfile(member)
                expected.append(member_identity(member))
        member = tarfile.TarInfo(member_name)
        member.size, member.mode = cache.stat().st_size, 0o600
        with cache.open("rb") as payload:
            reader = HashingReader(payload)
            target.addfile(member, reader)
        expected.append(member_identity(member, reader.digest.hexdigest()))
    actual = []
    with tarfile.open(destination, "r|gz") as archive:
        for member in archive:
            payload_sha256 = None
            if member.isfile():
                value = hashlib.sha256()
                with archive.extractfile(member) as payload:
                    for block in iter(lambda: payload.read(1024**2), b""):
                        value.update(block)
                payload_sha256 = value.hexdigest()
            actual.append(member_identity(member, payload_sha256))
    if actual != expected:
        raise ValueError("The new assets must preserve every original member and payload.")
    return dict(original_members=len(expected) - 1, cache_sha256=expected[-1]["sha256"],
                cache_bytes=cache.stat().st_size, original_members_and_payloads_unchanged=True)


def verify_inputs(assets, cache, runtime, gpu_id, temporary):
    from evaluation.count_tokens import OLMES_REVISION, SUITES, TokenObserver
    from evaluation.encoding_cache import EncodingReader, input_identity
    from evaluation.tokenizer_adapter import TokenizerAdapter

    prefix = f"autoresearch/gpus/{gpu_id}"
    names = {f"{prefix}/olmes/manifest.json": "manifest.json",
             f"{prefix}/tokenizer/tokenizer.pkl": "tokenizer.pkl",
             f"{prefix}/tokenizer/token_bytes.pt": "token_bytes.pt"}
    remaining = set(names)
    with tarfile.open(assets, "r|gz") as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            if name in names:
                if name not in remaining or not member.isfile():
                    raise ValueError("The assets must contain one original request manifest and tokenizer.")
                with archive.extractfile(member) as source:
                    (temporary / names[name]).write_bytes(source.read())
                remaining.remove(name)
    if remaining:
        raise ValueError("The assets must contain the original request manifest and tokenizer.")
    manifest_path, tokenizer_path = temporary / "manifest.json", temporary / "tokenizer.pkl"
    manifest = json.loads(manifest_path.read_text())
    native = runtime["olmes"]
    if (not manifest["complete"] or manifest["olmes_revision"] != OLMES_REVISION
            or tuple(manifest["suites"]) != SUITES or native["revision"] != OLMES_REVISION
            or native["request_manifest_sha256"] != digest(manifest_path)
            or native["leaf_tasks"] != len(manifest["tasks"])
            or native["examples"] != manifest["total_examples"]
            or native["requests"] != manifest["total_requests"]):
        raise ValueError("The original assets and runtime must describe the same complete native suites.")
    for name in ("tokenizer.pkl", "token_bytes.pt"):
        if digest(temporary / name) != runtime["tokenizer"][name]:
            raise ValueError("The original assets and runtime must use the same tokenizer and byte table.")
    observer = TokenObserver(TokenizerAdapter(tokenizer_path, manifest["max_length"]),
                             manifest["max_length"], manifest["batch_size"])
    reader = EncodingReader(cache, input_identity(observer, tokenizer_path, manifest_path))
    try:
        if reader.metadata["recorded_calls"] != manifest["total_examples"]:
            raise ValueError("The encoding cache must cover every native example.")
        return reader.metadata
    finally:
        reader.close()


def package(runtime_path, cache, destination, *, gpu=None):
    runtime_path, cache, destination = map(Path, (runtime_path, cache, destination))
    original_bytes = runtime_path.read_bytes()
    runtime = json.loads(original_bytes)
    profile = get_profile(gpu) if gpu else active_gpu()
    if runtime.get("gpu_id") != profile.identifier or runtime.get("evaluation_metric") != "olmes_bpb":
        raise ValueError("Choose the selected GPU's native OLMES runtime.")
    if destination.exists():
        raise FileExistsError("Keep every existing runtime variant and prepared experiment.")
    assets = runtime_path.parent / "gpu-assets.tar.gz"
    if digest(assets) != runtime["gpu_assets_sha256"]:
        raise ValueError("The original evaluation assets differ from their runtime.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".cache-bindings-") as temporary:
        metadata = verify_inputs(assets, cache, runtime, profile.identifier, Path(temporary))
    destination.mkdir()
    name = encoding_cache_asset_path(profile.identifier)
    verification = add_cache(assets, cache, destination / "gpu-assets.tar.gz", name)
    for filename, _ in archive_specs(runtime):
        if filename != "gpu-assets.tar.gz":
            os.link(runtime_path.parent / filename, destination / filename)
    packaged = deepcopy(runtime)
    packaged["gpu_assets_sha256"] = digest(destination / "gpu-assets.tar.gz")
    packaged["encoding_cache"] = dict(format_version=metadata["format_version"], asset_path=name,
                                      sha256=verification["cache_sha256"], bytes=verification["cache_bytes"],
                                      identity=metadata["identity"])
    (destination / "runtime.json").write_text(json.dumps(packaged, indent=2) + "\n")
    report = dict(complete=True, gpu_id=profile.identifier, quality_evaluations=0, selected_for_rental=False,
                  base_runtime_sha256=hashlib.sha256(original_bytes).hexdigest(),
                  base_assets_sha256=runtime["gpu_assets_sha256"],
                  gpu_assets_sha256=packaged["gpu_assets_sha256"],
                  base_assets_bytes=assets.stat().st_size,
                  packaged_assets_bytes=(destination / "gpu-assets.tar.gz").stat().st_size,
                  encoding_cache=packaged["encoding_cache"], **verification)
    (destination / "variant.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=tuple(read_config()["gpus"]))
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(package(args.runtime, args.cache, args.output, gpu=args.gpu)), flush=True)


if __name__ == "__main__":
    main()
