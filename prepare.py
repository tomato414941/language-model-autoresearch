"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import tarfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

from gpu_config import active_gpu

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = active_gpu().settings()["training_seconds"]
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = str(active_gpu().cache_directory("tokenizer"))
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/4c8f30d6756da75362432a4d5569e1b229263b71"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_01822)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

FINEMATH_REPOSITORY = "HuggingFaceTB/finemath"
FINEMATH_REVISION = "e92b25a616738fe95dc186b64dfb19f9c8525594"
FINEMATH_SUBSETS = ("finemath-3plus", "finemath-4plus")
FINEMATH_DIR = os.path.join(CACHE_DIR, "finemath")
COMMON_PILE_REPOSITORY = "common-pile/stackv2_edu_filtered"
COMMON_PILE_REVISION = "c354dbe88469a1153e97c6a63ac50591849654de"
COMMON_PILE_DIR = os.path.join(CACHE_DIR, "common_pile", "stackv2_edu_filtered")
OLMES_REVISION = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"
OLMES_SUITE_FIELDS = ("qa_bpb", "math_bpb", "code_bpb")
OLMES_REQUEST_DIR = str(active_gpu().cache_directory("olmes"))

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    # Count what's already downloaded
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Additional data downloads
# ---------------------------------------------------------------------------

def _file_sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _additional_manifest(source):
    directory = source["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        metadata = json.loads(path.read_text())
    else:
        base = (f"https://huggingface.co/api/datasets/{source['repository']}/tree/"
                f"{source['revision']}")
        if source["subset"]:
            base += "/" + source["subset"]
        url = base + "?recursive=true&expand=false"
        files = []
        while url:
            if url.split("?", 1)[0] != base:
                raise ValueError("The manifest pagination left its pinned dataset.")
            with requests.get(url, timeout=30) as response:
                response.raise_for_status()
                entries = response.json()
                url = response.links.get("next", {}).get("url")
            for entry in entries:
                if entry["type"] != "file" or not re.fullmatch(source["pattern"], entry["path"]):
                    continue
                source_path = entry["path"]
                files.append(dict(path=source_path, bytes=entry["size"],
                                  sha256=(entry.get("lfs") or {}).get("oid")))
        metadata = {name: source[name] for name in ("repository", "revision", "subset", "license")}
        metadata["files"] = files
    if any(metadata[name] != source[name] for name in ("repository", "revision", "subset")):
        raise ValueError("The cached manifest differs from the approved source.")
    files = metadata["files"]
    if not files or len({entry["path"] for entry in files}) != len(files):
        raise ValueError("The manifest must contain unique training shards.")
    for entry in files:
        if not re.fullmatch(source["pattern"], entry["path"]):
            raise ValueError("The manifest contains an unexpected shard path.")
        if not isinstance(entry["bytes"], int) or entry["bytes"] <= 0:
            raise ValueError("A shard must have a positive byte count.")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("A shard must have its published SHA-256.")
    files.sort(key=lambda entry: entry["path"])
    if not path.exists():
        with tempfile.NamedTemporaryFile("w", dir=directory, suffix=".tmp", delete=False) as output:
            json.dump(metadata, output, indent=2)
            output.write("\n")
        os.replace(output.name, path)
    return files


def _additional_shard_matches(path, entry):
    return (path.is_file() and path.stat().st_size == entry["bytes"]
            and _file_sha256(path) == entry["sha256"])


def _download_additional_shard(source, entry):
    directory = source["directory"]
    path = directory / Path(entry["path"]).name
    if _additional_shard_matches(path, entry):
        return path
    url = (f"https://huggingface.co/datasets/{source['repository']}/resolve/"
           f"{source['revision']}/{entry['path']}?download=true")
    for attempt in range(3):
        temporary = None
        try:
            value = hashlib.sha256()
            size = 0
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile("wb", dir=directory, suffix=".tmp", delete=False) as output:
                    temporary = Path(output.name)
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            size += len(block)
                            if size > entry["bytes"]:
                                raise ValueError("The shard exceeds its published size.")
                            output.write(block)
                            value.update(block)
            if size != entry["bytes"] or value.hexdigest() != entry["sha256"]:
                raise ValueError("The shard differs from its published checksum.")
            os.replace(temporary, path)
            print(f"{source['label']}: downloaded {entry['path']}", flush=True)
            return path
        except (requests.RequestException, OSError, ValueError):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def _download_additional_data(source, num_shards, download_workers):
    if not isinstance(num_shards, int) or (num_shards != -1 and num_shards <= 0):
        raise ValueError("Choose a positive shard count, or -1 for all shards.")
    if not isinstance(download_workers, int) or download_workers <= 0:
        raise ValueError("The download worker count must be positive.")
    files = _additional_manifest(source)
    selected = files if num_shards == -1 else files[:num_shards]
    directory = source["directory"]
    paths = [directory / Path(entry["path"]).name for entry in selected]
    pending = [entry for path, entry in zip(paths, selected, strict=True)
               if not _additional_shard_matches(path, entry)]
    needed = sum(entry["bytes"] for entry in pending)
    if needed > shutil.disk_usage(directory).free:
        raise OSError(f"{source['label']} needs {needed} additional bytes for the selected shards.")
    if pending:
        with ThreadPoolExecutor(max_workers=min(download_workers, len(pending))) as pool:
            list(pool.map(lambda entry: _download_additional_shard(source, entry), pending))
    print(f"{source['label']}: {len(paths)} verified shards ready", flush=True)
    return paths


def download_finemath(subset, num_shards=1, download_workers=4):
    """Download an approved FineMath shard prefix without content filtering."""
    if subset not in FINEMATH_SUBSETS:
        raise ValueError("Choose an approved FineMath subset.")
    source = dict(repository=FINEMATH_REPOSITORY, revision=FINEMATH_REVISION, subset=subset,
                  directory=Path(FINEMATH_DIR) / subset, license="ODC-By-1.0", label="FineMath",
                  pattern=re.escape(subset) + r"/train-\d+-of-\d+\.parquet")
    return _download_additional_data(source, num_shards, download_workers)


def download_common_pile(num_shards=1, download_workers=4):
    """Download original educational code shards, preserving every file's license metadata."""
    source = dict(repository=COMMON_PILE_REPOSITORY, revision=COMMON_PILE_REVISION, subset="",
                  directory=Path(COMMON_PILE_DIR), license="Original per-file open licenses",
                  label="Common Pile", pattern=r"stack-edu-\d+\.json\.gz")
    return _download_additional_data(source, num_shards, download_workers)


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


def bpb_improves(candidate, incumbent):
    """Keep a candidate whose finite val_bpb is lower than the incumbent's."""
    if not math.isfinite(candidate) or not math.isfinite(incumbent):
        raise ValueError("A val_bpb decision requires two finite values.")
    return candidate < incumbent


def suites_improve(candidate, incumbent):
    """Keep a candidate whose BPB falls on average over the suites, each suite counted by its rate of change."""
    if not incumbent or set(candidate) != set(incumbent):
        raise ValueError("A suite decision compares the same suites.")
    rates = []
    for name, base in incumbent.items():
        value = candidate[name]
        if not (math.isfinite(value) and math.isfinite(base) and value > 0 and base > 0):
            raise ValueError("A suite decision requires finite positive BPB values.")
        rates.append((base - value) / base)
    return sum(rates) / len(rates) > 0


def prepare_olmes(count_dir=None, request_dir=None):
    """Prepare every native suite request on the CPU using the isolated OLMES environment."""
    root = Path(__file__).resolve().parent
    executable = os.environ.get("OLMES_PYTHON", str(root / "evaluation/.venv/bin/python"))
    count_dir = Path(count_dir) if count_dir is not None else root / "results/olmes-token-counts"
    request_dir = Path(request_dir) if request_dir is not None else Path(OLMES_REQUEST_DIR)
    common = ["--tokenizer", str(Path(TOKENIZER_DIR) / "tokenizer.pkl")]
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("VAST_", "PROCESSING_"))}
    cache = root / "results/olmes-cache"
    environment.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"),
                       HF_DATASETS_CACHE=str(cache / "datasets"))
    subprocess.run([executable, "-m", "evaluation.count_tokens", "--output", str(count_dir), *common],
                   cwd=root, env=environment, check=True)
    subprocess.run([executable, "-m", "evaluation.prepare_requests", "--counts", str(count_dir),
                    "--output", str(request_dir), *common], cwd=root, env=environment, check=True)
    return request_dir


def evaluate_olmes(model, tokenizer, output_dir="olmes-results"):
    """Evaluate one fixed checkpoint once, using complete native OLMES scoring and macros."""
    from evaluation.arc_macro import APPROVED_CONFIG_FIX
    from evaluation.checkpoint import export_checkpoint

    root = Path(__file__).resolve().parent
    executable = os.environ.get("OLMES_PYTHON", str(root / "evaluation/.venv/bin/python"))
    request_dir = Path(os.environ.get("OLMES_REQUEST_DIR", OLMES_REQUEST_DIR))
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    checkpoint = root / "final-model.pt"
    if checkpoint.exists() or output_dir.exists():
        raise FileExistsError("This experiment already has an OLMES evaluation attempt.")
    source_sha256 = export_checkpoint(model, checkpoint)
    torch.cuda.empty_cache()
    parent_vram_mb = torch.cuda.memory_allocated() / 1024 ** 2
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("VAST_", "PROCESSING_"))}
    with (root / "olmes.log").open("x") as log:
        try:
            subprocess.run([
                executable, "-m", "evaluation.run_prepared", "--checkpoint", str(checkpoint),
                "--tokenizer", str(Path(TOKENIZER_DIR) / "tokenizer.pkl"),
                "--requests", str(request_dir), "--output", os.path.relpath(output_dir, root),
            ], cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        finally:
            if output_dir.is_dir():
                with tarfile.open(root / "olmes-details.tar.gz", "w:gz", compresslevel=1) as archive:
                    archive.add(output_dir, arcname="olmes-results")
    summary = json.loads((output_dir / "summary.json").read_text())
    values = summary["suites"]
    if (not summary["complete"] or summary["olmes_revision"] != OLMES_REVISION
            or summary["approved_config_fix"] != APPROVED_CONFIG_FIX
            or summary["model_source_sha256"] != source_sha256
            or set(values) != set(OLMES_SUITE_FIELDS)
            or not all(math.isfinite(value) and value > 0 for value in values.values())):
        raise ValueError("All three complete native suite results are required.")
    summary["combined_peak_vram_mb"] = parent_vram_mb + summary["peak_vram_mb"]
    (root / "olmes-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(output_dir / "metrics-all.jsonl", root / "olmes-metrics.jsonl")
    return summary

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=70, help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    parser.add_argument("--finemath", nargs="+", choices=FINEMATH_SUBSETS, default=[],
                        help="Download these FineMath subsets instead of the default data/tokenizer prep.")
    parser.add_argument("--finemath-shards", type=int, default=1,
                        help="Training shards per FineMath subset (-1 = all); no content filtering.")
    parser.add_argument("--common-pile", action="store_true",
                        help="Download Common Pile's educational code into its separate directory.")
    parser.add_argument("--common-pile-shards", type=int, default=1,
                        help="Common Pile training shards (-1 = all); preserve original file licenses.")
    parser.add_argument("--olmes", action="store_true",
                        help="Prepare all three pinned OLMES suites on the CPU before rental.")
    parser.add_argument("--olmes-counts-dir", default=None,
                        help="Reuse or write complete native CPU token counts in this directory.")
    args = parser.parse_args(argv)

    if args.finemath or args.common_pile or args.olmes:
        for subset in args.finemath:
            download_finemath(subset, args.finemath_shards, args.download_workers)
        if args.common_pile:
            download_common_pile(args.common_pile_shards, args.download_workers)
        if args.olmes:
            prepare_olmes(count_dir=args.olmes_counts_dir)
        return

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")


if __name__ == "__main__":
    main()
