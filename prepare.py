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
import pickle
import ast
import csv
import hashlib
import importlib.util
import json
import random
import tempfile
import zipfile
from decimal import Decimal
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 1800       # training time budget in seconds (30 minutes, one run in a rental of at most 1 hour)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval
CORE_MIN_DELTA = 0.01    # absolute CORE improvement required for adoption

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/4c8f30d6756da75362432a4d5569e1b229263b71"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_01822)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# The evaluation frame is pinned before the baseline and only changed by the user.
CORE_DIR = os.path.join(CACHE_DIR, "core")
CORE_NANOCHAT_REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
CORE_SOURCE_SHA256 = {
    "nanochat/core_eval.py": "9303997ff17fa48ad5e5730f33b7e79a687cc2a1fdad928666bb63dc3514c51d",
    "scripts/base_eval.py": "644083a99ded710f677b25168ca76fd0416dc7ef3890d346a72f76a20bb522fe",
    "LICENSE": "42bb09f339a8af8b90d3b4969b6f6e4ba4aff3b5c614103b031fccdf71c8f772",
}
CORE_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
CORE_BUNDLE_SHA256 = "90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200"


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_core_file(path, expected):
    if not os.path.isfile(path) or _file_sha256(path) != expected:
        raise RuntimeError(f"CORE asset is missing or changed: {path}. Run prepare.py on the processing server.")


def _download_core_file(url, path, expected):
    if os.path.exists(path):
        _verify_core_file(path, expected)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        _verify_core_file(temp_path, expected)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _core_source_dir():
    return os.path.join(CORE_DIR, "nanochat", CORE_NANOCHAT_REVISION)


def verify_core_assets():
    import yaml

    for name, expected in CORE_SOURCE_SHA256.items():
        _verify_core_file(os.path.join(_core_source_dir(), name), expected)
    bundle_path = os.path.join(CORE_DIR, "eval_bundle.zip")
    _verify_core_file(bundle_path, CORE_BUNDLE_SHA256)
    with zipfile.ZipFile(bundle_path) as archive:
        tasks = yaml.safe_load(archive.read("eval_bundle/core.yaml"))["icl_tasks"]
        if len(tasks) != 22 or len({task["label"] for task in tasks}) != 22:
            raise RuntimeError("CORE requires all 22 upstream tasks.")
        names = ["core.yaml", "eval_meta_data.csv"]
        names.extend("eval_data/" + task["dataset_uri"] for task in tasks)
        for name in names:
            expected = hashlib.sha256(archive.read("eval_bundle/" + name)).hexdigest()
            _verify_core_file(os.path.join(CORE_DIR, "eval_bundle", name), expected)
    return tasks


def prepare_core():
    """Prepare the unchanged nanochat evaluator and its complete official bundle on CPU."""
    source_url = f"https://raw.githubusercontent.com/karpathy/nanochat/{CORE_NANOCHAT_REVISION}"
    for name, expected in CORE_SOURCE_SHA256.items():
        _download_core_file(f"{source_url}/{name}", os.path.join(_core_source_dir(), name), expected)
    bundle_path = os.path.join(CORE_DIR, "eval_bundle.zip")
    _download_core_file(CORE_BUNDLE_URL, bundle_path, CORE_BUNDLE_SHA256)
    bundle_dir = os.path.join(CORE_DIR, "eval_bundle")
    if not os.path.exists(bundle_dir):
        with tempfile.TemporaryDirectory(dir=CORE_DIR) as temp_dir:
            with zipfile.ZipFile(bundle_path) as archive:
                archive.extractall(temp_dir)
            os.rename(os.path.join(temp_dir, "eval_bundle"), bundle_dir)
    tasks = verify_core_assets()
    print(f"CORE: all {len(tasks)} upstream tasks ready (nanochat {CORE_NANOCHAT_REVISION})")


def _load_nanochat_core():
    import yaml

    source_dir = _core_source_dir()
    core_path = os.path.join(source_dir, "nanochat", "core_eval.py")
    spec = importlib.util.spec_from_file_location("_nanochat_core_eval", core_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Load the original aggregation function without importing nanochat's training stack.
    # Both complete source files are byte-for-byte upstream files, verified before loading.
    runner_path = os.path.join(source_dir, "scripts", "base_eval.py")
    with open(runner_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=runner_path)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_core")
    namespace = dict(os=os, csv=csv, time=time, json=json, random=random, yaml=yaml,
                     get_base_dir=lambda: CORE_DIR, print0=print, evaluate_task=module.evaluate_task)
    exec(compile(ast.Module(body=[function], type_ignores=[]), runner_path, "exec"), namespace)
    return namespace["evaluate_core"]


@torch.no_grad()
def evaluate_core(model, tokenizer, device):
    """Run nanochat CORE once, with all examples, original prompts and original scoring."""
    tasks = verify_core_assets()
    evaluator = _load_nanochat_core()
    # Eager execution avoids compiling a different graph for every evaluation prompt.
    model = getattr(model, "_orig_mod", model)
    result = evaluator(model, tokenizer, device, max_per_task=-1)
    labels = {task["label"] for task in tasks}
    if set(result["results"]) != labels or set(result["centered_results"]) != labels:
        raise RuntimeError("CORE did not finish all 22 tasks.")
    scores = [result["core_metric"], *result["results"].values(), *result["centered_results"].values()]
    if not all(math.isfinite(score) for score in scores):
        raise RuntimeError("CORE produced a non-finite score.")
    return result


def save_core_results(result, directory="results"):
    """Preserve nanochat's result dictionary and its native per-task CSV format."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "core.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    with open(os.path.join(directory, "core.csv"), "w", encoding="utf-8", newline="") as f:
        f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        for label in result["results"]:
            acc = result["results"][label]
            centered = result["centered_results"][label]
            f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
        f.write(f"{'CORE':<35}, {'':<10}, {result['core_metric']:<10.6f}\n")


def core_improves(candidate, incumbent):
    """Apply the predeclared absolute threshold to unrounded CORE scores."""
    if not math.isfinite(candidate) or not math.isfinite(incumbent):
        raise ValueError("A CORE decision requires two finite scores.")
    return Decimal(str(candidate)) >= Decimal(str(incumbent)) + Decimal(str(CORE_MIN_DELTA))

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

    # nanochat calls the tokenizer directly; use the exact same encoding method.
    __call__ = encode


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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=70, help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    parser.add_argument("--core-only", action="store_true", help="Prepare only the pinned CORE evaluation assets")
    args = parser.parse_args()

    if args.core_only:
        prepare_core()
        sys.exit(0)

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()

    # Step 3: Prepare the complete public CORE evaluation before renting the GPU.
    prepare_core()
    print()
    print("Done! Ready to train.")
