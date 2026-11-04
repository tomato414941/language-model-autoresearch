"""Fixed data and evaluation for the loop. Do not modify.

The data is the speedrun's FineWeb: modded-nanogpt's GPT-2 tokens of FineWeb's 10B-token sample, published as
kjj0/fineweb10B-gpt2. Each document begins with GPT-2's end-of-text token. The first EVAL_TOKENS tokens of the
validation shard, the speedrun's validation size, are the fixed validation set.

Usage: python prepare.py [--train-shards N]
"""

import argparse
import math
import os
import shutil
import time
import urllib.request
from glob import glob

import numpy as np
import torch

# Fixed constants
MAX_SEQ_LEN = 2048  # tokens in each validation row
TIME_BUDGET = 300  # seconds of training, not counting startup and compilation
EVAL_TOKENS = 10_485_760  # the speedrun's validation size
VOCAB_SIZE = 50257  # GPT-2's tokens
EOT_TOKEN = 50256  # GPT-2's end of text, which begins each document

REPOSITORY = "kjj0/fineweb10B-gpt2"
REVISION = "889765ea1f903759787add96995d81171b632d0c"
TRAIN_SHARD_COUNT = 103  # 100M tokens each
VAL_FILENAME = "fineweb_val_000000.bin"
DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "language-model-autoresearch")
HEADER_INTS = 256
MAGIC = 20240520
VERSION = 1


def shard_tokens(path):
    """Return a shard's tokens as a read-only uint16 array, after checking its header and size."""
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert len(header) == HEADER_INTS and header[0] == MAGIC and header[1] == VERSION, f"{path} is not a token shard"
    count = int(header[2])
    assert os.path.getsize(path) == HEADER_INTS * 4 + 2 * count, f"{path} does not hold {count} tokens"
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_INTS * 4, shape=(count,))


def train_shards():
    """Return the downloaded training shards' tokens, in order. Training reads only these."""
    paths = sorted(glob(os.path.join(DATA_DIR, "fineweb_train_*.bin")))
    assert paths, "no training shards: run prepare.py first"
    return [shard_tokens(path) for path in paths]


def token_bytes():
    """Return each token's length in bytes, with 0 for the end-of-text token."""
    return np.load(os.path.join(DATA_DIR, "token_bytes.npy"))


@torch.no_grad()
def evaluate(model, batch_size):
    """Return (val_bpb, val_loss), the fixed metric. Do not change.

    val_bpb is the cross-entropy of every target but the end-of-text token, in bits, over those targets' bytes,
    so it does not depend on the tokenizer. val_loss is the mean cross-entropy per target in nats, as the speedrun
    reports it. The model is called as model(x, y, reduction="none") and returns each target's cross-entropy.
    """
    tokens = shard_tokens(os.path.join(DATA_DIR, VAL_FILENAME))
    assert len(tokens) > EVAL_TOKENS
    lengths = torch.from_numpy(token_bytes()).cuda()
    rows = EVAL_TOKENS // MAX_SEQ_LEN
    nats, nats_with_bytes, nbytes = 0.0, 0.0, 0
    for first in range(0, rows, batch_size):
        count = min(batch_size, rows - first)
        window = tokens[first * MAX_SEQ_LEN:(first + count) * MAX_SEQ_LEN + 1]
        window = torch.from_numpy(window.astype(np.int64)).cuda()
        x = window[:-1].view(count, MAX_SEQ_LEN)
        y = window[1:].view(count, MAX_SEQ_LEN)
        loss = model(x, y, reduction="none").view(-1).float()
        target_bytes = lengths[y.view(-1)]
        nats += loss.sum().item()
        nats_with_bytes += (loss * (target_bytes > 0)).sum().item()
        nbytes += target_bytes.sum().item()
    return nats_with_bytes / (math.log(2) * nbytes), nats / (rows * MAX_SEQ_LEN)


def download(filename, attempts=5):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{filename}"
    partial = path + ".partial"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as f:
                shutil.copyfileobj(response, f, length=1 << 20)
            shard_tokens(partial)
            os.replace(partial, path)
            print(f"downloaded {filename}")
            return
        except (OSError, AssertionError) as error:
            print(f"attempt {attempt} for {filename} failed: {error}")
            if os.path.exists(partial):
                os.remove(partial)
            if attempt < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {filename}")


def write_token_bytes():
    path = os.path.join(DATA_DIR, "token_bytes.npy")
    if os.path.exists(path):
        return
    import tiktoken

    encoding = tiktoken.get_encoding("gpt2")
    lengths = [len(encoding.decode_single_token_bytes(token)) for token in range(EOT_TOKEN)] + [0]
    assert len(lengths) == VOCAB_SIZE
    np.save(path, np.array(lengths, dtype=np.int64))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the speedrun's FineWeb tokens")
    parser.add_argument("--train-shards", type=int, default=10, help="training shards of 100M tokens to download")
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    download(VAL_FILENAME)
    for index in range(1, min(args.train_shards, TRAIN_SHARD_COUNT) + 1):
        download(f"fineweb_train_{index:06d}.bin")
    write_token_bytes()
    print(f"ready in {DATA_DIR}")
