"""Record and replay native OLMES tokenization while keeping native inference intact."""

from array import array
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sqlite3
import struct
import sys
from types import MethodType


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def model_signature(model):
    from evaluation.tokenizer_adapter import TokenizerAdapter

    tokenizer = model.tokenizer
    encoding = tokenizer.encoding
    digest = hashlib.sha256(encoding._pat_str.encode())
    for token, rank in sorted(encoding._mergeable_ranks.items()):
        digest.update(struct.pack("<II", len(token), rank))
        digest.update(token)
    for token, rank in sorted(encoding._special_tokens.items()):
        value = token.encode()
        digest.update(struct.pack("<II", len(value), rank))
        digest.update(value)
    methods = "\n".join(inspect.getsource(getattr(model, name)) for name in
                        ("_encode_pair", "tok_encode", "_mask_tokens", "loglikelihood_verbose"))
    return dict(vocab_size=model.vocab_size, encoding_sha256=digest.hexdigest(),
                tokenizer_class=type(tokenizer).__name__,
                tokenizer_adapter_sha256=hashlib.sha256(inspect.getsource(TokenizerAdapter).encode()).hexdigest(),
                native_encoding_source_sha256=hashlib.sha256(methods.encode()).hexdigest(),
                tiktoken_version=version("tiktoken"),
                model_class=model.AUTO_MODEL_CLASS.__name__, add_bos_token=model.add_bos_token,
                custom_prefix_token_id=model.custom_prefix_token_id, prefix_token_id=model.prefix_token_id,
                bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)


def input_identity(model, tokenizer_path, request_manifest):
    return dict(tokenizer_sha256=file_digest(tokenizer_path),
                request_manifest_sha256=file_digest(request_manifest), model=model_signature(model))


def request_key(kind, values):
    payload = json.dumps([kind, values], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).digest()


@contextmanager
def use_encodings(model, cache):
    """Temporarily wrap only native encoding methods, restoring them even after failure."""
    if model_signature(model) != cache.identity["model"]:
        raise ValueError("The cache and model must use the same native encoder settings.")
    missing = object()
    saved = {name: model.__dict__.get(name, missing) for name in ("_encode_pair", "tok_encode")}
    native_pair, native_single = model._encode_pair, model.tok_encode
    inside_pair = False

    def pair(instance, context, continuation):
        nonlocal inside_pair
        values = [context, continuation]
        if isinstance(cache, EncodingReader):
            return tuple(cache.get("pair", values))
        inside_pair = True
        try:
            result = native_pair(context, continuation)
        finally:
            inside_pair = False
        cache.put("pair", values, result)
        return result

    def single(instance, *args, **kwargs):
        if inside_pair:
            return native_single(*args, **kwargs)
        values = [args, kwargs]
        if isinstance(cache, EncodingReader):
            return cache.get("single", values)[0]
        result = native_single(*args, **kwargs)
        cache.put("single", values, [result])
        return result

    model._encode_pair, model.tok_encode = MethodType(pair, model), MethodType(single, model)
    try:
        yield cache
    finally:
        for name, value in saved.items():
            if value is missing:
                delattr(model, name)
            else:
                setattr(model, name, value)


class EncodingWriter:
    def __init__(self, path, identity):
        self.path, self.identity = Path(path), identity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.open("xb").close()
        self.connection = sqlite3.connect(self.path)
        self.code = "h" if identity["model"]["vocab_size"] <= 32768 else "i"
        self.metadata = dict(format_version=1, complete=False, identity=identity, dtype=self.code,
                             byte_order="little", itemsize=array(self.code).itemsize)
        self.calls = self.stored_tokens = 0
        self.connection.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE encodings (key BLOB PRIMARY KEY, first BLOB NOT NULL, second BLOB)")
        self.connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(self.metadata),))
        self.connection.commit()

    def put(self, kind, values, vectors):
        if len(vectors) != (2 if kind == "pair" else 1):
            raise ValueError("Store the complete native encoding vectors for each request.")
        blobs = []
        for vector in vectors:
            values_array = array(self.code, vector)
            if sys.byteorder != "little":
                values_array.byteswap()
            blobs.append(values_array.tobytes())
        cursor = self.connection.execute("INSERT OR IGNORE INTO encodings VALUES (?, ?, ?)",
                                         (request_key(kind, values), blobs[0], blobs[1] if len(blobs) == 2 else None))
        self.calls += 1
        if cursor.rowcount:
            self.stored_tokens += sum(map(len, vectors))

    def finish(self):
        self.metadata.update(complete=True, recorded_calls=self.calls, stored_tokens=self.stored_tokens,
                             unique_encodings=self.connection.execute("SELECT count(*) FROM encodings").fetchone()[0])
        self.connection.execute("UPDATE metadata SET value = ?", (json.dumps(self.metadata),))
        self.connection.commit()
        self.close()

    def close(self):
        self.connection.close()


class EncodingReader:
    def __init__(self, path, identity):
        self.path, self.identity = Path(path), identity
        self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            self.metadata = json.loads(self.connection.execute("SELECT value FROM metadata").fetchone()[0])
            self.code = self.metadata["dtype"]
            if (self.metadata["format_version"] != 1 or not self.metadata["complete"]
                    or self.metadata["identity"] != identity or self.code not in ("h", "i")
                    or self.metadata["byte_order"] != "little"
                    or array(self.code).itemsize != self.metadata["itemsize"]):
                raise ValueError("Use a complete native encoding cache matching these inputs.")
        except BaseException:
            self.connection.close()
            raise
        self.hits = 0

    def get(self, kind, values):
        row = self.connection.execute("SELECT first, second FROM encodings WHERE key = ?",
                                      (request_key(kind, values),)).fetchone()
        if row is None:
            raise ValueError("The native encoding cache does not contain this request.")
        if (row[1] is None) != (kind == "single"):
            raise ValueError("The native encoding cache has incomplete request vectors.")
        result = []
        for payload in row[:1] if kind == "single" else row:
            values_array = array(self.code, payload)
            if sys.byteorder != "little":
                values_array.byteswap()
            result.append(values_array.tolist())
        self.hits += 1
        return result

    def close(self):
        self.connection.close()
