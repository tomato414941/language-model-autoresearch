"""Run native OLMES scoring and aggregation on its prebuilt request objects."""

from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from oe_eval import run_eval
from oe_eval.components.instances import RequestInstance
from oe_eval.utils import hash_dict, save_jsonl

from evaluation.count_tokens import OLMES_REVISION, SUITES
from evaluation.arc_macro import configure_aggregation

SUITE_FIELDS = dict(zip(SUITES, ("qa_bpb", "math_bpb", "code_bpb"), strict=True))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_requests(directory, entry):
    path = Path(directory) / entry["requests_file"]
    if sha256_file(path) != entry["requests_sha256"]:
        raise ValueError("A prepared native request file is incomplete.")
    digest = hashlib.sha256()
    raw = []
    with gzip.open(path, "rb") as source:
        for line in source:
            digest.update(line)
            raw.append(json.loads(line))
    if digest.hexdigest() != entry["uncompressed_requests_sha256"] or len(raw) != entry["requests"]:
        raise ValueError("The native request payload differs from its CPU preparation.")
    return raw, [RequestInstance.from_dict(instance) for instance in raw]


def score_task(model, entry, raw_requests, instances, model_config, compute_config):
    started = time.monotonic()
    task = run_eval.load_task(entry["native_config"])
    if task._task_hash != entry["task_hash"]:
        raise ValueError("The native task configuration differs from its prepared requests.")
    task.task_config = deepcopy(entry["resolved_task_config"])
    task.set_model(model, model_config)
    task._instances = instances
    results = run_eval.evaluate(model, instances, task.task_config, model_config)
    if len(results) != entry["requests"]:
        raise ValueError("Native inference did not return every request.")
    task.model = None
    # These are the native runner's metric initialization steps, including gold-answer BPB.
    task.make_metrics()
    if task.task_config["primary_metric"] == "bits_per_byte_corr":
        if not task.task_config["compute_gold_bpb"]:
            raise ValueError("The native gold-answer BPB configuration is required.")
        task._metrics = [run_eval.MCAccuracy(**task.task_config["metric_kwargs"])]
    if not task._metrics or any(metric.deferred_metric for metric in task._metrics):
        raise ValueError("These complete BPB suites require native immediate metrics.")
    model_hash = hash_dict(model_config, run_eval.MODEL_DEFAULTS)
    full_config = {
        "task_name": task.task_name, "task_hash": task._task_hash["hash"],
        "model_hash": model_hash["hash"], "model_config": model_config,
        "task_config": task.task_config, "compute_config": deepcopy(compute_config),
        "processing_time": time.monotonic() - started,
        "current_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "num_instances": 0, "beaker_info": {},
    }
    metrics = run_eval.compute_save_metrics(
        entry["index"], task, full_config, compute_config, raw_requests, results)
    value = metrics["metrics"]["primary_score"]
    if not math.isfinite(value) or value < 0:
        raise ValueError("A native task did not produce finite nonnegative BPB.")
    return metrics


def aggregate_suites(metrics, expected_aliases):
    actual_aliases = [task["task_config"]["metadata"]["alias"] for task in metrics]
    if len(actual_aliases) != len(set(actual_aliases)) or set(actual_aliases) != set(expected_aliases):
        raise ValueError("All native leaf tasks must finish exactly once.")
    configure_aggregation()
    aggregates = run_eval.add_aggregate_tasks(metrics)
    selected = {task["task_config"]["metadata"]["alias"]: task for task in aggregates
                if task["task_config"]["metadata"]["alias"] in SUITE_FIELDS}
    if set(selected) != set(SUITES):
        raise ValueError("All three complete native suite aggregates are required.")
    suites = {}
    for alias, field in SUITE_FIELDS.items():
        value = float(selected[alias]["metrics"]["primary_score"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Every native suite must produce finite positive BPB.")
        suites[field] = value
    return suites, aggregates


def evaluate_prepared(model, directory, output):
    directory, output = Path(directory), Path(output)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["complete"] or manifest["olmes_revision"] != OLMES_REVISION or tuple(manifest["suites"]) != SUITES:
        raise ValueError("A complete pinned native request bundle is required.")
    if model.max_length != manifest["max_length"] or model.batch_size != manifest["batch_size"]:
        raise ValueError("The model connection must match the CPU runtime estimate.")
    output.mkdir(parents=True, exist_ok=False)
    config = deepcopy(manifest["native_config"])
    compute = config["compute_config"]
    compute["output_dir"] = str(output)
    metrics = []
    for entry in manifest["tasks"]:
        raw, instances = read_requests(directory, entry)
        metrics.append(score_task(model, entry, raw, instances, config["model_config"], compute))
    suites, aggregates = aggregate_suites(metrics, [entry["alias"] for entry in manifest["tasks"]])
    save_jsonl(output / "metrics-all.jsonl", aggregates + metrics)
    return suites, manifest
