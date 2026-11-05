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
from oe_eval.components.requests import GenerateUntilAndLoglikelihoodRequest
from oe_eval.utils import hash_dict, save_jsonl

from evaluation.count_tokens import OLMES_REVISION, SUITES
from evaluation.arc_macro import configure_aggregation
from evaluation.gold_bpb import GoldAnswerBPB, gold_request_indices

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
    instances = []
    for item in raw:
        if item["request_type"] == "loglikelihood" and "perplexity_context" in item["request"]:
            # Native gold-BPB tasks retain their richer generation request object.
            # Restore that original class and every field while keeping its likelihood request type.
            values = dict(item)
            values["request"] = GenerateUntilAndLoglikelihoodRequest(**item["request"])
            instances.append(RequestInstance(**values))
        else:
            instances.append(RequestInstance.from_dict(item))
    return raw, instances


def score_task(model, entry, raw_requests, instances, model_config, compute_config):
    started = time.monotonic()
    task = run_eval.load_task(entry["native_config"])
    if task._task_hash != entry["task_hash"]:
        raise ValueError("The native task configuration differs from its prepared requests.")
    task.task_config = deepcopy(entry["resolved_task_config"])
    task.set_model(model, model_config)
    selected = gold_request_indices(instances, task.task_config)
    if len(selected) != entry["examples"]:
        raise ValueError("Every original example must retain its correct-answer measurement.")
    task._instances = [instances[index] for index in selected]
    results = run_eval.evaluate(model, task._instances, task.task_config, model_config)
    if len(results) != len(selected):
        raise ValueError("Inference did not return every correct-answer measurement.")
    task.model = None
    task.make_metrics()
    task._metrics = [GoldAnswerBPB(**(task.task_config["metric_kwargs"] or {}))]
    model_hash = hash_dict(model_config, run_eval.MODEL_DEFAULTS)
    full_config = {
        "task_name": task.task_name, "task_hash": task._task_hash["hash"],
        "model_hash": model_hash["hash"], "model_config": model_config,
        "task_config": task.task_config, "compute_config": deepcopy(compute_config),
        "processing_time": time.monotonic() - started,
        "current_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "num_instances": 0, "beaker_info": {},
        "computation": {"method": "correct_answer_only", "source_requests": entry["requests"],
                        "computed_requests": len(selected)},
    }
    metrics = run_eval.compute_save_metrics(
        entry["index"], task, full_config, compute_config, [raw_requests[index] for index in selected], results)
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
    manifest["computed_requests"] = sum(task["computation"]["computed_requests"] for task in metrics)
    suites, aggregates = aggregate_suites(metrics, [entry["alias"] for entry in manifest["tasks"]])
    save_jsonl(output / "metrics-all.jsonl", aggregates + metrics)
    return suites, manifest
