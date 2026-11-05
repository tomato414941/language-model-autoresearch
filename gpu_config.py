"""User-owned GPU selection, budgets and locations for independent experiment tracks."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "gpus.json"
SETTING_FIELDS = ("training_seconds", "rental_seconds", "max_hourly_cost")


@dataclass(frozen=True)
class GPUProfile:
    identifier: str
    values: dict

    @property
    def gpu_name(self):
        return self.values["gpu_name"]

    def directory(self, root=None):
        return Path(root if root is not None else ROOT) / "gpus" / self.identifier

    def train_source(self, root=None):
        return self.directory(root) / "train.py"

    def results_directory(self, root=None):
        return self.directory(root) / "results"

    def artifacts_directory(self, root=None):
        return self.directory(root) / "artifacts"

    def cache_directory(self, kind):
        return Path.home() / ".cache" / "autoresearch" / "gpus" / self.identifier / kind

    def settings(self, require=True):
        values = {name: self.values.get(name) for name in SETTING_FIELDS}
        missing = [name for name, value in values.items() if value is None]
        if missing:
            if require:
                raise ValueError(
                    f"The user must set {', '.join(missing)} for {self.identifier} in gpus.json "
                    "when first selecting that GPU."
                )
            return values
        for name in ("training_seconds", "rental_seconds"):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer number of seconds.")
        price = values["max_hourly_cost"]
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price < 0:
            raise ValueError("max_hourly_cost must be a finite nonnegative number.")
        if values["training_seconds"] >= values["rental_seconds"]:
            raise ValueError("Training and the complete evaluation must fit inside the rental.")
        return values


def read_config(path=None):
    data = json.loads(Path(path if path is not None else CONFIG_PATH).read_text())
    profiles = data.get("gpus")
    if not isinstance(profiles, dict) or data.get("active_gpu") not in profiles:
        raise ValueError("gpus.json must select exactly one configured active_gpu.")
    for identifier, values in profiles.items():
        if not re.fullmatch(r"[a-z0-9_]+", identifier):
            raise ValueError("GPU identifiers must be simple directory names.")
        if not isinstance(values, dict) or not isinstance(values.get("gpu_name"), str):
            raise ValueError("Every GPU profile must name its GPU.")
    return data


def get_profile(identifier=None, path=None):
    data = read_config(path)
    identifier = data["active_gpu"] if identifier is None else identifier
    if identifier not in data["gpus"]:
        raise ValueError("The requested GPU is not configured.")
    return GPUProfile(identifier, data["gpus"][identifier])


def active_gpu():
    return get_profile()


def profile_for_job(job):
    identifier = job.get("gpu_id")
    if identifier is not None:
        profile = get_profile(identifier)
        if job.get("gpu_name", profile.gpu_name) != profile.gpu_name:
            raise ValueError("The job's GPU identity differs from its track.")
        return profile
    # Before tracks were introduced, this loop used only the H100.
    name = job.get("gpu_name", "H100 SXM")
    for identifier, values in read_config()["gpus"].items():
        if values["gpu_name"] == name:
            return GPUProfile(identifier, values)
    raise ValueError("The historical job's GPU is not configured.")


def source_files(profile=None):
    profile = active_gpu() if profile is None else profile
    return ("train.py", "prepare.py", "program.md", "pyproject.toml", "uv.lock",
            "gpu_config.py", "gpus.json", f"gpus/{profile.identifier}/train.py")
