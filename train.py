"""Run the user-selected GPU's independent training script."""

import runpy

from gpu_config import active_gpu


def main():
    profile = active_gpu()
    profile.settings()
    runpy.run_path(str(profile.train_source()), run_name="__main__")


if __name__ == "__main__":
    main()
