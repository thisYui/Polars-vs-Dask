"""
src/core/experiment_config.py
Loads and validates experiment configuration from YAML files in configs/.
"""

import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


class ExperimentConfig:
    def __init__(self, cfg: dict):
        self.track      = cfg["track"]
        self.sizes      = cfg["sizes"]          # list[int | str]  e.g. [1_000_000] or ["1GB"]
        self.frameworks = cfg["frameworks"]     # list[str]
        self.workloads  = cfg["workloads"]      # list[str]
        self.data_types = cfg["data_types"]     # list[str]  ["real"] | ["synthetic"] | both
        self.repeat     = cfg["repeat"]         # int  — timed runs
        self.warmup     = cfg["warmup"]         # int  — warmup runs (excluded)

        self._validate()

    # ── validation ───────────────────────────────────────────

    def _validate(self) -> None:
        if self.track == "physical":
            assert self.data_types == ["synthetic"], \
                "Physical track must use synthetic data only"

        if self.track == "validation":
            assert len(self.sizes) == 1, \
                "Validation track must use a single fixed size"

    # ── helpers ──────────────────────────────────────────────

    def size_labels(self) -> list[str]:
        """
        Return sizes as string labels suitable for CLI/path lookup.
        Integers are converted to canonical labels (1_000_000 → "1M", etc.).
        String labels (e.g. "1GB") are returned as-is.
        """
        _INT_TO_LABEL = {
            1_000_000:   "1M",
            5_000_000:   "5M",
            10_000_000:  "10M",
            50_000_000:  "50M",
            100_000_000: "100M",
        }
        labels = []
        for s in self.sizes:
            if isinstance(s, int):
                label = _INT_TO_LABEL.get(s)
                if label is None:
                    raise ValueError(
                        f"Unknown integer size {s!r} — add it to _INT_TO_LABEL "
                        f"or use a string label like '10M' in the YAML."
                    )
                labels.append(label)
            else:
                labels.append(str(s))
        return labels

    def __repr__(self) -> str:
        return (
            f"<ExperimentConfig track={self.track!r} "
            f"sizes={self.sizes} frameworks={self.frameworks}>"
        )


# ── loader ───────────────────────────────────────────────────

def load_experiment_config(name: str) -> ExperimentConfig:
    """
    Load a named experiment config from configs/<name>.yaml.

    Args:
        name: stem of the YAML file, e.g. "logical", "physical", "validation"

    Returns:
        Validated ExperimentConfig instance.
    """
    path = CONFIG_DIR / f"{name}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Experiment config not found: {path}\n"
            f"Available configs: {[p.stem for p in CONFIG_DIR.glob('*.yaml')]}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    return ExperimentConfig(cfg)