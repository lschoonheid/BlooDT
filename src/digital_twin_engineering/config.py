"""Pipeline configuration, loaded from a YAML file such as configs/default.yaml."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .channels import Perturbation


@dataclass(frozen=True)
class FeatureConfig:
    windows_s: tuple[int, ...] = (60, 300, 900, 1800)


@dataclass(frozen=True)
class SuggestionConfig:
    # drug -> the setting channel a clinician changes (see docs/adr/0004-...)
    drugs: dict[str, str] = field(
        default_factory=lambda: {
            "propofol": "propofol_target",
            "remifentanil": "remifentanil_target",
        }
    )
    window_s: int = 60  # an intervention within this long after a decision point counts
    min_hold_s: float = 30
    min_relative_change: float = 0.05


@dataclass(frozen=True)
class Config:
    channels: tuple[str, ...]
    experts: tuple[dict[str, Any], ...] = ()
    targets: tuple[str, ...] = ("map", "bis", "hypotension")
    horizons_s: tuple[int, ...] = (300,)
    # how far ahead the twin gives the expected effect of each suggested target change
    effect_horizons_s: tuple[int, ...] = (300,)
    step_s: float = 1.0
    decision_every_s: int = 10
    features: FeatureConfig = FeatureConfig()
    suggestion: SuggestionConfig = SuggestionConfig()
    perturbation: Perturbation | None = None  # training-time sensor faults
    model: dict[str, Any] = field(default_factory=dict)  # LightGBM parameters
    folds: int = 5
    test_fraction: float = 0.2
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "Config":
        """From a YAML config's contents, or `asdict` of a Config (as in a run's config.json)."""
        spec = dict(spec)
        for key in ("channels", "experts", "targets", "horizons_s", "effect_horizons_s"):
            if key in spec:
                spec[key] = tuple(spec[key])
        if "features" in spec:
            spec["features"] = FeatureConfig(tuple(spec["features"]["windows_s"]))
        if "suggestion" in spec:
            spec["suggestion"] = SuggestionConfig(**spec["suggestion"])
        if spec.get("perturbation"):
            spec["perturbation"] = Perturbation(**spec["perturbation"])
        return cls(**spec)

    @property
    def all_channels(self) -> tuple[str, ...]:
        """Configured channels plus every expert input and suggestion setting (an
        input `<channel>__staleness` comes with its channel)."""
        from . import experts
        from .channels import STALENESS

        needed = list(self.channels)
        needed += [n.removesuffix(STALENESS) for spec in self.experts for n in experts.build(spec).inputs]
        needed += list(self.suggestion.drugs.values())
        return tuple(dict.fromkeys(needed))

    def channels_key(self, perturbed: bool) -> dict[str, Any]:
        """The parts of the config a case's channel grid depends on (its cache key)."""
        return {
            "channels": self.all_channels,
            "step_s": self.step_s,
            "perturbation": asdict(self.perturbation) if perturbed and self.perturbation else None,
            "seed": self.seed,
        }

    def expert_key(self, spec: dict[str, Any], perturbed: bool) -> dict[str, Any]:
        """The parts of the config one expert's signals depend on. Channels are
        perturbed independently, so the other channels loaded don't matter."""
        from . import experts

        version = getattr(experts.kinds()[spec["kind"]], "version", 0)
        return {
            **({"code": version} if version else {}),
            "expert": spec,
            "step_s": self.step_s,
            "perturbation": asdict(self.perturbation) if perturbed and self.perturbation else None,
            "seed": self.seed,
        }

    def rows_key(self, perturbed: bool) -> dict[str, Any]:
        """The parts of the config a case's decision-point rows depend on."""
        return {
            "channels": self.channels_key(perturbed),
            "experts": [self.expert_key(spec, perturbed) for spec in self.experts],
            "targets": self.targets,
            "horizons_s": self.horizons_s,
            "decision_every_s": self.decision_every_s,
            "features": asdict(self.features),
            "suggestion": asdict(self.suggestion),
        }
