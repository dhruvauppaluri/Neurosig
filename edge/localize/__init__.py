"""Spatial localization engines. Algorithms live in sibling modules, not in viz or scripts."""

from edge.localize.amplitude import AmplitudeLocalizer
from edge.localize.base import LocalizationEngine, NotImplementedLocalizer
from edge.localize.beamform import BeamformLocalizer
from edge.localize.robust import RobustScanLocalizer
from edge.localize.template import TemplateLocalizer
from edge.localize.weighted import WeightedScanLocalizer

LOCALIZATION_METHODS = (
    "amplitude",
    "template",
    "beamform",
    "weighted_scan",
    "robust_scan",
)


def get_default_localizer() -> AmplitudeLocalizer:
    """Viewer and pipeline default: cheapest verified Python engine."""
    return AmplitudeLocalizer()


def get_localizer(method: str, array=None, **kwargs):
    engines = {
        "amplitude": AmplitudeLocalizer,
        "template": TemplateLocalizer,
        "beamform": BeamformLocalizer,
        "weighted_scan": WeightedScanLocalizer,
        "robust_scan": RobustScanLocalizer,
    }
    if method not in engines:
        raise ValueError(f"unknown localization method {method}")
    cls = engines[method]
    if method == "amplitude":
        return cls(**kwargs)
    return cls(array=array, **kwargs)


__all__ = [
    "AmplitudeLocalizer",
    "BeamformLocalizer",
    "LOCALIZATION_METHODS",
    "LocalizationEngine",
    "NotImplementedLocalizer",
    "RobustScanLocalizer",
    "TemplateLocalizer",
    "WeightedScanLocalizer",
    "get_default_localizer",
    "get_localizer",
]
