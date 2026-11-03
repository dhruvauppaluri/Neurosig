"""Spatial localization engines. Algorithms live in sibling modules, not in viz or scripts."""

from edge.localize.amplitude import AmplitudeLocalizer
from edge.localize.base import LocalizationEngine, NotImplementedLocalizer
from edge.localize.beamform import BeamformLocalizer
from edge.localize.template import TemplateLocalizer


def get_default_localizer() -> AmplitudeLocalizer:
    """Viewer and pipeline default: cheapest verified Python engine."""
    return AmplitudeLocalizer()


__all__ = [
    "AmplitudeLocalizer",
    "BeamformLocalizer",
    "LocalizationEngine",
    "NotImplementedLocalizer",
    "TemplateLocalizer",
    "get_default_localizer",
]
