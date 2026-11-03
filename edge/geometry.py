"""Scientific 32-site linear probe in micrometers. Display mapping lives in edge.coords."""

from __future__ import annotations

from edge.coords import SCIENTIFIC_PITCH_UM
from edge.types import ElectrodeArray, ElectrodeSite


def linear_probe_um(
    n_channels: int = 32,
    pitch_um: float = SCIENTIFIC_PITCH_UM,
    description: str = "Stylized 32-site linear probe, 20 µm pitch (not a fabricated device)",
) -> ElectrodeArray:
    sites = tuple(
        ElectrodeSite(
            channel_id=channel,
            xyz_um=(0.0, channel * pitch_um, 0.0),
            label=f"ch{channel:02d}",
        )
        for channel in range(n_channels)
    )
    return ElectrodeArray(sites=sites, description=description, frame="probe_um")
