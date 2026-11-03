"""LocalizationEngine protocol and placeholder used until algorithms are wired."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from edge.types import LocalizationEstimate, Snippet


@runtime_checkable
class LocalizationEngine(Protocol):
    method: str

    def estimate(self, snippet: Snippet) -> LocalizationEstimate:
        ...


class NotImplementedLocalizer:
    method = "not_implemented"

    def estimate(self, snippet: Snippet) -> LocalizationEstimate:
        return LocalizationEstimate(
            available=False,
            method=self.method,
            xyz_um=None,
            ground_truth_xyz_um=snippet.ground_truth_xyz_um,
            error_um=None,
            latency_s=0.0,
            confidence=None,
            reason="Localization not implemented",
            event_id=snippet.event_id,
        )
