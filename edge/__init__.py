"""Edge-computer localization, ASIC link, and decoding. Isolated from RTL and DSP arithmetic."""

from __future__ import annotations

SEED = 20261103
SAMPLE_RATE_HZ = 32_000.0
N_CHANNELS = 32
SNIPPET_PRE_MS = 1.0
SNIPPET_POST_MS = 1.5
