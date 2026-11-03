"""Edge decoding experiments. Reuse scripts.synthetic_decoder.train; do not replace RTL decoder."""

from edge.decode.experiment import build_report, run_neuropixels_count_only, run_synthetic_comparison
from edge.decode.features import SPATIAL_DIM, baseline_counts, spatial_vector, stack_baseline, stack_spatial
from edge.decode.model import logistic_confidence, predict_labels, train
from edge.decode.spatial_task import generate_dataset

__all__ = [
    "SPATIAL_DIM",
    "baseline_counts",
    "build_report",
    "generate_dataset",
    "logistic_confidence",
    "predict_labels",
    "run_neuropixels_count_only",
    "run_synthetic_comparison",
    "spatial_vector",
    "stack_baseline",
    "stack_spatial",
    "train",
]
