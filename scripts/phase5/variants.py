"""Named single-axis experiments. Every entry differs from BASELINE in one field
unless it is in a combo set (built later from adopted singles)."""

from scripts.phase5.detectors import BASELINE as B

SINGLES = {
    "baseline": B,
    # common reference
    "car_none": B.replace(car="none"),
    "car_mean": B.replace(car="mean"),
    "car_trimmed": B.replace(car="trimmed"),
    # threshold level
    "thr_3.5": B.replace(threshold_sigma=3.5),
    "thr_4.5": B.replace(threshold_sigma=4.5),
    "thr_5.0": B.replace(threshold_sigma=5.0),
    # noise estimators / adaptivity
    "noise_mad": B.replace(noise="mad"),
    "noise_rms_clip": B.replace(noise="rms_clip"),
    "noise_iqr": B.replace(noise="iqr"),
    "noise_ema_causal": B.replace(noise="ema"),
    # shape / amplitude gates
    "width_ge2": B.replace(min_width_samples=2),
    "width_ge3": B.replace(min_width_samples=3),
    "amp_cap_30sigma": B.replace(max_amplitude_sigma=30.0),
    # artifact veto
    "artifact_0.25": B.replace(artifact_channel_fraction=0.25),
    "artifact_0.75": B.replace(artifact_channel_fraction=0.75),
    "artifact_off": B.replace(artifact_channel_fraction=2.0),
    # peak alignment / refractory
    "peak_search_0.25ms": B.replace(peak_search_ms=0.25),
    "peak_search_1.0ms": B.replace(peak_search_ms=1.0),
    "refractory_0.5ms": B.replace(refractory_ms=0.5),
    "refractory_2.0ms": B.replace(refractory_ms=2.0),
    # event merging
    "merge_global_0.25ms": B.replace(merge_ms=0.25),
    "merge_global_1.0ms": B.replace(merge_ms=1.0),
    "merge_nms_global": B.replace(merge="nms", merge_radius_um=0.0),
    "merge_nms_60um": B.replace(merge="nms", merge_radius_um=60.0),
    "merge_nms_100um": B.replace(merge="nms", merge_radius_um=100.0),
    # filter band
    "hp_250": B.replace(highpass_hz=250.0),
    "hp_400": B.replace(highpass_hz=400.0),
    "hp_500": B.replace(highpass_hz=500.0),
    "lp_4500": B.replace(lowpass_hz=4500.0),
    "lp_9000": B.replace(lowpass_hz=9000.0),
    "lp_3500": B.replace(lowpass_hz=3500.0),
    "lp_3000": B.replace(lowpass_hz=3000.0),
    "hp_200": B.replace(highpass_hz=200.0),
    # spatial combination of adjacent contacts
    "spatial_w0.5_40um": B.replace(spatial_weight=0.5, spatial_radius_um=40.0),
    "spatial_w0.5_60um": B.replace(spatial_weight=0.5, spatial_radius_um=60.0),
    "spatial_w1.0_40um": B.replace(spatial_weight=1.0, spatial_radius_um=40.0),
}

VARIANT_SETS = {"singles": SINGLES}

# Round 2 (selection split): refine the one adopted single (hp_200) and try near-miss combinations.
# The reference for the adoption rule in this round is "hp_200", not the frozen baseline.
H = B.replace(highpass_hz=200.0)
ROUND2 = {
    "baseline": B,
    "hp_200": H,
    "hp_150": B.replace(highpass_hz=150.0),
    "hp_100": B.replace(highpass_hz=100.0),
    "hp_200+amp_cap_30": H.replace(max_amplitude_sigma=30.0),
    "hp_200+noise_ema": H.replace(noise="ema"),
    "hp_200+lp_4500": H.replace(lowpass_hz=4500.0),
    "hp_200+car_trimmed": H.replace(car="trimmed"),
    "hp_200+merge_nms_100um": H.replace(merge="nms", merge_radius_um=100.0),
}
VARIANT_SETS["round2"] = ROUND2
