export type EventKind =
  | "synthetic_source"
  | "neuropixels_spike"
  | "electrode_activity"
  | "detected_spike"
  | "missed_spike"
  | "localization_estimate";

export interface NeuralEvent {
  schema_version: number;
  event_id: string;
  t_s: number;
  xyz_mm: [number, number, number];
  coordinate_frame: string;
  position_quality: string;
  region_id: string;
  amplitude: number;
  amplitude_units: string;
  source_id: string;
  duration_ms: number;
  kind: EventKind;
  channel_id: number | null;
  ground_truth_event_id: string | null;
  metadata: Record<string, unknown> | null;
}

export interface RegionVolume {
  region_id: string;
  label: string;
  center_mm: [number, number, number];
  radii_mm: [number, number, number];
  color: string;
  hemisphere: string;
  homologous_mouse: string | null;
  description: string;
}

export interface Session {
  schema_version: number;
  mode: "synthetic" | "neuropixels" | "hybrid";
  seed: number;
  running: boolean;
  speed: number;
  rate_scale: number;
  t_s: number;
  visible_kinds: EventKind[];
  selected_region: string | null;
  selected_channel: number | null;
  run_golden_model: boolean;
  detector_note: string;
  metrics: {
    simulated_event_count: number;
    recorded_spike_count: number;
    detection_count: number;
    missed_count: number;
    precision: number | null;
    recall: number | null;
    active_channels: number;
    localization_count?: number;
  };
  regions: RegionVolume[];
  electrodes: {
    description: string;
    n_channels: number;
    sites: { channel_id: number; xyz_mm: [number, number, number]; label: string }[];
  };
  neuropixels: {
    dataset: string;
    species: string;
    session_id: string;
    disclaimer: string;
    n_units: number;
    n_channels: number;
    duration_s: number;
    has_raw_voltage: boolean;
    playback_kind: string;
  };
  disclaimer: string;
  localization: {
    available: boolean;
    method: string;
    xyz_mm: [number, number, number] | null;
    error_um: number | null;
    confidence: number | null;
    unverified: boolean;
    reason: string;
  };
  honesty: Record<string, string>;
}

export const KIND_COLORS: Record<EventKind, string> = {
  synthetic_source: "#4cc9f0",
  neuropixels_spike: "#ffd166",
  electrode_activity: "#f72585",
  detected_spike: "#80ffdb",
  missed_spike: "#ff4d6d",
  localization_estimate: "#ffffff",
};

export const KIND_LABELS: Record<EventKind, string> = {
  synthetic_source: "Synthetic source (cyan pulse)",
  neuropixels_spike: "Neuropixels spike (gold pulse)",
  electrode_activity: "Electrode activity (probe flash)",
  detected_spike: "NeuroSig detection (mint octahedron)",
  missed_spike: "Missed spike (red octahedron)",
  localization_estimate: "Localization estimate (white pulse)",
};
