import { REGION_DOWNSTREAM, REGION_TARGETS, type Vec3 } from "./brainGeometry.ts";
import type { EventKind, NeuralEvent, Session } from "./types.ts";

/** Travel animation depicts only the toy whole-brain generator. */
export function shouldAnimateTravel(kind: EventKind): boolean {
  return kind === "synthetic_source";
}

export function destinationFor(event: NeuralEvent, session: Session): Vec3 {
  if (event.kind === "electrode_activity" || event.kind === "detected_spike" || event.kind === "neuropixels_spike") {
    const site = session.electrodes.sites.find((entry) => entry.channel_id === event.channel_id);
    if (site) return site.xyz_mm;
  }
  if (event.region_id === "visual_cortex" && session.electrodes.sites.length) {
    let best = session.electrodes.sites[0];
    let bestD = Infinity;
    for (const site of session.electrodes.sites) {
      const dx = site.xyz_mm[0] - event.xyz_mm[0];
      const dy = site.xyz_mm[1] - event.xyz_mm[1];
      const dz = site.xyz_mm[2] - event.xyz_mm[2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = site;
      }
    }
    return best.xyz_mm;
  }
  const downstream = REGION_DOWNSTREAM[event.region_id];
  return REGION_TARGETS[downstream] ?? REGION_TARGETS.thalamus;
}
