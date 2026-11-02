/**
 * Brain Project GLB coordinates are metres: +X is anatomical left, +Y is
 * superior, and +Z is anterior. The viewer uses millimetres: +X right,
 * +Y anterior, +Z superior. The origin is near the commissural midpoint.
 *
 * A uniform 0.84 scale fits the existing illustrative region volumes. This
 * is a visual registration, not an atlas-to-atlas or clinical transform.
 */
export const ASSET_MM_SCALE = 840;
export const ASSET_Y_ORIGIN_M = 1.628;
export const VIEWER_ANTERIOR_OFFSET_MM = 7;

export type Point3 = [number, number, number];

export function assetToViewer([x, y, z]: Point3): Point3 {
  return [-ASSET_MM_SCALE * x, ASSET_MM_SCALE * z + VIEWER_ANTERIOR_OFFSET_MM, ASSET_MM_SCALE * (y - ASSET_Y_ORIGIN_M)];
}

export function viewerToAsset([x, y, z]: Point3): Point3 {
  return [-x / ASSET_MM_SCALE, z / ASSET_MM_SCALE + ASSET_Y_ORIGIN_M, (y - VIEWER_ANTERIOR_OFFSET_MM) / ASSET_MM_SCALE];
}

export function probeSpan(sites: { xyz_mm: Point3 }[]): number {
  if (sites.length < 2) return 0;
  const a = sites[0].xyz_mm;
  const b = sites[sites.length - 1].xyz_mm;
  return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
}
