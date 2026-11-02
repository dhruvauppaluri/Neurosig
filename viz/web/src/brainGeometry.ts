import * as THREE from "three";

export type Vec3 = [number, number, number];

export interface Tract {
  id: string;
  points: Vec3[];
  label: string;
}

export const TRACTS: Tract[] = [
  {
    id: "callosal",
    label: "Callosal (stylized)",
    points: [
      [-42, 36, 16],
      [-18, 22, 10],
      [0, 18, 8],
      [18, 22, 10],
      [42, 36, 16],
    ],
  },
  {
    id: "corticothalamic_l",
    label: "Corticothalamic left (stylized)",
    points: [
      [-32, 8, 40],
      [-18, 0, 18],
      [0, -8, 4],
    ],
  },
  {
    id: "corticothalamic_r",
    label: "Corticothalamic right (stylized)",
    points: [
      [32, 8, 40],
      [18, 0, 18],
      [0, -8, 4],
    ],
  },
  {
    id: "visual_radiation",
    label: "Optic radiation (stylized)",
    points: [
      [0, -8, 4],
      [16, -28, 6],
      [8, -62, 8],
      [0, -72, 8],
    ],
  },
  {
    id: "fornix_l",
    label: "Fornix left (stylized)",
    points: [
      [-28, -18, -10],
      [-12, -12, -2],
      [0, -8, 4],
    ],
  },
  {
    id: "fornix_r",
    label: "Fornix right (stylized)",
    points: [
      [28, -18, -10],
      [12, -12, -2],
      [0, -8, 4],
    ],
  },
];

export const REGION_TARGETS: Record<string, Vec3> = {
  frontal_left: [-28, 42, 18],
  frontal_right: [28, 42, 18],
  motor_left: [-32, 4, 42],
  motor_right: [32, 4, 42],
  parietal_left: [-28, -28, 36],
  parietal_right: [28, -28, 36],
  temporal_left: [-52, -8, -18],
  temporal_right: [52, -8, -18],
  visual_cortex: [0, -72, 8],
  hippocampus_left: [-28, -18, -12],
  hippocampus_right: [28, -18, -12],
  thalamus: [0, -8, 4],
};

export const REGION_DOWNSTREAM: Record<string, string> = {
  frontal_left: "motor_left",
  frontal_right: "motor_right",
  motor_left: "thalamus",
  motor_right: "thalamus",
  parietal_left: "visual_cortex",
  parietal_right: "visual_cortex",
  temporal_left: "hippocampus_left",
  temporal_right: "hippocampus_right",
  hippocampus_left: "thalamus",
  hippocampus_right: "thalamus",
  thalamus: "visual_cortex",
  visual_cortex: "thalamus",
};

export function curveThrough(points: Vec3[]): THREE.CatmullRomCurve3 {
  return new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)), false, "catmullrom", 0.25);
}

export function travelCurve(from: Vec3, to: Vec3): THREE.QuadraticBezierCurve3 {
  const a = new THREE.Vector3(...from);
  const b = new THREE.Vector3(...to);
  const mid = a.clone().lerp(b, 0.5);
  mid.x *= 0.45;
  mid.z -= 14;
  mid.y -= 4;
  return new THREE.QuadraticBezierCurve3(a, mid, b);
}
