import * as THREE from "three";

function hash3(x: number, y: number, z: number): number {
  return Math.sin(x * 17.13 + y * 9.71 + z * 13.27) * 43758.5453;
}

function noise(x: number, y: number, z: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fy = y - iy;
  const fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const uz = fz * fz * (3 - 2 * fz);
  const n000 = hash3(ix, iy, iz);
  const n100 = hash3(ix + 1, iy, iz);
  const n010 = hash3(ix, iy + 1, iz);
  const n110 = hash3(ix + 1, iy + 1, iz);
  const n001 = hash3(ix, iy, iz + 1);
  const n101 = hash3(ix + 1, iy, iz + 1);
  const n011 = hash3(ix, iy + 1, iz + 1);
  const n111 = hash3(ix + 1, iy + 1, iz + 1);
  const x00 = n000 + (n100 - n000) * ux;
  const x10 = n010 + (n110 - n010) * ux;
  const x01 = n001 + (n101 - n001) * ux;
  const x11 = n011 + (n111 - n011) * ux;
  const y0 = x00 + (x10 - x00) * uy;
  const y1 = x01 + (x11 - x01) * uy;
  return (y0 + (y1 - y0) * uz) % 1;
}

function fbm(x: number, y: number, z: number): number {
  return (
    0.55 * noise(x * 0.07, y * 0.07, z * 0.07) +
    0.28 * noise(x * 0.16, y * 0.16, z * 0.16) +
    0.17 * noise(x * 0.33, y * 0.31, z * 0.29)
  );
}

/** Build one cortical hemisphere. +1 is right, -1 is left. Coordinates in mm. */
export function makeHemisphereGeometry(side: 1 | -1, radius = 54, detail = 112): THREE.BufferGeometry {
  const geometry = new THREE.SphereGeometry(radius, detail, detail);
  const pos = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 1) {
    vertex.fromBufferAttribute(pos, i);
    let { x, y, z } = vertex;

    x *= 0.88;
    y *= 1.36;
    z *= 1.1;

    // Flatten the medial face so two lobes read as hemispheres, not one blob.
    const medial = Math.max(0, 20 - side * x);
    x += side * medial * 0.95;

    // Temporal lobe hangs inferior-lateral.
    const temporalGate = Math.exp(-((y + 6) ** 2) / 900) * Math.exp(-((z + 6) ** 2) / 700);
    x += side * 16 * temporalGate;
    z -= 22 * temporalGate;

    // Occipital pole
    if (y < -18) {
      y -= 6 * Math.max(0, (-y - 18) / 40);
      z *= 0.94;
    }
    // Frontal pole — keep rounded, avoid a horn.
    if (y > 32) {
      y = 32 + (y - 32) * 0.55;
      x *= 0.94;
    }

    const gyri =
      3.2 * Math.sin(y * 0.31 + side * 0.4) * Math.sin(z * 0.27 + x * 0.05) +
      2.1 * Math.sin(x * 0.38 + y * 0.22) +
      1.4 * (fbm(x, y, z) - 0.5) * 2.0;

    const sylvian = 7.5 * Math.exp(-((z + 4) ** 2) / 55) * Math.exp(-((y - 4) ** 2) / 220);
    const central = 6.2 * Math.exp(-((y - 10) ** 2) / 38) * Math.exp(-((z - 24) ** 2) / 110);
    const calcarine = 4.5 * Math.exp(-((y + 55) ** 2) / 90) * Math.exp(-(z ** 2) / 70);

    const n = new THREE.Vector3(x, y, z).normalize();
    x += n.x * (0.55 * gyri - 0.35 * sylvian);
    y += n.y * (0.55 * gyri - 0.2 * central);
    z += n.z * (0.55 * gyri) - sylvian * 0.45 - central * 0.55 - calcarine * 0.4;

    x += side * 8;
    if (side * x < 3.5) {
      x = side * 3.5;
    }
    pos.setXYZ(i, x, y, z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

export function makeCerebellumGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.SphereGeometry(22, 64, 48);
  const pos = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 1) {
    vertex.fromBufferAttribute(pos, i);
    const folia = 1.6 * Math.sin(vertex.x * 1.4) * Math.sin(vertex.z * 0.9);
    vertex.x *= 1.35;
    vertex.y *= 0.72;
    vertex.z *= 0.85;
    const split = Math.exp(-(vertex.x ** 2) / 18) * 2.2;
    vertex.y -= split * 0.2;
    vertex.addScaledVector(vertex.clone().normalize(), folia);
    vertex.y -= 78;
    vertex.z -= 16;
    pos.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

export function makeBrainstemGeometry(): THREE.CylinderGeometry {
  return new THREE.CylinderGeometry(5.5, 7.5, 36, 18, 1, false);
}

export function cortexMaterial(opacity = 0.7): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color("#7eb4d0"),
    emissive: new THREE.Color("#0c2a3c"),
    emissiveIntensity: 0.22,
    roughness: 0.62,
    metalness: 0.04,
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
}

export function innerMaterial(): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color("#07141f"),
    emissive: new THREE.Color("#0b2a3c"),
    emissiveIntensity: 0.4,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });
}

export type Vec3 = [number, number, number];

export interface Tract {
  id: string;
  points: Vec3[];
  label: string;
}

/** Stylized white-matter guides for travel animation — not measured tracts. */
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
