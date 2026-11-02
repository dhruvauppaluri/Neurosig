import * as THREE from "three";

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function hash3(x: number, y: number, z: number): number {
  const s = Math.sin(x * 17.13 + y * 9.71 + z * 13.27) * 43758.5453;
  return s - Math.floor(s);
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
  return y0 + (y1 - y0) * uz;
}

export type Vec3 = [number, number, number];

/**
 * Compact stylized human cerebrum in millimetres.
 * +X right, +Y anterior, +Z superior. Hemispheres kiss across a thin fissure.
 */
export function hemispherePoint(
  side: 1 | -1,
  nx: number,
  ny: number,
  nz: number,
  withFolds = true,
): Vec3 {
  const ax = Math.abs(nx);
  let x = 1.08 + ax * 49;
  let y = ny * 74;
  let z = nz * 54;

  const front = clamp((ny - 0.28) / 0.72, 0, 1);
  x *= 1 - 0.18 * front * front;
  z *= 1 - 0.16 * front;
  if (nz < 0) z *= 1 - 0.28 * front;
  z -= 6 * front * front;
  y += 4 * front * front;

  const back = clamp((-ny - 0.32) / 0.68, 0, 1);
  x *= 1 - 0.22 * back;
  z -= 12 * back;
  y -= 6 * back * back;

  const vertex = Math.exp(-((ny + 0.08) * (ny + 0.08)) / 0.38) * Math.max(0, nz);
  z += 10 * vertex;

  const medial = Math.exp(-(ax * ax) / 0.06);
  z *= 1 - 0.04 * medial;

  const sylvian =
    Math.exp(-((ny - 0.06) * (ny - 0.06)) / 0.16) *
    Math.exp(-((nz + 0.02) * (nz + 0.02)) / 0.016) *
    (0.2 + 0.8 * Math.min(1, ax / 0.38));
  z -= 8.5 * sylvian;
  x -= 2.4 * sylvian;

  const temporal =
    Math.exp(-((ny - 0.1) * (ny - 0.1)) / 0.13) *
    Math.exp(-((nz + 0.5) * (nz + 0.5)) / 0.08) *
    Math.min(1, Math.max(0, (ax - 0.18) / 0.28));
  x += 14 * temporal;
  z -= 30 * temporal;
  y += 7 * temporal;

  if (ax < 0.28 && nz < -0.05) {
    const g = (0.28 - ax) / 0.28;
    z += 10 * g * Math.max(0, -nz);
    x -= 6 * g * Math.max(0, -nz);
  }

  if (withFolds) {
    const lateral = Math.min(1, ax / 0.28);
    const dorsal = Math.max(0, nz);
    const gyri =
      2.1 * Math.sin(10.5 * ny + 3.2 * nz) * lateral * (0.45 + 0.55 * dorsal) +
      1.35 * Math.sin(16 * nz + 5 * ny) * lateral +
      0.85 * Math.sin(9 * ny * Math.PI) * temporal * 2.2 +
      0.4 * (noise(nx * 2.6, ny * 2.6, nz * 2.6) - 0.5);
    const central =
      Math.exp(-((ny - 0.12) * (ny - 0.12)) / 0.008) *
      Math.exp(-((nz - 0.38) * (nz - 0.38)) / 0.04) *
      lateral;
    const n = Math.hypot(x, y, z) || 1;
    const d = gyri - 3.2 * central;
    x += (x / n) * d + 1.6 * gyri * lateral * (x > 0 ? 1 : -1);
    y += (y / n) * d * 0.7;
    z += (z / n) * d * 0.7;
  }

  x *= side;
  const minMedial = 1.08;
  if (x * side < minMedial) x = side * minMedial;
  return [x, y, z];
}

function offsetOut(point: Vec3, amount: number): Vec3 {
  const n = Math.hypot(point[0], point[1], point[2]) || 1;
  return [
    point[0] + (point[0] / n) * amount,
    point[1] + (point[1] / n) * amount,
    point[2] + (point[2] / n) * amount,
  ];
}

/**
 * One cortical hemisphere in millimetres.
 * +X is right, +Y is anterior, +Z is superior.
 */
export function makeHemisphereGeometry(side: 1 | -1, detail = 192): THREE.BufferGeometry {
  const geometry = new THREE.SphereGeometry(1, detail, detail);
  const pos = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 1) {
    vertex.fromBufferAttribute(pos, i);
    const [x, y, z] = hemispherePoint(side, vertex.x, vertex.y, vertex.z, true);
    pos.setXYZ(i, x, y, z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

export function makeCerebellumGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.SphereGeometry(1, 96, 72);
  const pos = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 1) {
    vertex.fromBufferAttribute(pos, i);
    let x = vertex.x * 50;
    let y = vertex.y * 26;
    let z = vertex.z * 22;
    if (y > 0) y *= 0.62;
    const vermis = Math.exp(-(x * x) / 140);
    y -= 2.6 * vermis;
    z *= 1 + 0.1 * vermis;
    const folia = 0.9 * Math.sin(z * 4.8) * (1 - 0.3 * vermis);
    const n = Math.hypot(x, y, z) || 1;
    x += (x / n) * folia * 0.18;
    y += (y / n) * folia * 0.12;
    z += (z / n) * folia;
    pos.setXYZ(i, x, y - 54, z - 40);
  }
  geometry.computeVertexNormals();
  return geometry;
}

export function makeBrainstemGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.CylinderGeometry(3.4, 5.6, 40, 24, 10, false);
  const pos = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 1) {
    vertex.fromBufferAttribute(pos, i);
    const t = (vertex.y + 20) / 40;
    const pons = Math.exp(-((t - 0.58) * (t - 0.58)) / 0.035);
    const scale = 1 + 0.42 * pons;
    pos.setXYZ(i, vertex.x * scale, vertex.y, vertex.z * scale);
  }
  geometry.computeVertexNormals();
  return geometry;
}

function sampleArc(
  side: 1 | -1,
  count: number,
  nx: (t: number) => number,
  ny: (t: number) => number,
  nz: (t: number) => number,
): Vec3[] {
  const points: Vec3[] = [];
  for (let i = 0; i <= count; i += 1) {
    const t = i / count;
    points.push(offsetOut(hemispherePoint(side, nx(t), ny(t), nz(t), true), 0.85));
  }
  return points;
}

/** Faint sci-fi sulcal traces sitting on the cortical envelope. */
export function makeSulcalPolylines(side: 1 | -1): Vec3[][] {
  const lat = (t: number) => 0.62 + 0.2 * Math.sin(t * Math.PI);
  const traces: Vec3[][] = [
    sampleArc(side, 48, lat, (t) => 0.52 - 0.98 * t, (t) => 0.08 - 0.22 * Math.sin(t * Math.PI)),
    sampleArc(side, 30, (t) => 0.34 + 0.5 * t, () => 0.16, (t) => 0.16 + 0.5 * t),
    sampleArc(side, 28, (t) => 0.34 + 0.48 * t, () => -0.04, (t) => 0.2 + 0.44 * t),
    sampleArc(side, 24, (t) => 0.4 + 0.32 * t, (t) => 0.4 - 0.1 * t, (t) => 0.1 + 0.5 * t),
    sampleArc(side, 26, lat, (t) => 0.08 - 0.22 * t, (t) => -0.18 - 0.22 * t),
    sampleArc(side, 22, (t) => 0.28 + 0.42 * t, (t) => -0.55 + 0.06 * t, (t) => 0.02 + 0.3 * t),
  ];
  for (let k = 0; k < 5; k += 1) {
    const z0 = 0.08 + k * 0.12;
    traces.push(
      sampleArc(side, 36, lat, (t) => 0.48 - 0.9 * t, () => z0),
    );
  }
  return traces;
}

export function holographicMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    uniforms: {
      uTime: { value: 0 },
    },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vPos;
      void main() {
        vec4 world = modelMatrix * vec4(position, 1.0);
        vPos = position;
        vNormal = normalize(normalMatrix * normal);
        vView = cameraPosition - world.xyz;
        gl_Position = projectionMatrix * viewMatrix * world;
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vPos;
      uniform float uTime;
      void main() {
        vec3 n = normalize(vNormal);
        vec3 v = normalize(vView);
        float fresnel = pow(1.0 - max(dot(n, v), 0.0), 2.15);
        float scan = 0.035 * sin(vPos.y * 0.32 + uTime * 1.35);
        float inner = 0.14 * pow(max(dot(n, v), 0.0), 1.8);
        float fold = smoothstep(0.05, 0.38, length(dFdx(n)) + length(dFdy(n)));
        vec3 core = vec3(0.01, 0.09, 0.16);
        vec3 rim = vec3(0.55, 0.95, 1.0);
        vec3 color = mix(core, rim, fresnel * 0.98 + scan);
        color += rim * fresnel * 0.55;
        color += vec3(0.1, 0.45, 0.62) * inner;
        color += vec3(0.45, 0.92, 1.0) * fold * 0.55;
        float alpha = 0.05 + 0.68 * fresnel + 0.12 * fold;
        gl_FragColor = vec4(color, alpha);
      }
    `,
  });
}

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
