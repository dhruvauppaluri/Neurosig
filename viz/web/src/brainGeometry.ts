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

/**
 * One cortical hemisphere in millimetres.
 * +X is right, +Y is anterior, +Z is superior.
 */
export function makeHemisphereGeometry(side: 1 | -1, detail = 196): THREE.BufferGeometry {
  const geometry = new THREE.SphereGeometry(1, detail, detail);
  const pos = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 1) {
    vertex.fromBufferAttribute(pos, i);
    const nx = vertex.x;
    const ny = vertex.y;
    const nz = vertex.z;
    const theta = Math.atan2(ny, Math.max(0.05, Math.abs(nx)));
    const phi = Math.acos(clamp(nz, -1, 1));

    let x = 8 + Math.abs(nx) * 58;
    let y = ny * 84;
    let z = nz * 60;

    if (ny < -0.32) {
      const t = (-ny - 0.32) / 0.68;
      x *= 1 - 0.1 * t;
      z *= 1 - 0.06 * t;
      y -= 10 * t * t;
    }
    if (ny > 0.42) {
      const t = (ny - 0.42) / 0.58;
      x *= 1 - 0.2 * t;
      z *= 1 - 0.14 * t;
    }

    const temporal =
      Math.exp(-(((ny + 0.02) / 0.4) ** 2)) *
      Math.exp(-(((nz + 0.38) / 0.48) ** 2)) *
      Math.max(0.15, Math.abs(nx));
    x += 24 * temporal;
    z -= 30 * temporal;
    y += 2 * temporal;

    const sylvian = Math.exp(-(((nz + 0.02) / 0.16) ** 2)) * Math.exp(-(((ny - 0.04) / 0.42) ** 2));
    z -= 9 * sylvian;
    x -= 3.5 * sylvian;

    const central = Math.exp(-(((ny - 0.14) / 0.065) ** 2)) * Math.exp(-(((nz - 0.42) / 0.2) ** 2));
    const calcarine = Math.exp(-(((ny + 0.62) / 0.12) ** 2)) * Math.exp(-((nz / 0.22) ** 2));
    const intraparietal = Math.exp(-(((ny + 0.22) / 0.1) ** 2)) * Math.exp(-(((nz - 0.5) / 0.18) ** 2));

    const gyri =
      2.15 * Math.sin(15 * phi + 5 * theta) * Math.sin(10 * theta + 2 * phi) +
      1.45 * Math.sin(27 * phi + 8 * theta) +
      0.95 * Math.sin(41 * theta + 19 * phi) * Math.max(0.2, Math.sin(phi)) +
      0.7 * (noise(x * 0.09, y * 0.09, z * 0.09) - 0.5) * 2;

    const len = Math.hypot(x, y, z) || 1;
    const gx = x / len;
    const gy = y / len;
    const gz = z / len;
    const displacement = 2.05 * gyri - 7.2 * central - 4.2 * sylvian - 3.4 * calcarine - 3.6 * intraparietal;
    x += gx * displacement;
    y += gy * displacement;
    z += gz * displacement * 0.9;

    x *= side;
    if (side * x < 2.4) {
      x = side * 2.4;
    }
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
    let x = vertex.x * 28;
    let y = vertex.y * 16;
    let z = vertex.z * 18;
    const vermis = Math.exp(-(x * x) / 28);
    x *= 1 + 0.08 * (1 - vermis);
    y -= 4 * vermis;
    const folia = 1.15 * Math.sin(x * 1.65) * Math.sin(z * 1.1 + y * 0.4);
    const n = Math.hypot(x, y, z) || 1;
    x += (x / n) * folia;
    y += (y / n) * folia * 0.4;
    z += (z / n) * folia * 0.5;
    pos.setXYZ(i, x, y - 76, z - 20);
  }
  geometry.computeVertexNormals();
  return geometry;
}

export function makeBrainstemGeometry(): THREE.CylinderGeometry {
  return new THREE.CylinderGeometry(4.2, 6.4, 34, 20, 1, false);
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
        float fresnel = pow(1.0 - max(dot(n, v), 0.0), 2.35);
        float scan = 0.04 * sin(vPos.y * 0.35 + uTime * 1.4);
        vec3 core = vec3(0.03, 0.14, 0.20);
        vec3 rim = vec3(0.42, 0.90, 1.0);
        vec3 color = mix(core, rim, fresnel * 0.95 + scan);
        color += rim * fresnel * 0.35;
        float alpha = 0.075 + 0.58 * fresnel;
        gl_FragColor = vec4(color, alpha);
      }
    `,
  });
}

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
