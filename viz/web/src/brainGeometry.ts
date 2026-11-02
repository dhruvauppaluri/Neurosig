import * as THREE from "three";

function fbm(x: number, y: number, z: number): number {
  return (
    0.55 * Math.sin(x * 0.09 + z * 0.07) * Math.cos(y * 0.08) +
    0.28 * Math.sin(x * 0.21 + y * 0.17) +
    0.17 * Math.sin(z * 0.33 + x * 0.14)
  );
}

export function makeCortexGeometry(radius = 58, detail = 96): THREE.BufferGeometry {
  const geometry = new THREE.SphereGeometry(radius, detail, detail);
  const pos = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 1) {
    vertex.fromBufferAttribute(pos, i);
    const n = vertex.clone().normalize();
    // Dual-hemisphere cleft
    const cleft = Math.exp(-((vertex.x / 6) ** 2)) * 7.5;
    const sulcus = 3.8 * fbm(vertex.x, vertex.y, vertex.z);
    const occipital = 1.0 + 0.12 * Math.max(0, -n.y);
    const frontal = 1.0 + 0.08 * Math.max(0, n.y);
    const scaleY = 1.38;
    const scaleZ = 1.08;
    const scaleX = 1.18;
    vertex.x *= scaleX;
    vertex.y *= scaleY;
    vertex.z *= scaleZ;
    vertex.addScaledVector(n, sulcus - cleft * Math.max(0, 1 - Math.abs(n.z)));
    vertex.multiplyScalar((occipital + frontal) * 0.5);
    pos.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

export const cortexMaterial = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  uniforms: {
    uTime: { value: 0 },
    uHighlight: { value: new THREE.Color("#4cc9f0") },
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
      float fresnel = pow(1.0 - max(dot(normalize(vNormal), normalize(vView)), 0.0), 2.15);
      float pulse = 0.5 + 0.5 * sin(uTime * 0.7 + vPos.y * 0.04);
      vec3 core = vec3(0.02, 0.07, 0.12);
      vec3 glow = vec3(0.22, 0.78, 0.95);
      vec3 color = mix(core, glow, fresnel * (0.75 + 0.25 * pulse));
      float alpha = 0.16 + 0.52 * fresnel;
      gl_FragColor = vec4(color, alpha);
    }
  `,
});
