import { OrbitControls, Stars } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { cortexMaterial, makeCortexGeometry } from "./brainGeometry";
import { KIND_COLORS, type EventKind, type NeuralEvent, type RegionVolume, type Session } from "./types";

interface Pulse {
  event: NeuralEvent;
  born: number;
}

function RegionHit({
  region,
  selected,
  onSelect,
}: {
  region: RegionVolume;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <mesh
      position={region.center_mm}
      scale={region.radii_mm}
      onClick={(event) => {
        event.stopPropagation();
        onSelect(region.region_id);
      }}
    >
      <sphereGeometry args={[1, 24, 16]} />
      <meshBasicMaterial
        color={region.color}
        transparent
        opacity={selected ? 0.22 : 0.035}
        depthWrite={false}
      />
    </mesh>
  );
}

function Probe({ session }: { session: Session }) {
  return (
    <group>
      {session.electrodes.sites.map((site) => (
        <mesh key={site.channel_id} position={site.xyz_mm}>
          <boxGeometry args={[0.7, 0.35, 0.35]} />
          <meshStandardMaterial
            color={session.selected_channel === site.channel_id ? "#80ffdb" : "#f72585"}
            emissive="#f72585"
            emissiveIntensity={0.4}
          />
        </mesh>
      ))}
    </group>
  );
}

function PulseField({ pulses, kinds }: { pulses: Pulse[]; kinds: Set<EventKind> }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);
  const maxCount = 2000;

  useFrame(() => {
    const instance = mesh.current;
    if (!instance) return;
    const now = performance.now() / 1000;
    let visible = 0;
    for (const pulse of pulses) {
      if (!kinds.has(pulse.event.kind)) continue;
      const age = now - pulse.born;
      const life = Math.max(pulse.event.duration_ms / 1000, 0.18);
      if (age > life * 4) continue;
      const fade = Math.exp(-age / life);
      const [x, y, z] = pulse.event.xyz_mm;
      dummy.position.set(x, y, z);
      const scale = (1.2 + 2.8 * fade) * (pulse.event.kind === "detected_spike" ? 1.4 : 1);
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      instance.setMatrixAt(visible, dummy.matrix);
      color.set(KIND_COLORS[pulse.event.kind]).multiplyScalar(0.4 + fade);
      instance.setColorAt(visible, color);
      visible += 1;
      if (visible >= maxCount) break;
    }
    instance.count = visible;
    instance.instanceMatrix.needsUpdate = true;
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, maxCount]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial transparent opacity={0.9} toneMapped={false} />
    </instancedMesh>
  );
}

export function BrainScene({
  session,
  pulses,
  onSelectRegion,
}: {
  session: Session;
  pulses: Pulse[];
  onSelectRegion: (id: string | null) => void;
}) {
  const geometry = useMemo(() => makeCortexGeometry(), []);
  const material = useMemo(() => cortexMaterial.clone(), []);
  const kinds = useMemo(() => new Set(session.visible_kinds), [session.visible_kinds]);

  useFrame(({ clock }) => {
    material.uniforms.uTime.value = clock.getElapsedTime();
  });

  return (
    <>
      <color attach="background" args={["#03050a"]} />
      <fog attach="fog" args={["#03050a", 180, 420]} />
      <ambientLight intensity={0.35} />
      <pointLight position={[80, 120, 90]} intensity={1.4} color="#7fe8ff" />
      <pointLight position={[-90, -40, -70]} intensity={0.6} color="#5e60ce" />
      <Stars radius={200} depth={40} count={1200} factor={3} fade speed={0.3} />
      <mesh geometry={geometry} material={material} />
      <mesh position={[0, -78, -18]} scale={[22, 16, 18]}>
        <sphereGeometry args={[1, 32, 24]} />
        <meshPhysicalMaterial
          color="#123040"
          transparent
          opacity={0.28}
          roughness={0.25}
          metalness={0.15}
          emissive="#1a6a88"
          emissiveIntensity={0.25}
        />
      </mesh>
      {session.regions.map((region) => (
        <RegionHit
          key={region.region_id}
          region={region}
          selected={session.selected_region === region.region_id}
          onSelect={(id) => onSelectRegion(session.selected_region === id ? null : id)}
        />
      ))}
      <Probe session={session} />
      <PulseField pulses={pulses} kinds={kinds} />
      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        minDistance={90}
        maxDistance={320}
        target={[0, -10, 0]}
      />
    </>
  );
}

export type { Pulse };
