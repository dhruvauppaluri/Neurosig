import { Line, OrbitControls, Stars } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import {
  REGION_DOWNSTREAM,
  REGION_TARGETS,
  TRACTS,
  curveThrough,
  holographicMaterial,
  makeBrainstemGeometry,
  makeCerebellumGeometry,
  makeHemisphereGeometry,
  makeSulcalPolylines,
  travelCurve,
  type Vec3,
} from "./brainGeometry";
import { KIND_COLORS, type EventKind, type NeuralEvent, type RegionVolume, type Session } from "./types";

interface Pulse {
  event: NeuralEvent;
  born: number;
}

const PULSE_LIFE_S = 0.12;
const SOURCE_KINDS: EventKind[] = ["synthetic_source", "neuropixels_spike"];
const DETECT_KINDS: EventKind[] = ["detected_spike", "missed_spike"];

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
      <sphereGeometry args={[1, 16, 12]} />
      <meshBasicMaterial color={region.color} transparent opacity={0} depthWrite={false} />
    </mesh>
  );
}

function SulcalGlow() {
  const lines = useMemo(() => [...makeSulcalPolylines(-1), ...makeSulcalPolylines(1)], []);
  return (
    <group>
      {lines.map((points, index) => (
        <Line
          key={`sulcus-${index}`}
          points={points}
          color="#7ef6ff"
          lineWidth={1.05}
          transparent
          opacity={0.42}
        />
      ))}
    </group>
  );
}

function BrainMesh() {
  const left = useMemo(() => makeHemisphereGeometry(-1), []);
  const right = useMemo(() => makeHemisphereGeometry(1), []);
  const cerebellum = useMemo(() => makeCerebellumGeometry(), []);
  const stem = useMemo(() => makeBrainstemGeometry(), []);
  const hologram = useMemo(() => holographicMaterial(), []);
  const cere = useMemo(() => holographicMaterial(), []);

  useFrame(({ clock }) => {
    hologram.uniforms.uTime.value = clock.getElapsedTime();
    cere.uniforms.uTime.value = clock.getElapsedTime();
  });

  return (
    <group>
      <mesh geometry={left} material={hologram} />
      <mesh geometry={right} material={hologram} />
      <mesh geometry={cerebellum} material={cere} />
      <SulcalGlow />
      <mesh geometry={stem} position={[0, -22, -52]} rotation={[Math.PI / 2, 0, 0]}>
        <meshStandardMaterial
          color="#1a4a5c"
          emissive="#1ec8e8"
          emissiveIntensity={0.22}
          transparent
          opacity={0.58}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

function TractGuides() {
  const lines = useMemo(
    () =>
      TRACTS.map((tract) => ({
        id: tract.id,
        points: curveThrough(tract.points).getPoints(40).map((point) => point.toArray() as Vec3),
      })),
    [],
  );
  return (
    <group>
      {lines.map((line) => (
        <Line key={line.id} points={line.points} color="#67e8f9" lineWidth={0.7} transparent opacity={0.12} />
      ))}
    </group>
  );
}

function Probe({
  session,
  pulses,
}: {
  session: Session;
  pulses: Pulse[];
}) {
  const sites = session.electrodes.sites;
  const first = sites[0]?.xyz_mm;
  const last = sites[sites.length - 1]?.xyz_mm;
  const mid =
    first && last
      ? ([0.5 * (first[0] + last[0]), 0.5 * (first[1] + last[1]), 0.5 * (first[2] + last[2])] as Vec3)
      : ([0, -72, 8] as Vec3);
  const length = first && last ? Math.hypot(last[0] - first[0], last[1] - first[1], last[2] - first[2]) + 4 : 20;
  const glow = useRef<number[]>(sites.map(() => 0));
  const meshes = useRef<(THREE.Mesh | null)[]>([]);

  useFrame((_, delta) => {
    const now = performance.now() / 1000;
    const levels = glow.current;
    for (let i = 0; i < levels.length; i += 1) levels[i] *= Math.max(0, 1 - delta * 8);
    for (const pulse of pulses) {
      const channel = pulse.event.channel_id;
      if (channel == null || channel < 0 || channel >= levels.length) continue;
      if (
        pulse.event.kind !== "electrode_activity" &&
        pulse.event.kind !== "detected_spike" &&
        pulse.event.kind !== "neuropixels_spike"
      ) {
        continue;
      }
      const age = now - pulse.born;
      if (age < 0.18) levels[channel] = Math.max(levels[channel], 1 - age / 0.18);
    }
    meshes.current.forEach((mesh, index) => {
      if (!mesh) return;
      const material = mesh.material as THREE.MeshStandardMaterial;
      const level = levels[index] ?? 0;
      material.emissiveIntensity = 0.2 + 3.2 * level;
      mesh.scale.setScalar(1 + 0.45 * level);
    });
  });

  return (
    <group>
      {first && last && (
        <mesh position={mid}>
          <cylinderGeometry args={[0.22, 0.32, length, 8]} />
          <meshStandardMaterial color="#fb7185" emissive="#f43f5e" emissiveIntensity={0.45} metalness={0.5} roughness={0.28} />
        </mesh>
      )}
      {sites.map((site, index) => (
        <mesh
          key={site.channel_id}
          position={site.xyz_mm}
          ref={(node) => {
            meshes.current[index] = node;
          }}
        >
          <sphereGeometry args={[0.38, 8, 8]} />
          <meshStandardMaterial
            color={session.selected_channel === site.channel_id ? "#80ffdb" : "#fb7185"}
            emissive="#fb7185"
            emissiveIntensity={0.35}
            toneMapped={false}
          />
        </mesh>
      ))}
    </group>
  );
}

function destinationFor(event: NeuralEvent, session: Session): Vec3 {
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

function InstancedPulses({
  pulses,
  kinds,
  filter,
  geometry,
  scaleMul,
}: {
  pulses: Pulse[];
  kinds: Set<EventKind>;
  filter: EventKind[];
  geometry: THREE.BufferGeometry;
  scaleMul: number;
}) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);
  const maxCount = 1200;
  const allowed = useMemo(() => new Set(filter), [filter]);

  useFrame(() => {
    const instance = mesh.current;
    if (!instance) return;
    const now = performance.now() / 1000;
    let visible = 0;
    for (const pulse of pulses) {
      if (!kinds.has(pulse.event.kind) || !allowed.has(pulse.event.kind)) continue;
      const age = now - pulse.born;
      if (age > PULSE_LIFE_S * 2.2) continue;
      const fade = Math.exp(-age / PULSE_LIFE_S);
      dummy.position.set(...pulse.event.xyz_mm);
      dummy.scale.setScalar(scaleMul * (0.65 + 0.55 * fade));
      dummy.updateMatrix();
      instance.setMatrixAt(visible, dummy.matrix);
      color.set(KIND_COLORS[pulse.event.kind]).multiplyScalar(0.55 + 0.9 * fade);
      instance.setColorAt(visible, color);
      visible += 1;
      if (visible >= maxCount) break;
    }
    instance.count = visible;
    instance.instanceMatrix.needsUpdate = true;
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={mesh} args={[geometry, undefined, maxCount]}>
      <meshBasicMaterial transparent opacity={0.92} toneMapped={false} />
    </instancedMesh>
  );
}

function PulseField({ pulses, kinds }: { pulses: Pulse[]; kinds: Set<EventKind> }) {
  const spark = useMemo(() => new THREE.SphereGeometry(0.42, 8, 8), []);
  const detect = useMemo(() => new THREE.OctahedronGeometry(0.55, 0), []);
  return (
    <group>
      <InstancedPulses pulses={pulses} kinds={kinds} filter={SOURCE_KINDS} geometry={spark} scaleMul={1} />
      <InstancedPulses pulses={pulses} kinds={kinds} filter={DETECT_KINDS} geometry={detect} scaleMul={1.15} />
    </group>
  );
}

function TravelTraces({
  pulses,
  session,
  enabled,
}: {
  pulses: Pulse[];
  session: Session;
  enabled: boolean;
}) {
  const head = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);
  const maxHeads = 180;
  const curves = useMemo(() => {
    const map = new Map<string, THREE.QuadraticBezierCurve3>();
    for (const pulse of pulses.slice(-280)) {
      if (pulse.event.kind === "electrode_activity" || pulse.event.kind === "missed_spike") continue;
      if (map.has(pulse.event.event_id)) continue;
      map.set(pulse.event.event_id, travelCurve(pulse.event.xyz_mm, destinationFor(pulse.event, session)));
    }
    return map;
  }, [pulses, session]);

  useFrame(() => {
    const instance = head.current;
    if (!instance || !enabled) {
      if (instance) instance.count = 0;
      return;
    }
    const now = performance.now() / 1000;
    let visible = 0;
    for (const pulse of pulses) {
      if (pulse.event.kind === "missed_spike" || pulse.event.kind === "electrode_activity") continue;
      const age = now - pulse.born;
      const duration = 0.48;
      if (age < 0 || age > duration) continue;
      const curve = curves.get(pulse.event.event_id);
      if (!curve) continue;
      const t = age / duration;
      dummy.position.copy(curve.getPoint(t));
      dummy.scale.setScalar(0.38 + 0.22 * (1 - t));
      dummy.updateMatrix();
      instance.setMatrixAt(visible, dummy.matrix);
      color.set(KIND_COLORS[pulse.event.kind] ?? "#ffffff");
      instance.setColorAt(visible, color);
      visible += 1;
      if (visible >= maxHeads) break;
    }
    instance.count = visible;
    instance.instanceMatrix.needsUpdate = true;
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true;
  });

  const trailGeoms = useMemo(() => {
    const list: { id: string; points: Vec3[]; kind: EventKind }[] = [];
    for (const pulse of pulses.slice(-24)) {
      if (pulse.event.kind === "electrode_activity" || pulse.event.kind === "missed_spike") continue;
      const curve = curves.get(pulse.event.event_id);
      if (!curve) continue;
      list.push({
        id: pulse.event.event_id,
        points: curve.getPoints(16).map((point) => point.toArray() as Vec3),
        kind: pulse.event.kind,
      });
    }
    return list;
  }, [curves, pulses]);

  return (
    <group visible={enabled}>
      <instancedMesh ref={head} args={[undefined, undefined, maxHeads]}>
        <sphereGeometry args={[0.32, 6, 6]} />
        <meshBasicMaterial toneMapped={false} transparent opacity={0.9} />
      </instancedMesh>
      {trailGeoms.map((trail) => (
        <Line
          key={trail.id}
          points={trail.points}
          color={KIND_COLORS[trail.kind]}
          lineWidth={0.9}
          transparent
          opacity={0.28}
        />
      ))}
    </group>
  );
}

export function BrainScene({
  session,
  pulses,
  onSelectRegion,
  showTraces,
}: {
  session: Session;
  pulses: Pulse[];
  onSelectRegion: (id: string | null) => void;
  showTraces: boolean;
}) {
  const kinds = useMemo(() => new Set(session.visible_kinds), [session.visible_kinds]);

  return (
    <>
      <color attach="background" args={["#02040a"]} />
      <fog attach="fog" args={["#02040a", 240, 520]} />
      <ambientLight intensity={0.22} />
      <pointLight position={[-160, 90, 40]} intensity={1.35} color="#c8f7ff" />
      <pointLight position={[80, -40, -70]} intensity={0.4} color="#245a78" />
      <spotLight position={[-40, 140, -20]} intensity={0.85} angle={0.7} penumbra={0.55} color="#e7fdff" />
      <Stars radius={240} depth={60} count={1400} factor={2.4} fade speed={0.18} />
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <BrainMesh />
        <TractGuides />
        {session.regions.map((region) => (
          <RegionHit
            key={region.region_id}
            region={region}
            selected={session.selected_region === region.region_id}
            onSelect={(id) => onSelectRegion(session.selected_region === id ? null : id)}
          />
        ))}
        <Probe session={session} pulses={pulses} />
        <PulseField pulses={pulses} kinds={kinds} />
        <TravelTraces pulses={pulses} session={session} enabled={showTraces} />
      </group>
      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        minDistance={90}
        maxDistance={380}
        target={[0, -8, 8]}
      />
    </>
  );
}

export type { Pulse };
