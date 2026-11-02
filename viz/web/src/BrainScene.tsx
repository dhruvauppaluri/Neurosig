import { Line, OrbitControls, Stars } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import {
  REGION_DOWNSTREAM,
  REGION_TARGETS,
  TRACTS,
  cortexMaterial,
  curveThrough,
  makeBrainstemGeometry,
  makeCerebellumGeometry,
  makeHemisphereGeometry,
  travelCurve,
  type Vec3,
} from "./brainGeometry";
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
      <sphereGeometry args={[1, 20, 14]} />
      <meshBasicMaterial
        color={region.color}
        transparent
        opacity={selected ? 0.18 : 0.0}
        depthWrite={false}
      />
    </mesh>
  );
}

function BrainMesh() {
  const left = useMemo(() => makeHemisphereGeometry(-1), []);
  const right = useMemo(() => makeHemisphereGeometry(1), []);
  const cerebellum = useMemo(() => makeCerebellumGeometry(), []);
  const stem = useMemo(() => makeBrainstemGeometry(), []);
  const outer = useMemo(() => cortexMaterial(0.5), []);
  const cere = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: "#5b8fb0",
        emissive: "#143044",
        emissiveIntensity: 0.3,
        roughness: 0.5,
        transparent: true,
        opacity: 0.55,
        depthWrite: false,
      }),
    [],
  );

  return (
    <group>
      <mesh geometry={left} material={outer} />
      <mesh geometry={right} material={outer} />
      <mesh geometry={cerebellum} material={cere} />
      <mesh geometry={stem} position={[0, -32, -28]} rotation={[0.55, 0, 0]}>
        <meshStandardMaterial color="#2a4a5c" emissive="#0e2430" emissiveIntensity={0.35} />
      </mesh>
    </group>
  );
}

function TractGuides() {
  const lines = useMemo(
    () =>
      TRACTS.map((tract) => ({
        id: tract.id,
        points: curveThrough(tract.points).getPoints(48).map((point) => point.toArray() as Vec3),
      })),
    [],
  );
  return (
    <group>
      {lines.map((line) => (
        <Line key={line.id} points={line.points} color="#4cc9f0" lineWidth={1.2} transparent opacity={0.18} />
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
  const mid = first && last ? ([0.5 * (first[0] + last[0]), 0.5 * (first[1] + last[1]), 0.5 * (first[2] + last[2])] as Vec3) : ([0, -72, 8] as Vec3);
  const length = first && last ? Math.hypot(last[0] - first[0], last[1] - first[1], last[2] - first[2]) + 6 : 20;
  const glow = useRef<number[]>(sites.map(() => 0));
  const meshes = useRef<(THREE.Mesh | null)[]>([]);

  useFrame((_, delta) => {
    const now = performance.now() / 1000;
    const levels = glow.current;
    for (let i = 0; i < levels.length; i += 1) levels[i] *= Math.max(0, 1 - delta * 6);
    for (const pulse of pulses) {
      const channel = pulse.event.channel_id;
      if (channel == null || channel < 0 || channel >= levels.length) continue;
      if (pulse.event.kind !== "electrode_activity" && pulse.event.kind !== "detected_spike" && pulse.event.kind !== "neuropixels_spike") {
        continue;
      }
      const age = now - pulse.born;
      if (age < 0.28) levels[channel] = Math.max(levels[channel], 1 - age / 0.28);
    }
    meshes.current.forEach((mesh, index) => {
      if (!mesh) return;
      const material = mesh.material as THREE.MeshStandardMaterial;
      const level = levels[index] ?? 0;
      material.emissiveIntensity = 0.25 + 4.5 * level;
      mesh.scale.setScalar(1 + 1.8 * level);
    });
  });

  return (
    <group>
      {first && last && (
        <mesh position={mid}>
          <cylinderGeometry args={[0.38, 0.48, length, 10]} />
          <meshStandardMaterial color="#f72585" emissive="#f72585" emissiveIntensity={0.35} metalness={0.4} roughness={0.3} />
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
          <sphereGeometry args={[0.9, 12, 12]} />
          <meshStandardMaterial
            color={session.selected_channel === site.channel_id ? "#80ffdb" : "#ff4d6d"}
            emissive="#ff4d6d"
            emissiveIntensity={0.4}
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

function PulseField({ pulses, kinds }: { pulses: Pulse[]; kinds: Set<EventKind> }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);
  const maxCount = 1600;

  useFrame(() => {
    const instance = mesh.current;
    if (!instance) return;
    const now = performance.now() / 1000;
    let visible = 0;
    for (const pulse of pulses) {
      if (!kinds.has(pulse.event.kind)) continue;
      const age = now - pulse.born;
      const life = 0.22;
      if (age > life * 3) continue;
      const fade = Math.exp(-age / life);
      dummy.position.set(...pulse.event.xyz_mm);
      dummy.scale.setScalar((1.4 + 2.2 * fade) * (pulse.event.kind === "detected_spike" ? 1.35 : 1));
      dummy.updateMatrix();
      instance.setMatrixAt(visible, dummy.matrix);
      color.set(KIND_COLORS[pulse.event.kind]).multiplyScalar(0.45 + fade);
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
      <meshBasicMaterial transparent opacity={0.95} toneMapped={false} />
    </instancedMesh>
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
  const maxHeads = 240;
  const curves = useMemo(() => {
    const map = new Map<string, THREE.QuadraticBezierCurve3>();
    for (const pulse of pulses.slice(-400)) {
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
      if (pulse.event.kind === "missed_spike") continue;
      const age = now - pulse.born;
      const duration = 0.55;
      if (age < 0 || age > duration) continue;
      const curve = curves.get(pulse.event.event_id);
      if (!curve) continue;
      const t = age / duration;
      const point = curve.getPoint(t);
      dummy.position.copy(point);
      dummy.scale.setScalar(1.15 + 1.4 * (1 - t));
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
    for (const pulse of pulses.slice(-36)) {
      const curve = curves.get(pulse.event.event_id);
      if (!curve) continue;
      list.push({
        id: pulse.event.event_id,
        points: curve.getPoints(18).map((point) => point.toArray() as Vec3),
        kind: pulse.event.kind,
      });
    }
    return list;
  }, [curves, pulses]);

  return (
    <group visible={enabled}>
      <instancedMesh ref={head} args={[undefined, undefined, maxHeads]}>
        <sphereGeometry args={[0.85, 8, 8]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      {trailGeoms.map((trail) => (
        <Line
          key={trail.id}
          points={trail.points}
          color={KIND_COLORS[trail.kind]}
          lineWidth={1.6}
          transparent
          opacity={0.45}
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
      <color attach="background" args={["#03050a"]} />
      <fog attach="fog" args={["#03050a", 220, 480]} />
      <ambientLight intensity={0.42} />
      <pointLight position={[90, 130, 80]} intensity={1.55} color="#b7f0ff" />
      <pointLight position={[-110, -20, -40]} intensity={0.7} color="#5e60ce" />
      <spotLight position={[0, 160, 40]} intensity={1.1} angle={0.7} penumbra={0.5} color="#dff6ff" />
      <directionalLight position={[-80, 40, 120]} intensity={0.85} color="#9ad7ff" />
      <directionalLight position={[90, -10, -80]} intensity={0.35} color="#4cc9f0" />
      <Stars radius={220} depth={50} count={900} factor={2.6} fade speed={0.25} />
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
      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        minDistance={80}
        maxDistance={340}
        target={[0, -8, 0]}
      />
    </>
  );
}

export type { Pulse };
