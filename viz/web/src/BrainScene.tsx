import { Line, OrbitControls, useGLTF } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { collectAnatomyMeshes } from "./anatomyParts";
import { ASSET_MM_SCALE, ASSET_Y_ORIGIN_M, VIEWER_ANTERIOR_OFFSET_MM, probeSpan } from "./anatomyTransform";
import { TRACTS, curveThrough, travelCurve, type Vec3 } from "./brainGeometry";
import { KIND_COLORS, type EventKind, type NeuralEvent, type RegionVolume, type Session } from "./types";
import { destinationFor, shouldAnimateTravel } from "./visualSemantics";

interface Pulse {
  event: NeuralEvent;
  born: number;
}

const PULSE_LIFE_S = 0.3;
const SOURCE_KINDS: EventKind[] = ["synthetic_source", "neuropixels_spike"];
const DETECT_KINDS: EventKind[] = ["detected_spike", "missed_spike"];
const LOCALIZE_KINDS: EventKind[] = ["localization_estimate"];

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

const ANATOMY_URL = "/models/brain.glb";
const ANATOMY_MATRIX = new THREE.Matrix4().set(
  -ASSET_MM_SCALE, 0, 0, 0,
  0, 0, ASSET_MM_SCALE, VIEWER_ANTERIOR_OFFSET_MM,
  0, ASSET_MM_SCALE, 0, -ASSET_MM_SCALE * ASSET_Y_ORIGIN_M,
  0, 0, 0, 1,
);

function BrainMesh({ opacity, showWiring }: { opacity: number; showWiring: boolean }) {
  const { scene } = useGLTF(ANATOMY_URL, "/draco/");
  const parts = useMemo(() => collectAnatomyMeshes(scene), [scene]);
  // Atlas structures overlap. Resolve the nearest cortical surface first so
  // the user-facing opacity does not accumulate over dozens of structures.
  const cortexDepth = useMemo(() => new THREE.MeshBasicMaterial({ transparent: true, opacity: 1, colorWrite: false, depthWrite: true, side: THREE.DoubleSide }), []);
  const front = useMemo(() => new THREE.MeshStandardMaterial({ color: "#c9a9a1", roughness: 0.88, transparent: true, opacity, depthWrite: false, depthFunc: THREE.EqualDepth, side: THREE.DoubleSide }), [opacity]);
  const deepMaterial = useMemo(() => new THREE.MeshStandardMaterial({ color: "#ad9387", roughness: 0.9, transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide }), []);
  const tractMaterial = useMemo(() => new THREE.MeshBasicMaterial({ color: "#b49b6b", transparent: true, opacity: 0.26, depthWrite: false, side: THREE.DoubleSide }), []);

  useEffect(() => () => front.dispose(), [front]);
  useEffect(() => () => cortexDepth.dispose(), [cortexDepth]);
  useEffect(() => () => { deepMaterial.dispose(); tractMaterial.dispose(); }, [deepMaterial, tractMaterial]);

  return (
    <group matrix={ANATOMY_MATRIX} matrixAutoUpdate={false}>
      {parts.deep.map((part) => <mesh key={part.uuid} geometry={part.geometry} material={deepMaterial} renderOrder={1} />)}
      {showWiring && parts.tracts.map((part) => <mesh key={part.uuid} geometry={part.geometry} material={tractMaterial} renderOrder={2} />)}
      {parts.cortex.map((part) => <mesh key={`depth-${part.uuid}`} geometry={part.geometry} material={cortexDepth} renderOrder={9} />)}
      {parts.cortex.map((part) => <mesh key={`front-${part.uuid}`} geometry={part.geometry} material={front} renderOrder={10} />)}
    </group>
  );
}

function CameraFit() {
  const { camera, size } = useThree();
  useEffect(() => {
    if (!(camera instanceof THREE.PerspectiveCamera)) return;
    const aspect = Math.max(0.25, size.width / Math.max(1, size.height));
    const halfVertical = THREE.MathUtils.degToRad(camera.fov) / 2;
    const distance = 1.28 * Math.max(95 / Math.tan(halfVertical), 65 / (Math.tan(halfVertical) * aspect));
    camera.position.set(-0.68 * distance, 0.28 * distance, 0.68 * distance);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [camera, size.width, size.height]);
  return null;
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
        <Line key={line.id} points={line.points} color="#d7b986" lineWidth={0.9} transparent opacity={0.4} />
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
  const length = probeSpan(sites) + 4;
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
        <mesh position={mid} renderOrder={20}>
          <cylinderGeometry args={[0.48, 0.56, length, 8]} />
          <meshStandardMaterial color="#e4e9ec" emissive="#f72585" emissiveIntensity={0.4} metalness={0.65} roughness={0.3} transparent opacity={0.9} depthTest={false} depthWrite={false} toneMapped={false} />
        </mesh>
      )}
      {sites.map((site, index) => (
        <mesh
          key={site.channel_id}
          position={site.xyz_mm}
          renderOrder={21}
          ref={(node) => {
            meshes.current[index] = node;
          }}
        >
          <sphereGeometry args={[0.48, 10, 10]} />
          <meshStandardMaterial
            color={session.selected_channel === site.channel_id ? "#80ffdb" : "#f72585"}
            emissive="#fb7185"
            emissiveIntensity={0.35}
            transparent
            depthTest={false}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      ))}
    </group>
  );
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
  const spark = useMemo(() => new THREE.SphereGeometry(1.15, 10, 10), []);
  const detect = useMemo(() => new THREE.OctahedronGeometry(1.35, 0), []);
  const localize = useMemo(() => new THREE.TetrahedronGeometry(1.55, 0), []);
  return (
    <group>
      <InstancedPulses pulses={pulses} kinds={kinds} filter={SOURCE_KINDS} geometry={spark} scaleMul={1} />
      <InstancedPulses pulses={pulses} kinds={kinds} filter={DETECT_KINDS} geometry={detect} scaleMul={1.15} />
      <InstancedPulses pulses={pulses} kinds={kinds} filter={LOCALIZE_KINDS} geometry={localize} scaleMul={1.25} />
    </group>
  );
}

function LocalizationLinks({ pulses, kinds }: { pulses: Pulse[]; kinds: Set<EventKind> }) {
  const lines = useMemo(() => {
    if (!kinds.has("localization_estimate")) return [];
    const byId = new Map(pulses.map((pulse) => [pulse.event.event_id, pulse.event]));
    const out: { id: string; a: [number, number, number]; b: [number, number, number] }[] = [];
    for (const pulse of pulses) {
      if (pulse.event.kind !== "localization_estimate") continue;
      const gtId = pulse.event.ground_truth_event_id;
      if (!gtId) continue;
      const gt = byId.get(gtId);
      if (!gt) continue;
      out.push({ id: pulse.event.event_id, a: gt.xyz_mm, b: pulse.event.xyz_mm });
    }
    return out;
  }, [pulses, kinds]);
  return (
    <group>
      {lines.map((line) => (
        <Line key={line.id} points={[line.a, line.b]} color="#ffffff" lineWidth={1.1} transparent opacity={0.55} />
      ))}
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
      if (!shouldAnimateTravel(pulse.event.kind)) continue;
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
      if (!shouldAnimateTravel(pulse.event.kind)) continue;
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
      if (!shouldAnimateTravel(pulse.event.kind)) continue;
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
        <sphereGeometry args={[0.95, 8, 8]} />
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
  cortexOpacity,
}: {
  session: Session;
  pulses: Pulse[];
  onSelectRegion: (id: string | null) => void;
  showTraces: boolean;
  cortexOpacity: number;
}) {
  const kinds = useMemo(() => new Set(session.visible_kinds), [session.visible_kinds]);

  return (
    <>
      <color attach="background" args={["#090d14"]} />
      <ambientLight intensity={0.7} />
      <hemisphereLight args={["#f4dfd4", "#565267", 1.1]} />
      <directionalLight position={[-140, 180, 130]} intensity={2.2} color="#fff0e4" />
      <directionalLight position={[90, 40, -100]} intensity={0.8} color="#bfd6e8" />
      <CameraFit />
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <Suspense fallback={null}><BrainMesh opacity={cortexOpacity} showWiring={showTraces} /></Suspense>
        {showTraces && <TractGuides />}
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
        <LocalizationLinks pulses={pulses} kinds={kinds} />
        <TravelTraces pulses={pulses} session={session} enabled={showTraces && kinds.has("synthetic_source")} />
      </group>
      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        minDistance={120}
        maxDistance={1200}
        target={[0, 0, 0]}
      />
    </>
  );
}

export type { Pulse };
