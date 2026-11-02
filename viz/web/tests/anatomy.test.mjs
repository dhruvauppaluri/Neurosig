import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { assetToViewer, probeSpan, viewerToAsset } from "../src/anatomyTransform.ts";
import { collectAnatomyMeshes } from "../src/anatomyParts.ts";
import { destinationFor, shouldAnimateTravel } from "../src/visualSemantics.ts";
import * as THREE from "three";

const glb = readFileSync(new URL("../public/models/brain.glb", import.meta.url));
assert.equal(glb.toString("ascii", 0, 4), "glTF");
const jsonLength = glb.readUInt32LE(12);
const asset = JSON.parse(glb.toString("utf8", 20, 20 + jsonLength));

function boundsFor(name) {
  const node = asset.nodes.find((entry) => entry.name === name);
  assert.ok(node, `missing anatomical part: ${name}`);
  const accessorIndex = asset.meshes[node.mesh].primitives[0].attributes.POSITION;
  const bounds = asset.accessors[accessorIndex];
  assert.ok(bounds.min && bounds.max, `missing bounds for ${name}`);
  return [bounds.min, bounds.max];
}

test("asset contains bilateral cortex, brainstem, cerebellum, and wiring", () => {
  // The renderer reuses each primitive's geometry under one shared matrix.
  assert.ok(asset.nodes.every((node) => !["matrix", "translation", "rotation", "scale"].some((key) => key in node)));
  const categories = new Set(asset.nodes.map((node) => node.extras?.bx_cat));
  for (const category of ["cortex", "brainstem", "cerebellum", "tracts"]) {
    assert.ok(categories.has(category), `missing ${category}`);
  }
  const left = boundsFor("Superior frontal gyrus.l");
  const right = boundsFor("Superior frontal gyrus.r");
  assert.ok(assetToViewer(left[0])[0] < 0);
  assert.ok(assetToViewer(right[1])[0] > 0);
});

test("millimetre transform aligns front, rear, and the 32-contact visual probe", () => {
  const cortexFront = assetToViewer(boundsFor("Superior frontal gyrus.l")[1]);
  const opticRear = assetToViewer(boundsFor("Optic radiation.l")[0]);
  assert.ok(cortexFront[1] > 50, `frontal landmark at ${cortexFront[1]} mm`);
  assert.ok(opticRear[1] < -45, `posterior landmark at ${opticRear[1]} mm`);

  const sites = Array.from({ length: 32 }, (_, channel) => ({ xyz_mm: [4, -80 + channel * 0.5, 8] }));
  assert.equal(probeSpan(sites), 15.5);
  for (const site of sites) {
    const roundTrip = assetToViewer(viewerToAsset(site.xyz_mm));
    roundTrip.forEach((value, axis) => assert.ok(Math.abs(value - site.xyz_mm[axis]) < 1e-9));
    assert.ok(site.xyz_mm[1] >= -81 && site.xyz_mm[1] <= -64);
  }
});

test("recorded spikes stay local; only synthetic sources animate along wiring", () => {
  assert.equal(shouldAnimateTravel("synthetic_source"), true);
  assert.equal(shouldAnimateTravel("neuropixels_spike"), false);
  assert.equal(shouldAnimateTravel("detected_spike"), false);
  const site = { channel_id: 7, xyz_mm: [4, -76.5, 8] };
  const session = { electrodes: { sites: [site] } };
  const recorded = { kind: "neuropixels_spike", channel_id: 7, region_id: "visual_cortex", xyz_mm: [0, -72, 8] };
  assert.deepEqual(destinationFor(recorded, session), site.xyz_mm);
});

test("nested GLB primitives inherit their named anatomical category", () => {
  const root = new THREE.Group();
  const cortex = new THREE.Group();
  cortex.name = "Superior frontal gyrus.l";
  cortex.userData.bx_cat = "cortex";
  cortex.add(new THREE.Mesh(), new THREE.Mesh());
  const tract = new THREE.Group();
  tract.name = "Optic radiation.l";
  tract.userData.bx_cat = "tracts";
  tract.add(new THREE.Mesh());
  const bulkWhiteMatter = new THREE.Mesh();
  bulkWhiteMatter.userData.bx_cat = "white_matter";
  root.add(cortex, tract, bulkWhiteMatter);
  const parts = collectAnatomyMeshes(root);
  assert.equal(parts.cortex.length, 2);
  assert.equal(parts.tracts.length, 1);
  assert.equal(parts.deep.length, 0);
});
