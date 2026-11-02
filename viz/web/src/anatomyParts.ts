import * as THREE from "three";

const VISIBLE_TRACTS = [
  "Anterior thalamic radiation", "Arcuate fasciculus", "Corticospinal tract",
  "Inferior fronto-occipital fasciculus", "Inferior longitudinal fasciculus",
  "Optic radiation", "Superior longitudinal fasciculus II", "Uncinate fasciculus",
];

/** GLTFLoader puts metadata on a parent group when a named part has multiple primitives. */
export function collectAnatomyMeshes(scene: THREE.Object3D): { cortex: THREE.Mesh[]; deep: THREE.Mesh[]; tracts: THREE.Mesh[] } {
  const cortex: THREE.Mesh[] = [];
  const deep: THREE.Mesh[] = [];
  const tracts: THREE.Mesh[] = [];
  const visit = (object: THREE.Object3D, category?: string, sourceName?: string) => {
    const partCategory = (object.userData.bx_cat as string | undefined) ?? category;
    const partName = object.userData.bx_cat ? object.name : sourceName;
    if (object instanceof THREE.Mesh) {
      if (partCategory === "cortex") cortex.push(object);
      if (partCategory === "cerebellum" || partCategory === "brainstem") deep.push(object);
      if (partCategory === "tracts" && VISIBLE_TRACTS.some((name) => partName?.startsWith(name))) tracts.push(object);
    }
    object.children.forEach((child) => visit(child, partCategory, partName));
  };
  visit(scene);
  return { cortex, deep, tracts };
}
