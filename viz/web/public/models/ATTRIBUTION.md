# Anatomical mesh attribution

`brain.glb` and `brain-manifest.json` come from [Brain Project](https://github.com/itayinbarr/brainproject), commit `2929e94f521a8ddceab26bc100a98dc06b0da060`. The anatomy assets are **CC BY-SA 4.0**, © Z-Anatomy contributors and BodyParts3D / Database Center for Life Science (DBCLS). See the accompanying `LICENSE-CC-BY-SA-4.0.txt` and the [upstream attribution](https://github.com/itayinbarr/brainproject#attribution--licence).

Neurosig has not modified the source GLB geometry. At runtime it selects named anatomy parts, applies the documented display transform, and uses new materials. The source asset SHA-256 is `76a49ea4526a4880613aec7a02756bd7301b0b9d0680d7cae33e197b672c5453`.

The Draco decoder under `../draco/` is the glTF variant distributed with Three.js 0.170.0. Draco is licensed under Apache 2.0; its license text is kept beside the decoder.
