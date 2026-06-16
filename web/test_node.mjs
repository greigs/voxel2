// Geometry sanity test (run with: node test_node.mjs). Builds the small make_test_vox
// model in-memory, runs the pipeline, and checks every tile body (and labeled body +
// number inlay) is a closed 2-manifold (each edge shared by exactly two triangles).

import fs from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { Grid } from "./src/grid.js";
import { computeLayers, FACE_DIRECTIONS } from "./src/voxelOps.js";
import { generateFaceTiles } from "./src/tiles.js";
import * as labels from "./src/labels.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function buildTestVox() {
  // Mirror make_test_vox.py: 2x2x2 cube minus (1,1,1), plus a stick at (0,0,2),(0,0,3).
  const nx = 2, ny = 2, nz = 4;
  const solid = new Grid(nx, ny, nz, Uint8Array);
  const colorIndex = new Grid(nx, ny, nz, Uint8Array);
  for (let x = 0; x < 2; x++)
    for (let y = 0; y < 2; y++)
      for (let z = 0; z < 2; z++) {
        if (x === 1 && y === 1 && z === 1) continue;
        solid.set(x, y, z, 1);
        colorIndex.set(x, y, z, 1 + ((x + y + z) % 4));
      }
  solid.set(0, 0, 2, 1); colorIndex.set(0, 0, 2, 2);
  solid.set(0, 0, 3, 1); colorIndex.set(0, 0, 3, 2);
  const paletteByIndex = [];
  for (let i = 0; i < 256; i++) paletteByIndex.push([80, 80, 80, 255]);
  paletteByIndex[1] = [220, 60, 60, 255];
  paletteByIndex[2] = [70, 180, 70, 255];
  paletteByIndex[3] = [70, 110, 220, 255];
  paletteByIndex[4] = [235, 200, 60, 255];
  return { solid, colorIndex, paletteByIndex, size: [nx, ny, nz] };
}

function countExposedFaces(solid) {
  const dims = solid.shape;
  let count = 0;
  for (let x = 0; x < dims[0]; x++)
    for (let y = 0; y < dims[1]; y++)
      for (let z = 0; z < dims[2]; z++) {
        if (!solid.get(x, y, z)) continue;
        for (const [axis, sign] of FACE_DIRECTIONS) {
          const nb = [x, y, z]; nb[axis] += sign;
          if (nb[axis] < 0 || nb[axis] >= dims[axis] || !solid.get(nb[0], nb[1], nb[2])) count++;
        }
      }
  return count;
}

function manifoldReport(mesh) {
  const { vertices, faces } = mesh.toIndexed();
  const edges = new Map();
  for (const [a, b, c] of faces) {
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      const k = u < v ? `${u}_${v}` : `${v}_${u}`;
      edges.set(k, (edges.get(k) || 0) + 1);
    }
  }
  let bad = 0;
  const badEdges = [];
  for (const [k, cnt] of edges.entries()) if (cnt !== 2) { bad++; if (badEdges.length < 8) { const [i, j] = k.split("_").map(Number); badEdges.push({ cnt, a: vertices[i], b: vertices[j] }); } }
  const V = vertices.length, F = faces.length, E = edges.size;
  return { V, F, E, euler: V - E + F, bad, badEdges };
}

let failures = 0;
const assert = (cond, msg) => { if (!cond) { failures++; console.error("FAIL:", msg); } };

const parsed = buildTestVox();
const params = {
  scaleFactor: 10, erosionVoxels: 1, voxelSizeMm: 1.25, gapMm: 0.1,
  pegSizeVoxels: null, pegDepthVoxels: 3,
};
const layers = computeLayers(parsed, params);

const expectedFaces = countExposedFaces(parsed.solid);
console.log(`Scaled shape: ${layers.scaledSolid.shape}, peg ${layers.pegSizeVoxels}vox depth ${layers.pegDepthVoxels}vox`);

// --- unlabeled bodies must be closed manifolds ---
const tilesPlain = generateFaceTiles(layers, { withLabels: false });
assert(tilesPlain.length === expectedFaces, `tile count ${tilesPlain.length} == exposed faces ${expectedFaces}`);
let plainBad = 0;
for (const t of tilesPlain) {
  const r = manifoldReport(t.mesh);
  if (r.bad !== 0 || r.euler !== 2) { plainBad++; if (plainBad <= 3) console.error("  bad plain tile", t.number, r); }
}
assert(plainBad === 0, `all ${tilesPlain.length} plain bodies are closed manifolds (bad=${plainBad})`);

// --- labeled bodies + number inlays must be closed manifolds ---
const fontBuf = fs.readFileSync(path.join(__dirname, "public/fonts/DejaVuSans-Bold.ttf"));
labels.parseFontBuffer(fontBuf.buffer.slice(fontBuf.byteOffset, fontBuf.byteOffset + fontBuf.byteLength));

const tilesLabeled = generateFaceTiles(layers, { withLabels: true });
let labeledBodyBad = 0, numberBad = 0, numbered = 0;
for (const t of tilesLabeled) {
  const rb = manifoldReport(t.mesh);
  if (rb.bad !== 0 || rb.euler !== 2) {
    labeledBodyBad++;
    if (labeledBodyBad <= 3) {
      console.error(`  bad labeled body num=${t.number} cc=${t.colorCode} axis=${t.axis} sign=${t.sign}`, { V: rb.V, F: rb.F, E: rb.E, euler: rb.euler, bad: rb.bad });
      for (const be of rb.badEdges) console.error("    edge cnt", be.cnt, "a", be.a.map((x) => +x.toFixed(3)), "b", be.b.map((x) => +x.toFixed(3)));
    }
  }
  if (t.numberMesh) {
    numbered++;
    const rn = manifoldReport(t.numberMesh);
    if (rn.bad !== 0) { numberBad++; if (numberBad <= 3) console.error("  bad number", t.number, rn); }
  }
}
assert(numbered > 0, `at least some tiles got number inlays (${numbered}/${tilesLabeled.length})`);
assert(labeledBodyBad === 0, `all labeled bodies are closed manifolds (bad=${labeledBodyBad})`);
assert(numberBad === 0, `all number inlays are closed manifolds (bad=${numberBad})`);

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
