// Flat, mitered, per-face skin tiles. Direct port of tiles.py: one solid jigsaw tile per
// exposed voxel face, with a flat outer square, 45-degree inward-sloping side walls, and a
// central square registration peg. Single source of truth for tile geometry.

import { Mesh, add, scale } from "./mesh.js";
import { argwhere } from "./grid.js";
import { FACE_DIRECTIONS } from "./voxelOps.js";
import * as labels from "./labels.js";

export const DEFAULT_EMBOSS_DEPTH_MM = 0.6;
export const DEFAULT_PEG_CLEARANCE_MM = 0.1;
export const DEFAULT_PEG_DEPTH_CLEARANCE_MM = 0.2;
export const DEFAULT_TILE_CLEARANCE_MM = 0.1;

// Deterministic pleasant-ish RGB [0..1] per tile face (mirrors tiles.tile_color intent).
export function tileColor(tile) {
  const [x, y, z] = tile.voxel;
  let h = ((Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791) ^
    Math.imul(tile.axis * 2 + (tile.sign > 0 ? 1 : 0), 2654435761)) >>> 0);
  // xorshift to spread bits, then derive 3 channels in [0.30, 0.90].
  const next = () => {
    h ^= h << 13; h >>>= 0;
    h ^= h >> 17;
    h ^= h << 5; h >>>= 0;
    return h / 0xffffffff;
  };
  return [0.30 + 0.60 * next(), 0.30 + 0.60 * next(), 0.30 + 0.60 * next()];
}

// Build one tile body mesh. Parameters in millimeters; see tiles.build_face_tile.
// When includeOuterFace is false, the flat top quad is omitted so labels.js can add a
// number pocket in its place.
export function buildFaceTile(C, uhat, vhat, w, L, T, pegU0, pegV0, pegSize, pegDepth,
                             includeOuterFace = true) {
  const n = scale(w, -1); // outward normal

  const P00 = C;
  const P10 = add(C, scale(uhat, L));
  const P11 = add(add(C, scale(uhat, L)), scale(vhat, L));
  const P01 = add(C, scale(vhat, L));

  const Q00 = add(add(add(C, scale(uhat, T)), scale(vhat, T)), scale(w, T));
  const Q10 = add(add(add(C, scale(uhat, L - T)), scale(vhat, T)), scale(w, T));
  const Q11 = add(add(add(C, scale(uhat, L - T)), scale(vhat, L - T)), scale(w, T));
  const Q01 = add(add(add(C, scale(uhat, T)), scale(vhat, L - T)), scale(w, T));

  const pu1 = pegU0 + pegSize;
  const pv1 = pegV0 + pegSize;
  const R00 = add(add(add(C, scale(uhat, pegU0)), scale(vhat, pegV0)), scale(w, T));
  const R10 = add(add(add(C, scale(uhat, pu1)), scale(vhat, pegV0)), scale(w, T));
  const R11 = add(add(add(C, scale(uhat, pu1)), scale(vhat, pv1)), scale(w, T));
  const R01 = add(add(add(C, scale(uhat, pegU0)), scale(vhat, pv1)), scale(w, T));

  const S00 = add(R00, scale(w, pegDepth));
  const S10 = add(R10, scale(w, pegDepth));
  const S11 = add(R11, scale(w, pegDepth));
  const S01 = add(R01, scale(w, pegDepth));

  const m = new Mesh();
  const nu = scale(uhat, -1), nv = scale(vhat, -1);
  if (includeOuterFace) m.addQuad(P00, P10, P11, P01, n); // flat outer face
  // 45-degree beveled side walls.
  m.addQuad(P00, P01, Q01, Q00, nu);
  m.addQuad(P10, P11, Q11, Q10, uhat);
  m.addQuad(P00, P10, Q10, Q00, nv);
  m.addQuad(P01, P11, Q11, Q01, vhat);
  // Inner face ring (frame between inner square and peg footprint).
  m.addQuad(Q00, Q10, R10, R00, w);
  m.addQuad(Q10, Q11, R11, R10, w);
  m.addQuad(Q11, Q01, R01, R11, w);
  m.addQuad(Q01, Q00, R00, R01, w);
  // Peg walls.
  m.addQuad(R00, R10, S10, S00, nv);
  m.addQuad(R10, R11, S11, S10, uhat);
  m.addQuad(R11, R01, S01, S11, vhat);
  m.addQuad(R01, R00, S00, S01, nu);
  // Peg bottom cap.
  m.addQuad(S00, S10, S11, S01, w);
  return m;
}

/**
 * Generate one tile per exposed face of every surface voxel. Mirrors
 * tiles.generate_face_tiles (with the same clearances). Returns an array of tiles:
 * { mesh, voxel:[x,y,z], axis, sign, number, colorCode, numberMesh }.
 *
 * Requires labels.loadFont() to have completed when withLabels is true.
 */
export function generateFaceTiles(layers, opts = {}) {
  const {
    withLabels = false,
    embossDepthMm = DEFAULT_EMBOSS_DEPTH_MM,
    pegClearanceMm = DEFAULT_PEG_CLEARANCE_MM,
    pegDepthClearanceMm = DEFAULT_PEG_DEPTH_CLEARANCE_MM,
    tileClearanceMm = DEFAULT_TILE_CLEARANCE_MM,
  } = opts;

  const vox = layers.initialVoxels;
  const SF = layers.scaleFactorInt | 0;
  const S = layers.voxelSizeMm;
  if (SF <= 0 || !vox.any()) return [];

  const colorCodeOf = layers.voxelColorCode;

  const L = SF * S;
  const T = layers.tileThicknessVoxels * S;
  const pegSize = layers.pegSizeVoxels * S;
  const pegDepth = layers.pegDepthVoxels * S;

  if (!(T > 0 && T < L / 2.0)) {
    throw new Error(`tile_thickness (${T}mm) must be >0 and < L/2 (${L / 2}mm). ` +
      "Reduce erosion or it is too large for the voxel size.");
  }
  const inner = L - 2.0 * T;
  if (pegSize > inner) {
    throw new Error(`peg_size (${pegSize}mm) exceeds inner face (${inner}mm). Reduce peg size.`);
  }

  const startVox = Math.floor(SF / 2) - Math.floor(layers.pegSizeVoxels / 2);
  const pegOff = startVox * S;
  const pegOffFit = pegOff + pegClearanceMm;
  const pegSizeFit = Math.max(0.2, pegSize - 2.0 * pegClearanceMm);
  const pegDepthFit = Math.max(0.2, pegDepth - pegDepthClearanceMm);

  const tc = Math.max(0.0, tileClearanceMm);
  if (tc >= (L - 2.0 * T) / 2.0) {
    throw new Error(`tile_clearance (${tc}mm) too large for this tile size.`);
  }
  const Lfit = L - 2.0 * tc;
  const pegOffFace = pegOffFit - tc;

  const dims = vox.shape;
  const tiles = [];
  let nextNumber = 1;

  for (const [x, y, z] of argwhere(vox)) {
    const blockMin = [x * SF * S, y * SF * S, z * SF * S];
    const cc = colorCodeOf ? colorCodeOf.get(x, y, z) : 0;
    for (const [axis, sign] of FACE_DIRECTIONS) {
      const nb = [x, y, z];
      nb[axis] += sign;
      const exposed = nb[axis] < 0 || nb[axis] >= dims[axis] ||
        !vox.get(nb[0], nb[1], nb[2]);
      if (!exposed) continue;

      const ua = (axis + 1) % 3;
      const va = (axis + 2) % 3;
      const uhat = [0, 0, 0]; uhat[ua] = 1.0;
      const vhat = [0, 0, 0]; vhat[va] = 1.0;
      const n = [0, 0, 0]; n[axis] = sign;
      const w = scale(n, -1);

      const C = blockMin.slice();
      C[axis] = blockMin[axis] + (sign > 0 ? L : 0.0);

      const Cfit = add(add(C, scale(uhat, tc)), scale(vhat, tc));

      const number = nextNumber++;
      const tile = {
        voxel: [x, y, z], axis, sign, number, colorCode: cc,
        numberMesh: null, mesh: null, outlineMesh: null,
      };

      let poly = null;
      if (withLabels) {
        poly = labels.labelPolygon([number, cc], Lfit);
      }

      if (poly) {
        const body = buildFaceTile(Cfit, uhat, vhat, w, Lfit, T,
          pegOffFace, pegOffFace, pegSizeFit, pegDepthFit, false);
        const numberMesh = labels.applyLabel(body, poly, Cfit, uhat, vhat, w, Lfit, embossDepthMm);
        tile.mesh = body;
        tile.numberMesh = numberMesh;
        // A clean tile (no number pocket) for drawing tile-only edge outlines. Cheap
        // (~14 quads) and never exported - viewer use only.
        tile.outlineMesh = buildFaceTile(Cfit, uhat, vhat, w, Lfit, T,
          pegOffFace, pegOffFace, pegSizeFit, pegDepthFit, true);
      } else {
        tile.mesh = buildFaceTile(Cfit, uhat, vhat, w, Lfit, T,
          pegOffFace, pegOffFace, pegSizeFit, pegDepthFit, true);
      }

      tiles.push(tile);
    }
  }
  return tiles;
}

export function dirName(axis, sign) {
  return ({ 0: "x", 1: "y", 2: "z" }[axis]) + (sign > 0 ? "p" : "n");
}
