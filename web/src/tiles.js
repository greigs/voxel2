// Flat, mitered, per-face skin tiles. Direct port of tiles.py: one solid jigsaw tile per
// exposed voxel face, with a flat outer square, 45-degree inward-sloping side walls, and a
// central square registration peg. Single source of truth for tile geometry.

import { Mesh, add, scale } from "./mesh.js";
import { argwhere } from "./grid.js";
import { FACE_DIRECTIONS } from "./voxelOps.js";
import * as labels from "./labels.js";
import { addCap, addWalls } from "./poly3d.js";

export const DEFAULT_EMBOSS_DEPTH_MM = 0.6;
export const DEFAULT_PEG_CLEARANCE_MM = 0.1;
export const DEFAULT_PEG_DEPTH_CLEARANCE_MM = 0.2;
export const DEFAULT_TILE_CLEARANCE_MM = 0.1;

// Crush-rib defaults: thin triangular ridges on each peg face that locally interfere with
// the base hole and deform on insertion, giving a consistent snug fit across orientations.
export const DEFAULT_PEG_RIB_COUNT = 1;       // ribs per peg face (0 disables)
export const DEFAULT_PEG_RIB_HEIGHT_MM = 0.2; // how far each rib protrudes past the peg face
export const DEFAULT_PEG_RIB_WIDTH_MM = 1;    // base width of each triangular rib

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

// Peg cross-section outline (CCW) in face (u,v) coords: the square corner (pegU0,pegV0)
// of side pegSize, with `count` triangular crush-rib tabs per edge (apex offset outward
// by `height`, base `width`). Returns an array of [u,v] with no closing duplicate.
function pegOutlineRing(pegU0, pegV0, pegSize, count, height, width) {
  const pu1 = pegU0 + pegSize, pv1 = pegV0 + pegSize;
  const hw = width / 2;
  const centers = [];
  for (let k = 1; k <= count; k++) centers.push((pegSize * k) / (count + 1));

  const ring = [];
  // Bottom edge: u: pegU0 -> pu1, v = pegV0, outward -v.
  ring.push([pegU0, pegV0]);
  for (const c of centers) {
    ring.push([pegU0 + c - hw, pegV0], [pegU0 + c, pegV0 - height], [pegU0 + c + hw, pegV0]);
  }
  // Right edge: v: pegV0 -> pv1, u = pu1, outward +u.
  ring.push([pu1, pegV0]);
  for (const c of centers) {
    ring.push([pu1, pegV0 + c - hw], [pu1 + height, pegV0 + c], [pu1, pegV0 + c + hw]);
  }
  // Top edge: u: pu1 -> pegU0, v = pv1, outward +v.
  ring.push([pu1, pv1]);
  for (const c of centers) {
    ring.push([pu1 - c + hw, pv1], [pu1 - c, pv1 + height], [pu1 - c - hw, pv1]);
  }
  // Left edge: v: pv1 -> pegV0, u = pegU0, outward -u.
  ring.push([pegU0, pv1]);
  for (const c of centers) {
    ring.push([pegU0, pv1 - c + hw], [pegU0 - height, pv1 - c], [pegU0, pv1 - c - hw]);
  }
  return ring;
}

// Build one tile body mesh. Parameters in millimeters; see tiles.build_face_tile.
// When includeOuterFace is false, the flat top quad is omitted so labels.js can add a
// number pocket in its place. When `ribs` ({count, height, width}) is set, the peg gets
// triangular crush ribs; otherwise it is a plain square peg.
export function buildFaceTile(C, uhat, vhat, w, L, T, pegU0, pegV0, pegSize, pegDepth,
                             includeOuterFace = true, ribs = null) {
  const n = scale(w, -1); // outward normal

  const P00 = C;
  const P10 = add(C, scale(uhat, L));
  const P11 = add(add(C, scale(uhat, L)), scale(vhat, L));
  const P01 = add(C, scale(vhat, L));

  const Q00 = add(add(add(C, scale(uhat, T)), scale(vhat, T)), scale(w, T));
  const Q10 = add(add(add(C, scale(uhat, L - T)), scale(vhat, T)), scale(w, T));
  const Q11 = add(add(add(C, scale(uhat, L - T)), scale(vhat, L - T)), scale(w, T));
  const Q01 = add(add(add(C, scale(uhat, T)), scale(vhat, L - T)), scale(w, T));

  const m = new Mesh();
  const nu = scale(uhat, -1), nv = scale(vhat, -1);
  if (includeOuterFace) m.addQuad(P00, P10, P11, P01, n); // flat outer face
  // 45-degree beveled side walls.
  m.addQuad(P00, P01, Q01, Q00, nu);
  m.addQuad(P10, P11, Q11, Q10, uhat);
  m.addQuad(P00, P10, Q10, Q00, nv);
  m.addQuad(P01, P11, Q11, Q01, vhat);

  const useRibs = ribs && ribs.count > 0 && ribs.height > 0 && ribs.width > 0;

  if (useRibs) {
    // Ribbed peg: extrude a square+tabs cross-section so the mesh stays a closed manifold.
    const toWorld = (u, v, d) =>
      add(add(add(C, scale(uhat, u)), scale(vhat, v)), scale(w, d));
    const pegOutline = pegOutlineRing(pegU0, pegV0, pegSize, ribs.count, ribs.height, ribs.width);
    const innerSquare = [[T, T], [L - T, T], [L - T, L - T], [T, L - T]];
    // Top frame = inner square with the peg outline as a hole (peg outline reversed -> CW).
    addCap(m, [innerSquare, pegOutline.slice().reverse()], toWorld, T, w);
    // Peg side walls and bottom cap.
    addWalls(m, [pegOutline], toWorld, T, T + pegDepth, uhat, vhat);
    addCap(m, [pegOutline], toWorld, T + pegDepth, w);
    return m;
  }

  // Plain square peg.
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
    pegRibCount = DEFAULT_PEG_RIB_COUNT,
    pegRibHeightMm = DEFAULT_PEG_RIB_HEIGHT_MM,
    pegRibWidthMm = DEFAULT_PEG_RIB_WIDTH_MM,
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

  // Crush ribs, clamped so the ribbed peg outline stays inside the inner square [T, Lfit-T]
  // and the tabs do not overlap (width < pegSize/(count+1)).
  let ribs = null;
  const ribCount = Math.max(0, Math.floor(pegRibCount));
  if (ribCount > 0 && pegRibHeightMm > 0 && pegRibWidthMm > 0) {
    const maxH = Math.min(pegOffFace - T, (Lfit - T) - (pegOffFace + pegSizeFit));
    const ribHeight = Math.min(pegRibHeightMm, Math.max(0, maxH - 1e-6));
    const maxW = (pegSizeFit / (ribCount + 1)) * 0.98;
    const ribWidth = Math.min(pegRibWidthMm, Math.max(0, maxW));
    if (ribHeight > 0 && ribWidth > 0) {
      ribs = { count: ribCount, height: ribHeight, width: ribWidth };
    }
  }

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
          pegOffFace, pegOffFace, pegSizeFit, pegDepthFit, false, ribs);
        const numberMesh = labels.applyLabel(body, poly, Cfit, uhat, vhat, w, Lfit, embossDepthMm);
        tile.mesh = body;
        tile.numberMesh = numberMesh;
        // A clean tile (no number pocket, no ribs) for drawing tile-only edge outlines.
        // Cheap and never exported - viewer use only.
        tile.outlineMesh = buildFaceTile(Cfit, uhat, vhat, w, Lfit, T,
          pegOffFace, pegOffFace, pegSizeFit, pegDepthFit, true, null);
      } else {
        tile.mesh = buildFaceTile(Cfit, uhat, vhat, w, Lfit, T,
          pegOffFace, pegOffFace, pegSizeFit, pegDepthFit, true, ribs);
        // Clean outline (no ribs) for the viewer when ribs are enabled.
        if (ribs) {
          tile.outlineMesh = buildFaceTile(Cfit, uhat, vhat, w, Lfit, T,
            pegOffFace, pegOffFace, pegSizeFit, pegDepthFit, true, null);
        }
      }

      tiles.push(tile);
    }
  }
  return tiles;
}

export function dirName(axis, sign) {
  return ({ 0: "x", 1: "y", 2: "z" }[axis]) + (sign > 0 ? "p" : "n");
}
