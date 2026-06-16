// Voxel pipeline ported from process_vox.py: scale -> erode -> carve peg holes -> crop,
// plus color-code building and the perpendicular-peg depth clamp. The flat mitered tiles
// themselves are generated analytically in tiles.js from the layers produced here.

import { Grid, argwhere } from "./grid.js";

export const FACE_DIRECTIONS = [
  [0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1],
];

export function defaultPegSizeVoxels(scaleFactorInt) {
  return Math.max(1, Math.round(scaleFactorInt / 3.0));
}

// Each original voxel becomes a solid SF x SF x SF block (block-replication scaling).
export function scaleVoxels(solid, scaleFactor) {
  const sf = Math.round(scaleFactor);
  const [ox, oy, oz] = solid.shape;
  if (sf <= 0 || !solid.any()) {
    const s = Math.max(0, sf);
    return new Grid(ox * s, oy * s, oz * s, Uint8Array);
  }
  const out = new Grid(ox * sf, oy * sf, oz * sf, Uint8Array);
  for (let x = 0; x < ox; x++) {
    for (let y = 0; y < oy; y++) {
      for (let z = 0; z < oz; z++) {
        if (!solid.get(x, y, z)) continue;
        const xs = x * sf, ys = y * sf, zs = z * sf;
        for (let dz = 0; dz < sf; dz++) {
          for (let dy = 0; dy < sf; dy++) {
            for (let dx = 0; dx < sf; dx++) {
              out.set(xs + dx, ys + dy, zs + dz, 1);
            }
          }
        }
      }
    }
  }
  return out;
}

// 3D binary erosion with a full 3x3x3 (26-connectivity) structuring element and
// out-of-bounds treated as empty, repeated `iterations` times. Equivalent to scipy's
// binary_erosion(struct=generate_binary_structure(3,3), iterations=N). Implemented as
// separable 1D erosions along each axis (the cube element factorizes), which is far
// faster on large scaled grids.
export function erodeVoxels(grid, iterations) {
  let cur = grid.copy();
  for (let it = 0; it < iterations; it++) {
    cur = erodeAxis(cur, 0);
    cur = erodeAxis(cur, 1);
    cur = erodeAxis(cur, 2);
  }
  return cur;
}

function erodeAxis(grid, axis) {
  const { nx, ny, nz, data } = grid;
  const out = new Grid(nx, ny, nz, Uint8Array);
  const od = out.data;
  if (axis === 0) {
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        const base = nx * (y + ny * z);
        for (let x = 0; x < nx; x++) {
          if (!data[base + x]) continue;
          if (x === 0 || x === nx - 1) continue;
          if (data[base + x - 1] && data[base + x + 1]) od[base + x] = 1;
        }
      }
    }
  } else if (axis === 1) {
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        const base = nx * (y + ny * z);
        if (y === 0 || y === ny - 1) continue;
        const up = nx * (y - 1 + ny * z);
        const dn = nx * (y + 1 + ny * z);
        for (let x = 0; x < nx; x++) {
          if (!data[base + x]) continue;
          if (data[up + x] && data[dn + x]) od[base + x] = 1;
        }
      }
    }
  } else {
    for (let z = 0; z < nz; z++) {
      if (z === 0 || z === nz - 1) continue;
      for (let y = 0; y < ny; y++) {
        const base = nx * (y + ny * z);
        const up = nx * (y + ny * (z - 1));
        const dn = nx * (y + ny * (z + 1));
        for (let x = 0; x < nx; x++) {
          if (!data[base + x]) continue;
          if (data[up + x] && data[dn + x]) od[base + x] = 1;
        }
      }
    }
  }
  return out;
}

export function cropVoxelData(grid) {
  if (!grid.any()) return { cropped: new Grid(0, 0, 0), min: null };
  let xmin = Infinity, ymin = Infinity, zmin = Infinity;
  let xmax = -1, ymax = -1, zmax = -1;
  const { nx, ny, nz, data } = grid;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      const base = nx * (y + ny * z);
      for (let x = 0; x < nx; x++) {
        if (!data[base + x]) continue;
        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
        if (z < zmin) zmin = z; if (z > zmax) zmax = z;
      }
    }
  }
  const cx = xmax - xmin + 1, cy = ymax - ymin + 1, cz = zmax - zmin + 1;
  const cropped = new Grid(cx, cy, cz, Uint8Array);
  for (let z = 0; z < cz; z++) {
    for (let y = 0; y < cy; y++) {
      for (let x = 0; x < cx; x++) {
        cropped.set(x, y, z, grid.get(xmin + x, ymin + y, zmin + z));
      }
    }
  }
  return { cropped, min: [xmin, ymin, zmin] };
}

// Carve a square peg hole into the base under each exposed voxel face (scaled grid).
// Port of process_vox.carve_peg_holes. Returns { base, holes }.
export function carvePegHoles(baseUncropped, initialVoxels, sf, tileThickness, pegSize, pegDepth) {
  const base = baseUncropped.copy();
  const holes = new Grid(base.nx, base.ny, base.nz, Uint8Array);
  const dims = initialVoxels.shape;
  const expected = [dims[0] * sf, dims[1] * sf, dims[2] * sf];
  if (sf <= 0 || !initialVoxels.any() ||
      base.nx !== expected[0] || base.ny !== expected[1] || base.nz !== expected[2]) {
    return { base, holes };
  }

  const shape = base.shape;
  const start = Math.floor(sf / 2) - Math.floor(pegSize / 2);
  const T = tileThickness;
  const d = pegDepth;

  for (const [x, y, z] of argwhere(initialVoxels)) {
    const blockMin = [x * sf, y * sf, z * sf];
    for (const [axis, sign] of FACE_DIRECTIONS) {
      const nb = [x, y, z];
      nb[axis] += sign;
      const exposed = nb[axis] < 0 || nb[axis] >= dims[axis] ||
        !initialVoxels.get(nb[0], nb[1], nb[2]);
      if (!exposed) continue;

      const ua = (axis + 1) % 3;
      const va = (axis + 2) % 3;
      const lo = [0, 0, 0];
      const hi = [shape[0], shape[1], shape[2]];
      lo[ua] = blockMin[ua] + start;
      hi[ua] = lo[ua] + pegSize;
      lo[va] = blockMin[va] + start;
      hi[va] = lo[va] + pegSize;
      if (sign > 0) {
        const top = blockMin[axis] + sf;
        hi[axis] = top - T;
        lo[axis] = hi[axis] - d;
      } else {
        const bot = blockMin[axis];
        lo[axis] = bot + T;
        hi[axis] = lo[axis] + d;
      }

      for (let i = 0; i < 3; i++) {
        lo[i] = Math.max(0, Math.trunc(lo[i]));
        hi[i] = Math.min(Math.trunc(shape[i]), Math.trunc(hi[i]));
      }
      if (lo[0] < hi[0] && lo[1] < hi[1] && lo[2] < hi[2]) {
        for (let zz = lo[2]; zz < hi[2]; zz++) {
          for (let yy = lo[1]; yy < hi[1]; yy++) {
            for (let xx = lo[0]; xx < hi[0]; xx++) {
              base.set(xx, yy, zz, 0);
              holes.set(xx, yy, zz, 1);
            }
          }
        }
      }
    }
  }

  return { base, holes };
}

// Map the distinct palette indices used by solid voxels to compact codes 1..K.
// Port of process_vox.build_color_codes. Returns { codeGrid, legend } where legend is a
// Map of code -> source palette index.
export function buildColorCodes(colorIndex, solid) {
  const codeGrid = new Grid(colorIndex.nx, colorIndex.ny, colorIndex.nz, Uint8Array);
  if (!solid.any()) return { codeGrid, legend: new Map() };

  const used = new Set();
  const { nx, ny, nz } = solid;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        if (solid.get(x, y, z)) used.add(colorIndex.get(x, y, z));
      }
    }
  }
  const sorted = [...used].sort((a, b) => a - b);
  const idxToCode = new Map();
  const legend = new Map();
  sorted.forEach((idx, i) => {
    idxToCode.set(idx, i + 1);
    legend.set(i + 1, idx);
  });
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        if (solid.get(x, y, z)) {
          codeGrid.set(x, y, z, idxToCode.get(colorIndex.get(x, y, z)));
        }
      }
    }
  }
  return { codeGrid, legend };
}

function placeAligned(scaledShape, cropped, cropMin) {
  const [nx, ny, nz] = scaledShape;
  const aligned = new Grid(nx, ny, nz, Uint8Array);
  if (!cropped.any()) return aligned;
  if (cropMin) {
    const [xm, ym, zm] = cropMin;
    const [cx, cy, cz] = cropped.shape;
    if (xm >= 0 && ym >= 0 && zm >= 0 && xm + cx <= nx && ym + cy <= ny && zm + cz <= nz) {
      for (let z = 0; z < cz; z++) {
        for (let y = 0; y < cy; y++) {
          for (let x = 0; x < cx; x++) {
            aligned.set(xm + x, ym + y, zm + z, cropped.get(x, y, z));
          }
        }
      }
    }
  } else if (cropped.nx === nx && cropped.ny === ny && cropped.nz === nz) {
    aligned.data.set(cropped.data);
  }
  return aligned;
}

/**
 * Regenerate the expected base/solid layers + shared geometry params from a parsed .vox.
 * Mirrors process_vox.compute_expected_layers.
 */
export function computeLayers(parsed, opts) {
  const {
    scaleFactor,
    erosionVoxels,
    voxelSizeMm = 1.25,
    gapMm = 0.1,
    pegSizeVoxels = null,
    pegDepthVoxels = 3,
  } = opts;

  const initialVoxels = parsed.solid;
  const { codeGrid, legend } = buildColorCodes(parsed.colorIndex, initialVoxels);

  const scaled = scaleVoxels(initialVoxels, scaleFactor);
  const intSf = Math.round(scaleFactor);

  const tileThicknessVoxels = Math.trunc(erosionVoxels);
  let pegSize = pegSizeVoxels == null ? defaultPegSizeVoxels(intSf) : Math.trunc(pegSizeVoxels);
  let pegDepth = Math.trunc(pegDepthVoxels);

  // Clamp peg geometry so two perpendicular pegs on the SAME voxel never collide.
  if (intSf > 0) {
    let startV = Math.floor(intSf / 2) - Math.floor(pegSize / 2);
    if (startV < tileThicknessVoxels) {
      pegSize = Math.max(1, intSf - 2 * tileThicknessVoxels);
      startV = Math.floor(intSf / 2) - Math.floor(pegSize / 2);
    }
    let maxPegDepth = (intSf - startV - pegSize) - tileThicknessVoxels;
    maxPegDepth = Math.max(0, maxPegDepth);
    if (pegDepth > maxPegDepth) pegDepth = maxPegDepth;
  }

  let baseUncropped;
  if (erosionVoxels > 0 && scaled.any()) {
    baseUncropped = erodeVoxels(scaled, erosionVoxels);
  } else {
    baseUncropped = scaled.copy();
  }

  let cutoutVoxels = new Grid(scaled.nx, scaled.ny, scaled.nz, Uint8Array);
  if (baseUncropped.any() && intSf > 0) {
    const carved = carvePegHoles(baseUncropped, initialVoxels, intSf,
      tileThicknessVoxels, pegSize, pegDepth);
    baseUncropped = carved.base;
    cutoutVoxels = carved.holes;
  }

  let baseCropped, cropMin;
  if (baseUncropped.any()) {
    const c = cropVoxelData(baseUncropped);
    baseCropped = c.cropped;
    cropMin = c.min;
  } else {
    baseCropped = baseUncropped;
    cropMin = null;
  }

  const baseAligned = placeAligned(scaled.shape, baseCropped, cropMin);

  return {
    initialVoxels,
    paletteByIndex: parsed.paletteByIndex,
    scaledSolid: scaled,
    cutoutVoxels,
    baseCropped,
    cropMin,
    baseAligned,
    scaleFactorInt: intSf,
    voxelSizeMm,
    gapMm,
    tileThicknessVoxels,
    pegSizeVoxels: pegSize,
    pegDepthVoxels: pegDepth,
    voxelColorCode: codeGrid,
    colorLegend: legend,
  };
}
