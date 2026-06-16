// Binary STL writers. Tile bodies and number inlays are gapped per voxel (matching
// process_vox.save_tiles_stl); the base is the external-face surface of a voxel grid
// (matching process_vox.save_stl_file).

import { Mesh, sub, cross } from "../mesh.js";

function triNormal(ax, ay, az, bx, by, bz, cx, cy, cz) {
  const n = cross(sub([bx, by, bz], [ax, ay, az]), sub([cx, cy, cz], [ax, ay, az]));
  const l = Math.hypot(n[0], n[1], n[2]) || 1;
  return [n[0] / l, n[1] / l, n[2] / l];
}

// Concatenate the positions of several meshes into one binary STL Blob.
export function meshesToStlBlob(meshes) {
  let triCount = 0;
  for (const m of meshes) triCount += m.triCount;

  const buf = new ArrayBuffer(84 + triCount * 50);
  const dv = new DataView(buf);
  dv.setUint32(80, triCount, true);
  let o = 84;
  for (const m of meshes) {
    const p = m.positions;
    for (let i = 0; i < p.length; i += 9) {
      const nrm = triNormal(p[i], p[i + 1], p[i + 2], p[i + 3], p[i + 4], p[i + 5], p[i + 6], p[i + 7], p[i + 8]);
      dv.setFloat32(o, nrm[0], true); dv.setFloat32(o + 4, nrm[1], true); dv.setFloat32(o + 8, nrm[2], true);
      for (let k = 0; k < 9; k++) dv.setFloat32(o + 12 + k * 4, p[i + k], true);
      dv.setUint16(o + 48, 0, true);
      o += 50;
    }
  }
  return new Blob([buf], { type: "model/stl" });
}

function placedBodies(tiles, gapMm) {
  return tiles.map((t) => {
    const m = t.mesh.clone();
    m.translate([t.voxel[0] * gapMm, t.voxel[1] * gapMm, t.voxel[2] * gapMm]);
    return m;
  });
}

export function tileBodiesStl(tiles, gapMm) {
  return meshesToStlBlob(placedBodies(tiles, gapMm));
}

export function tileNumbersStl(tiles, gapMm) {
  const placed = [];
  for (const t of tiles) {
    if (!t.numberMesh) continue;
    const m = t.numberMesh.clone();
    m.translate([t.voxel[0] * gapMm, t.voxel[1] * gapMm, t.voxel[2] * gapMm]);
    placed.push(m);
  }
  return placed.length ? meshesToStlBlob(placed) : null;
}

export function hasNumbers(tiles) {
  return tiles.some((t) => t.numberMesh);
}

// External-face surface mesh of a boolean voxel grid, each voxel a cube of side s.
export function gridSurfaceMesh(grid, s) {
  const m = new Mesh();
  const { nx, ny, nz } = grid;
  const get = (x, y, z) => x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && grid.get(x, y, z);
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        if (!grid.get(x, y, z)) continue;
        const bx = x * s, by = y * s, bz = z * s;
        const v = [
          [bx, by, bz], [bx + s, by, bz], [bx + s, by + s, bz], [bx, by + s, bz],
          [bx, by, bz + s], [bx + s, by, bz + s], [bx + s, by + s, bz + s], [bx, by + s, bz + s],
        ];
        if (!get(x - 1, y, z)) { m.addTri(v[0], v[4], v[7]); m.addTri(v[0], v[7], v[3]); }
        if (!get(x + 1, y, z)) { m.addTri(v[1], v[2], v[6]); m.addTri(v[1], v[6], v[5]); }
        if (!get(x, y - 1, z)) { m.addTri(v[0], v[1], v[5]); m.addTri(v[0], v[5], v[4]); }
        if (!get(x, y + 1, z)) { m.addTri(v[3], v[7], v[6]); m.addTri(v[3], v[6], v[2]); }
        if (!get(x, y, z - 1)) { m.addTri(v[0], v[2], v[1]); m.addTri(v[0], v[3], v[2]); }
        if (!get(x, y, z + 1)) { m.addTri(v[4], v[5], v[6]); m.addTri(v[4], v[6], v[7]); }
      }
    }
  }
  return m;
}

export function baseStl(grid, s) {
  return meshesToStlBlob([gridSurfaceMesh(grid, s)]);
}
