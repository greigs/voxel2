// Manifold-safe helpers to turn 2D polygon rings into closed 3D geometry: triangulated
// caps and extruded side walls. Used by the peg builder (square + crush-rib tabs). Rings
// are arrays of [u, v] with no closing duplicate; callers pass outer rings CCW and holes
// CW (earcut convention). A toWorld(u, v, depth) callback maps ring coords into world.

import earcut from "earcut";
import { add, scale } from "./mesh.js";

export function ringSignedArea(ring) {
  let a = 0;
  for (let i = 0; i < ring.length; i++) {
    const p = ring[i];
    const q = ring[(i + 1) % ring.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

// Triangulate a polygon (outer ring + optional hole rings) into 2D triangles.
export function triangulateRings(rings) {
  const flat = [];
  const holeIndices = [];
  const pts = [];
  for (let ri = 0; ri < rings.length; ri++) {
    if (ri > 0) holeIndices.push(pts.length);
    for (const [x, y] of rings[ri]) {
      flat.push(x, y);
      pts.push([x, y]);
    }
  }
  const idx = earcut(flat, holeIndices.length ? holeIndices : null, 2);
  const tris = [];
  for (let i = 0; i < idx.length; i += 3) {
    tris.push([pts[idx[i]], pts[idx[i + 1]], pts[idx[i + 2]]]);
  }
  return tris;
}

// Add a flat cap (triangulated polygon-with-holes) at a constant depth, wound to `ref`.
export function addCap(mesh, rings, toWorld, depth, ref) {
  for (const t of triangulateRings(rings)) {
    mesh.addTriRef(
      toWorld(t[0][0], t[0][1], depth),
      toWorld(t[1][0], t[1][1], depth),
      toWorld(t[2][0], t[2][1], depth),
      ref,
    );
  }
}

// Add the side walls of each ring, extruded from depth d0 to d1. Each wall quad is wound
// so its normal points out of the polygon (exterior of a CCW ring), using the in-plane
// tangents uhat/vhat to build the reference normal.
export function addWalls(mesh, rings, toWorld, d0, d1, uhat, vhat) {
  for (const ring of rings) {
    const len = ring.length;
    for (let i = 0; i < len; i++) {
      const p0 = ring[i];
      const p1 = ring[(i + 1) % len];
      const ext2d = [p1[1] - p0[1], -(p1[0] - p0[0])]; // right-hand normal -> exterior for CCW
      const ref = add(scale(uhat, ext2d[0]), scale(vhat, ext2d[1]));
      const A = toWorld(p0[0], p0[1], d0);
      const B = toWorld(p1[0], p1[1], d0);
      const Bd = toWorld(p1[0], p1[1], d1);
      const Ad = toWorld(p0[0], p0[1], d1);
      mesh.addQuad(A, B, Bd, Ad, ref);
    }
  }
}
