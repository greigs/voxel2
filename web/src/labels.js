// Flat, two-color number inlays for skin tiles. Port of tile_labels.py, but with NO 3D
// boolean: instead of subtracting a pocket from the body, we rebuild the tile's flat top
// as (square - text) + a shallow text pocket, and the number fills that pocket exactly.
//
// Glyph outlines come from a bundled DejaVu Sans Bold (opentype.js), are turned into a
// filled even-odd region with polygon-clipping, triangulated with earcut, and extruded.

import opentype from "opentype.js";
import polygonClipping from "polygon-clipping";
import earcut from "earcut";
import { Mesh, add, scale, cross, dot } from "./mesh.js";

let FONT = null;

export function isReady() { return FONT !== null; }

// Parse and install a font from a raw ArrayBuffer (used by loadFont and by tests/Node).
export function parseFontBuffer(arrayBuffer) {
  FONT = opentype.parse(arrayBuffer);
  return FONT;
}

export async function loadFont(url) {
  if (FONT) return FONT;
  const buf = await fetch(url).then((r) => {
    if (!r.ok) throw new Error(`Failed to load font: ${url}`);
    return r.arrayBuffer();
  });
  return parseFontBuffer(buf);
}

// ---- glyph -> filled polygon (polygon-clipping MultiPolygon) ----------------------

function flattenCommands(commands, steps = 6) {
  // Returns array of closed rings (each a list of [x,y], y flipped to math up).
  const rings = [];
  let cur = null;
  let cx = 0, cy = 0; // current point in font coords (y down)
  const pt = (x, y) => [x, -y]; // flip to math coords (y up)

  for (const c of commands) {
    if (c.type === "M") {
      if (cur && cur.length >= 3) rings.push(cur);
      cur = [pt(c.x, c.y)];
      cx = c.x; cy = c.y;
    } else if (c.type === "L") {
      cur.push(pt(c.x, c.y));
      cx = c.x; cy = c.y;
    } else if (c.type === "Q") {
      for (let i = 1; i <= steps; i++) {
        const t = i / steps;
        const mt = 1 - t;
        const x = mt * mt * cx + 2 * mt * t * c.x1 + t * t * c.x;
        const y = mt * mt * cy + 2 * mt * t * c.y1 + t * t * c.y;
        cur.push(pt(x, y));
      }
      cx = c.x; cy = c.y;
    } else if (c.type === "C") {
      for (let i = 1; i <= steps; i++) {
        const t = i / steps;
        const mt = 1 - t;
        const x = mt * mt * mt * cx + 3 * mt * mt * t * c.x1 + 3 * mt * t * t * c.x2 + t * t * t * c.x;
        const y = mt * mt * mt * cy + 3 * mt * mt * t * c.y1 + 3 * mt * t * t * c.y2 + t * t * t * c.y;
        cur.push(pt(x, y));
      }
      cx = c.x; cy = c.y;
    } else if (c.type === "Z") {
      if (cur && cur.length >= 3) rings.push(cur);
      cur = null;
    }
  }
  if (cur && cur.length >= 3) rings.push(cur);
  return rings;
}

function closeRing(ring) {
  const r = ring.slice();
  const a = r[0], b = r[r.length - 1];
  if (a[0] !== b[0] || a[1] !== b[1]) r.push([a[0], a[1]]);
  return r;
}

// Even-odd fill of a string's glyph outlines -> polygon-clipping MultiPolygon at nominal
// em size 1.0.
function glyphFill(text) {
  if (!FONT) return null;
  const path = FONT.getPath(text, 0, 0, 1.0);
  const rings = flattenCommands(path.commands);
  if (!rings.length) return null;
  const polys = rings.map((r) => [closeRing(r)]); // each ring as its own Polygon
  let result = polys[0];
  for (let i = 1; i < polys.length; i++) {
    result = polygonClipping.xor(result, polys[i]);
  }
  // Ensure MultiPolygon form.
  return polygonClipping.union(result);
}

function boundsMP(mp) {
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const poly of mp) for (const ring of poly) for (const [x, y] of ring) {
    if (x < minx) minx = x; if (x > maxx) maxx = x;
    if (y < miny) miny = y; if (y > maxy) maxy = y;
  }
  return { minx, miny, maxx, maxy };
}

function transformMP(mp, s, tx, ty) {
  return mp.map((poly) => poly.map((ring) => ring.map(([x, y]) => [x * s + tx, y * s + ty])));
}

// Scale uniformly to fit (targetW,targetH) and recenter at (cx,cy).
function fitMP(mp, targetW, targetH, cx, cy) {
  const b = boundsMP(mp);
  const w = b.maxx - b.minx, h = b.maxy - b.miny;
  if (w <= 0 || h <= 0) return mp;
  const s = Math.min(targetW / w, targetH / h);
  // Scale about origin, then translate so its center lands on (cx,cy).
  const scaled = transformMP(mp, s, 0, 0);
  const sb = boundsMP(scaled);
  return transformMP(scaled, 1, cx - (sb.minx + sb.maxx) / 2, cy - (sb.miny + sb.maxy) / 2);
}

/**
 * Build a centered, multi-line text fill polygon (in centered face u,v millimeters) for
 * the given lines. Returns a polygon-clipping MultiPolygon or null. Mirrors
 * tile_labels.label_polygon.
 */
export function labelPolygon(lines, faceSize, marginFrac = 0.16) {
  if (!FONT) return null;
  const strs = lines.map((s) => String(s)).filter((s) => s !== "");
  if (!strs.length) return null;

  const avail = faceSize * (1.0 - 2.0 * marginFrac);
  const n = strs.length;
  const lineH = (avail / n) * 0.78;
  const gap = (avail / n) * 0.22;
  const totalH = n * lineH + (n - 1) * gap;
  const yTop = totalH / 2.0 - lineH / 2.0;

  const parts = [];
  for (let i = 0; i < strs.length; i++) {
    const g = glyphFill(strs[i]);
    if (!g || !g.length) continue;
    const cy = yTop - i * (lineH + gap);
    parts.push(fitMP(g, avail, lineH, 0.0, cy));
  }
  if (!parts.length) return null;
  let u = parts[0];
  for (let i = 1; i < parts.length; i++) u = polygonClipping.union(u, parts[i]);

  // Simplify glyph outlines. Curve flattening produces far more points than a few-mm
  // printed digit needs; without this, each tile balloons to thousands of triangles
  // (walls + lid + floor + number, all per outline vertex). Tolerance scales with the
  // text size so small faces stay legible.
  const b = boundsMP(u);
  const maxDim = Math.max(b.maxx - b.minx, b.maxy - b.miny);
  const tol = Math.max(1e-4, maxDim * 0.02);
  u = u.map((poly) => poly.map((ring) => simplifyRing(ring, tol)).filter((r) => r.length >= 3));
  u = u.filter((poly) => poly.length && poly[0].length >= 3);

  // Nudge into general position: a tiny rotation guarantees no two independent glyph
  // edges are exactly collinear, which would otherwise make earcut emit a lid diagonal
  // that overlaps a hole boundary (a non-manifold T-junction). Sub-degree -> invisible.
  return rotateMP(u, 0.013);
}

// Up-arrow outline (centered at origin, ~[-0.5,0.5], y up) as a polygon-clipping
// MultiPolygon. Used for the back-face orientation mark.
export function arrowPolygon() {
  const ring = [
    [0.0, 0.5],   // apex
    [0.5, 0.0],   // right wing
    [0.2, 0.0],
    [0.2, -0.5],  // shaft
    [-0.2, -0.5],
    [-0.2, 0.0],
    [-0.5, 0.0],  // left wing
  ];
  return polygonClipping.union([[closeRing(ring)]]);
}

/**
 * Lay out the engraved back marking: the position number in the band below the peg and an
 * up-arrow in the band above it, sized to the (possibly asymmetric) flat bands flanking
 * the central peg. Coordinates are face-centered millimeters (origin at the face center),
 * matching applyBackLabel.
 *
 * @param number      position number (integer/string)
 * @param innerSize   inner-face square side (L - 2T), mm
 * @param pegSize     peg footprint side (fit, including ribs), mm
 * @param pegCenterV  peg center offset along v in face-centered coords, mm (peg is not
 *                    always centered when the scale/peg-size parity differ)
 */
export function backLabelPolygon(number, innerSize, pegSize, pegCenterV = 0.0, marginFrac = 0.12) {
  if (!FONT) return null;
  const hi = innerSize / 2.0;
  const pegHalf = pegSize / 2.0;
  const pegBottom = pegCenterV - pegHalf;
  const pegTop = pegCenterV + pegHalf;
  const botBandH = pegBottom - (-hi); // band below the peg
  const topBandH = hi - pegTop;       // band above the peg
  if (botBandH <= 0.6 && topBandH <= 0.6) return null;

  const parts = [];

  // Position number, centered in the band below the peg.
  if (botBandH > 0.6) {
    const g = glyphFill(String(number));
    if (g && g.length) {
      const targetH = botBandH * (1.0 - 2.0 * marginFrac);
      const targetW = innerSize * (1.0 - 2.0 * marginFrac);
      const cy = (-hi + pegBottom) / 2.0;
      parts.push(fitMP(g, targetW, targetH, 0.0, cy));
    }
  }
  // Up-arrow, centered in the band above the peg.
  if (topBandH > 0.6) {
    const a = arrowPolygon();
    if (a && a.length) {
      const targetH = topBandH * (1.0 - 2.0 * marginFrac);
      const cy = (pegTop + hi) / 2.0;
      parts.push(fitMP(a, targetH, targetH, 0.0, cy));
    }
  }
  if (!parts.length) return null;

  let u = parts[0];
  for (let i = 1; i < parts.length; i++) u = polygonClipping.union(u, parts[i]);

  // Simplify + general position, as in labelPolygon (tolerance scaled to the band so the
  // small digits stay legible).
  const tol = Math.max(1e-4, Math.max(botBandH, topBandH) * 0.02);
  u = u.map((poly) => poly.map((ring) => simplifyRing(ring, tol)).filter((r) => r.length >= 3));
  u = u.filter((poly) => poly.length && poly[0].length >= 3);
  if (!u.length) return null;
  return rotateMP(u, 0.013);
}

// Ramer-Douglas-Peucker on an open polyline (endpoints kept).
function rdp(pts, tol) {
  if (pts.length < 3) return pts.slice();
  const a = pts[0], b = pts[pts.length - 1];
  let dx = b[0] - a[0], dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  dx /= len; dy /= len;
  let maxD = -1, idx = -1;
  for (let i = 1; i < pts.length - 1; i++) {
    const px = pts[i][0] - a[0], py = pts[i][1] - a[1];
    const d = Math.abs(px * dy - py * dx); // perpendicular distance
    if (d > maxD) { maxD = d; idx = i; }
  }
  if (maxD > tol) {
    const left = rdp(pts.slice(0, idx + 1), tol);
    const right = rdp(pts.slice(idx), tol);
    return left.slice(0, -1).concat(right);
  }
  return [a, b];
}

// Simplify a closed ring (no closing duplicate) with RDP, anchored at the two most
// distant points so concave outlines simplify cleanly.
function simplifyRing(ring, tol) {
  if (ring.length < 5) return ring;
  let k = 1, best = -1;
  for (let i = 1; i < ring.length; i++) {
    const d = Math.hypot(ring[i][0] - ring[0][0], ring[i][1] - ring[0][1]);
    if (d > best) { best = d; k = i; }
  }
  const firstHalf = ring.slice(0, k + 1);
  const secondHalf = ring.slice(k).concat([ring[0]]);
  const a = rdp(firstHalf, tol);
  const b = rdp(secondHalf, tol);
  const out = a.slice(0, -1).concat(b.slice(0, -1));
  return out.length >= 3 ? out : ring;
}

function rotateMP(mp, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return mp.map((poly) => poly.map((ring) => ring.map(([x, y]) => [x * c - y * s, x * s + y * c])));
}

// ---- geometry construction (no 3D boolean) ---------------------------------------

// Remove consecutive duplicate and collinear vertices. Collinear runs on a ring cause
// earcut to emit interior diagonals that overlap the boundary (T-junctions / non-manifold
// lids), so they must go before triangulating or wall-building.
function cleanRing(ring, eps = 1e-7) {
  let r = ring.slice();
  if (r.length > 1) {
    const a = r[0], b = r[r.length - 1];
    if (Math.abs(a[0] - b[0]) < eps && Math.abs(a[1] - b[1]) < eps) r = r.slice(0, -1);
  }
  // Drop consecutive duplicates.
  const dedup = [];
  for (const p of r) {
    const q = dedup[dedup.length - 1];
    if (!q || Math.abs(q[0] - p[0]) > eps || Math.abs(q[1] - p[1]) > eps) dedup.push(p);
  }
  if (dedup.length >= 2) {
    const f = dedup[0], l = dedup[dedup.length - 1];
    if (Math.abs(f[0] - l[0]) < eps && Math.abs(f[1] - l[1]) < eps) dedup.pop();
  }
  // Drop collinear vertices (cyclic).
  const out = [];
  const n = dedup.length;
  for (let i = 0; i < n; i++) {
    const p0 = dedup[(i - 1 + n) % n];
    const p1 = dedup[i];
    const p2 = dedup[(i + 1) % n];
    const cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
    if (Math.abs(cross) > eps) out.push(p1);
  }
  return out;
}

function signedArea(ring) {
  let a = 0;
  for (let i = 0; i < ring.length - 1; i++) {
    a += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
  }
  return a / 2;
}

// Clean each ring (no duplicate/collinear vertices), then enforce outer ring CCW and
// holes CW. Rings that collapse below a triangle are dropped.
function normalizePolygon(poly) {
  const out = [];
  poly.forEach((ring, idx) => {
    const r = cleanRing(ring);
    if (r.length < 3) return;
    const area = signedArea([...r, r[0]]);
    const wantCCW = idx === 0;
    if ((wantCCW && area < 0) || (!wantCCW && area > 0)) r.reverse();
    out.push(r);
  });
  return out;
}

function reverseRing(r) { return r.slice().reverse(); }

function triangulatePolygon(rings) {
  // rings: outer + holes, no closing dup. Returns 2D triangles [[u,v],[u,v],[u,v]].
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

/**
 * Add the number pocket to `body` and return the matching number inlay mesh.
 * On any failure, closes the body with a plain outer face and returns null.
 *
 * @param body     tile body mesh built WITHOUT its outer face
 * @param textfill polygon-clipping MultiPolygon in centered face coords
 * @param C        min-(u,v) corner of the outer face on the surface (world)
 * @param uhat,vhat,w  tile face frame (w inward)
 * @param L        face size (mm)
 * @param emboss   inlay depth (mm)
 */
export function applyLabel(body, textfill, C, uhat, vhat, w, L, emboss) {
  const half = L / 2.0;
  const n = scale(w, -1); // outward
  const toWorld = (u, v, depth) =>
    add(add(add(C, scale(uhat, u + half)), scale(vhat, v + half)), scale(w, depth));

  const closeOuterFace = () => {
    const P00 = toWorld(-half, -half, 0);
    const P10 = toWorld(half, -half, 0);
    const P11 = toWorld(half, half, 0);
    const P01 = toWorld(-half, half, 0);
    body.addQuad(P00, P10, P11, P01, n);
  };

  try {
    if (!textfill || !textfill.length) { closeOuterFace(); return null; }

    // For faces whose (uhat, vhat) basis is left-handed relative to the outward normal
    // (every negative-facing face), the glyphs would read mirrored when viewed from
    // outside. Mirror the text in u so it reads correctly; the body pocket and number are
    // both built from this same polygon, so they stay watertight and aligned.
    if (dot(cross(uhat, vhat), n) < 0) {
      textfill = textfill.map((poly) => poly.map((ring) => ring.map(([x, y]) => [-x, y])));
    }

    // Normalized text fill: each polygon = [digitOuter(CCW), counter(CW), ...].
    // Everything below is built from these exact rings so the body lid, pocket walls and
    // floor share identical edges (no 2D boolean -> no T-junctions / no 3D CSG).
    const fillN = textfill.map(normalizePolygon).filter((p) => p.length && p[0].length >= 3);
    if (!fillN.length) { closeOuterFace(); return null; }

    const square = [[-half, -half], [half, -half], [half, half], [-half, half]]; // CCW

    const number = new Mesh();

    const addTris = (mesh, tris, depth, ref) => {
      for (const t of tris) {
        mesh.addTriRef(
          toWorld(t[0][0], t[0][1], depth),
          toWorld(t[1][0], t[1][1], depth),
          toWorld(t[2][0], t[2][1], depth),
          ref,
        );
      }
    };

    const addCap = (mesh, polysN, depth, ref) => {
      for (const rings of polysN) addTris(mesh, triangulatePolygon(rings), depth, ref);
    };

    const addWalls = (mesh, polysN, d0, d1, refSign) => {
      for (const rings of polysN) {
        for (const ring of rings) {
          const len = ring.length;
          for (let i = 0; i < len; i++) {
            const p0 = ring[i];
            const p1 = ring[(i + 1) % len];
            const ext2d = [p1[1] - p0[1], -(p1[0] - p0[0])];
            let ref = add(scale(uhat, ext2d[0]), scale(vhat, ext2d[1]));
            if (refSign < 0) ref = scale(ref, -1);
            const A = toWorld(p0[0], p0[1], d0);
            const B = toWorld(p1[0], p1[1], d0);
            const Bd = toWorld(p1[0], p1[1], d1);
            const Ad = toWorld(p0[0], p0[1], d1);
            mesh.addQuad(A, B, Bd, Ad, ref);
          }
        }
      }
    };

    // Body lid = square with each digit's OUTER ring as a hole (digit strokes opened as
    // pockets), built from the same rings the walls use.
    const lidMainRings = [square, ...fillN.map((poly) => reverseRing(poly[0]))]; // holes CW
    addTris(body, triangulatePolygon(lidMainRings), 0, n);
    // Counters (inner holes of glyphs like 0/6/8) are solid islands in the lid.
    for (const poly of fillN) {
      for (let h = 1; h < poly.length; h++) {
        addTris(body, triangulatePolygon([reverseRing(poly[h])]), 0, n); // island CCW
      }
    }
    // Body pocket floor (digit annulus) at depth emboss, facing the opening, plus the
    // pocket side walls facing into the cavity.
    addCap(body, fillN, emboss, n);
    addWalls(body, fillN, 0, emboss, -1);

    // Number inlay: flush top (digit annulus) at depth 0, bottom at depth emboss, and
    // outward side walls. A separate watertight solid that fills the pocket exactly.
    addCap(number, fillN, 0, n);
    addCap(number, fillN, emboss, w);
    addWalls(number, fillN, 0, emboss, +1);

    if (number.triCount === 0) { return null; }
    return number;
  } catch (e) {
    // Never let a bad glyph abort a whole run.
    if (body.triCount === 0) closeOuterFace();
    return null;
  }
}

/**
 * Engrave the back (inner) face: build the inner-face annulus (inner square Q minus the
 * peg footprint minus the text outlines) at depth T, plus a shallow recess (pocket floor
 * + walls) for the number/arrow toward the outer face. Single color, no inlay returned.
 * On failure, rebuilds the plain annulus so the body stays closed.
 *
 * @param body         tile body mesh (this adds the inner-face geometry to it)
 * @param backfill     polygon-clipping MultiPolygon in face-centered coords (or null)
 * @param C            min-(u,v) outer-face corner (world)
 * @param uhat,vhat,w  tile face frame (w inward, so the back face normal is w)
 * @param L            face size (mm)
 * @param T            tile thickness (mm) - the inner face sits at depth T
 * @param depth        pocket depth (mm), must be < T
 * @param pegHoleRings array of peg footprint rings in face-centered coords
 * @returns a watertight inlay Mesh that fills the pocket flush with the back face (for the
 *          second extruder / preview), or null
 */
export function applyBackLabel(body, backfill, C, uhat, vhat, w, L, T, depth, pegHoleRings) {
  const half = L / 2.0;
  const innerHalf = half - T;
  const hi = innerHalf;
  const toWorld = (u, v, d) =>
    add(add(add(C, scale(uhat, u + half)), scale(vhat, v + half)), scale(w, d));

  const addTris = (mesh, tris, d, ref) => {
    for (const t of tris) {
      mesh.addTriRef(
        toWorld(t[0][0], t[0][1], d),
        toWorld(t[1][0], t[1][1], d),
        toWorld(t[2][0], t[2][1], d),
        ref,
      );
    }
  };
  const addCapB = (mesh, polysN, d, ref) => {
    for (const rings of polysN) addTris(mesh, triangulatePolygon(rings), d, ref);
  };
  const addWallsB = (mesh, polysN, d0, d1, refSign) => {
    for (const rings of polysN) for (const ring of rings) {
      const len = ring.length;
      for (let i = 0; i < len; i++) {
        const p0 = ring[i];
        const p1 = ring[(i + 1) % len];
        const ext2d = [p1[1] - p0[1], -(p1[0] - p0[0])];
        let ref = add(scale(uhat, ext2d[0]), scale(vhat, ext2d[1]));
        if (refSign < 0) ref = scale(ref, -1);
        mesh.addQuad(
          toWorld(p0[0], p0[1], d0), toWorld(p1[0], p1[1], d0),
          toWorld(p1[0], p1[1], d1), toWorld(p0[0], p0[1], d1), ref,
        );
      }
    }
  };
  const pegHolesCW = pegHoleRings.map((r) => {
    const area = signedArea([...r, r[0]]);
    return area > 0 ? reverseRing(r) : r.slice();
  });
  const Q = [[-hi, -hi], [hi, -hi], [hi, hi], [-hi, hi]]; // CCW
  // Fallback: Q minus peg, single earcut (one hole -> robust).
  const closeInnerRing = () => addTris(body, triangulatePolygon([Q, ...pegHolesCW]), T, w);

  try {
    if (!backfill || !backfill.length) { closeInnerRing(); return null; }

    // Back face is viewed from the +w side; mirror text in u when the basis is left-handed
    // relative to w so the digits/arrow read correctly (mirror vs the front, as needed when
    // the tile is flipped to install). The arrow stays pointing +v (mirror is in u only).
    let fill = backfill;
    if (dot(cross(uhat, vhat), w) < 0) {
      fill = fill.map((poly) => poly.map((ring) => ring.map(([x, y]) => [-x, y])));
    }
    const fillN = fill.map(normalizePolygon).filter((p) => p.length && p[0].length >= 3);
    if (!fillN.length) { closeInnerRing(); return null; }

    // Back lid = Q (perimeter kept intact so it matches the bevel walls) with the peg and
    // each text outer ring as holes; glyph counters added back as solid islands. Same
    // single-earcut structure the front lid uses (which is reliable), plus the peg hole.
    const lidRings = [Q, ...pegHolesCW, ...fillN.map((poly) => reverseRing(poly[0]))];
    addTris(body, triangulatePolygon(lidRings), T, w);
    for (const poly of fillN) {
      for (let h = 1; h < poly.length; h++) {
        addTris(body, triangulatePolygon([reverseRing(poly[h])]), T, w); // counter CCW
      }
    }

    // Body pocket: floor toward the outer face + walls facing into the cavity.
    const floorD = T - depth;
    addCapB(body, fillN, floorD, w);
    addWallsB(body, fillN, T, floorD, -1);

    // Flush two-color inlay solid that fills the pocket exactly (top flush with the back
    // face, bottom at the floor, outward side walls) - same construction as the front
    // number. Exported on the number extruder and shown in the viewer.
    const inlay = new Mesh();
    addCapB(inlay, fillN, T, w);                 // flush top (back surface), faces +w
    addCapB(inlay, fillN, floorD, scale(w, -1)); // bottom faces -w
    addWallsB(inlay, fillN, T, floorD, 1);       // outward side walls
    return inlay.triCount > 0 ? inlay : null;
  } catch (e) {
    closeInnerRing();
    return null;
  }
}
