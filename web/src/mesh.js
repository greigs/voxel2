// Minimal triangle-soup mesh used across the pipeline. Stores non-indexed positions
// (9 floats per triangle). Quads are wound to match a reference normal, mirroring
// tiles._add_quad in the Python.

export function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
export function norm(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

// Rotation matrix (row-major 9) that rotates unit vector `from` onto unit vector `to`.
export function alignVectors(from, to) {
  const f = norm(from), t = norm(to);
  const v = cross(f, t);
  const c = dot(f, t);
  if (c > 1 - 1e-9) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  if (c < -1 + 1e-9) {
    // 180 deg: rotate about any axis perpendicular to f.
    let axis = Math.abs(f[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    axis = norm(cross(f, axis));
    const [x, y, z] = axis;
    return [
      2 * x * x - 1, 2 * x * y, 2 * x * z,
      2 * x * y, 2 * y * y - 1, 2 * y * z,
      2 * x * z, 2 * y * z, 2 * z * z - 1,
    ];
  }
  const [vx, vy, vz] = v;
  const k = 1 / (1 + c);
  // R = I + [v]x + [v]x^2 * k
  return [
    1 + k * (-vy * vy - vz * vz), -vz + k * (vx * vy), vy + k * (vx * vz),
    vz + k * (vx * vy), 1 + k * (-vx * vx - vz * vz), -vx + k * (vy * vz),
    -vy + k * (vx * vz), vx + k * (vy * vz), 1 + k * (-vx * vx - vy * vy),
  ];
}

export class Mesh {
  constructor() {
    this.positions = []; // flat, length multiple of 9
  }

  get triCount() { return this.positions.length / 9; }

  addTri(a, b, c) {
    this.positions.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  }

  // Add a triangle wound so its normal agrees with refNormal.
  addTriRef(a, b, c, ref) {
    const nrm = cross(sub(b, a), sub(c, a));
    if (dot(nrm, ref) < 0) { const tmp = b; b = c; c = tmp; }
    this.addTri(a, b, c);
  }

  // Add quad a-b-c-d wound to agree with refNormal (port of tiles._add_quad).
  addQuad(a, b, c, d, ref) {
    const nrm = cross(sub(b, a), sub(c, a));
    if (dot(nrm, ref) < 0) {
      const na = d, nb = c, nc = b, nd = a;
      a = na; b = nb; c = nc; d = nd;
    }
    this.addTri(a, b, c);
    this.addTri(a, c, d);
  }

  append(other) {
    for (let i = 0; i < other.positions.length; i++) this.positions.push(other.positions[i]);
  }

  clone() {
    const m = new Mesh();
    m.positions = this.positions.slice();
    return m;
  }

  translate(t) {
    const p = this.positions;
    for (let i = 0; i < p.length; i += 3) {
      p[i] += t[0]; p[i + 1] += t[1]; p[i + 2] += t[2];
    }
    return this;
  }

  applyRotation(R) {
    const p = this.positions;
    for (let i = 0; i < p.length; i += 3) {
      const x = p[i], y = p[i + 1], z = p[i + 2];
      p[i] = R[0] * x + R[1] * y + R[2] * z;
      p[i + 1] = R[3] * x + R[4] * y + R[5] * z;
      p[i + 2] = R[6] * x + R[7] * y + R[8] * z;
    }
    return this;
  }

  bounds() {
    const p = this.positions;
    if (p.length === 0) return { min: [0, 0, 0], max: [0, 0, 0] };
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.length; i += 3) {
      for (let k = 0; k < 3; k++) {
        const v = p[i + k];
        if (v < min[k]) min[k] = v;
        if (v > max[k]) max[k] = v;
      }
    }
    return { min, max };
  }

  extents() {
    const { min, max } = this.bounds();
    return [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  }

  // Deduplicated vertices + integer faces, for indexed formats (3MF).
  toIndexed(precision = 1e-4) {
    const map = new Map();
    const vertices = [];
    const faces = [];
    const p = this.positions;
    const key = (x, y, z) =>
      `${Math.round(x / precision)},${Math.round(y / precision)},${Math.round(z / precision)}`;
    const idxOf = (x, y, z) => {
      const k = key(x, y, z);
      let id = map.get(k);
      if (id === undefined) {
        id = vertices.length;
        vertices.push([x, y, z]);
        map.set(k, id);
      }
      return id;
    };
    for (let i = 0; i < p.length; i += 9) {
      const a = idxOf(p[i], p[i + 1], p[i + 2]);
      const b = idxOf(p[i + 3], p[i + 4], p[i + 5]);
      const c = idxOf(p[i + 6], p[i + 7], p[i + 8]);
      if (a !== b && b !== c && a !== c) faces.push([a, b, c]);
    }
    return { vertices, faces };
  }
}
