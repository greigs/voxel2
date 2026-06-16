// Dense 3D grid stored in a flat typed array, indexed [x, y, z] to mirror the numpy
// arrays used by the Python pipeline (shape = [nx, ny, nz]).

export class Grid {
  constructor(nx, ny, nz, ArrayType = Uint8Array, data = null) {
    this.nx = nx | 0;
    this.ny = ny | 0;
    this.nz = nz | 0;
    this.data = data || new ArrayType(this.nx * this.ny * this.nz);
  }

  get shape() {
    return [this.nx, this.ny, this.nz];
  }

  idx(x, y, z) {
    return x + this.nx * (y + this.ny * z);
  }

  get(x, y, z) {
    return this.data[x + this.nx * (y + this.ny * z)];
  }

  set(x, y, z, v) {
    this.data[x + this.nx * (y + this.ny * z)] = v;
  }

  inBounds(x, y, z) {
    return x >= 0 && y >= 0 && z >= 0 && x < this.nx && y < this.ny && z < this.nz;
  }

  any() {
    const d = this.data;
    for (let i = 0; i < d.length; i++) if (d[i]) return true;
    return false;
  }

  sum() {
    const d = this.data;
    let s = 0;
    for (let i = 0; i < d.length; i++) if (d[i]) s++;
    return s;
  }

  copy() {
    return new Grid(this.nx, this.ny, this.nz, this.data.constructor, this.data.slice());
  }

  // Iterate every set cell, calling fn(x, y, z) in x-fastest order (matches
  // numpy argwhere ordering closely enough; tile numbering only needs determinism).
  forEachSet(fn) {
    const { nx, ny, nz, data } = this;
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        const base = nx * (y + ny * z);
        for (let x = 0; x < nx; x++) {
          if (data[base + x]) fn(x, y, z);
        }
      }
    }
  }
}

// argwhere-compatible ordering: numpy iterates the LAST axis fastest, i.e. z fastest,
// then y, then x. Tile numbering in the Python code relies on np.argwhere(vox) order,
// so reproduce it exactly for matching assembly numbers.
export function argwhere(grid) {
  const { nx, ny, nz, data } = grid;
  const out = [];
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        if (data[x + nx * (y + ny * z)]) out.push([x, y, z]);
      }
    }
  }
  return out;
}
