"""Purpose-built generator for flat, mitered, per-face skin tiles.

Each exposed face of each surface voxel becomes one solid jigsaw tile:

  * a flat full-face outer square (the visible, colorable surface),
  * side walls cut at 45 degrees sloping inward (so coplanar neighbors butt
    gaplessly at the surface and convex-edge neighbors meet as clean miters, with
    no required assembly order), and
  * a single central square peg that drops into a matching hole in the base.

All geometry is expressed in scaled-grid millimeters, matching the base STL and the
gapped layout used elsewhere. This module is the single source of truth for tile
geometry; process_vox (writing), verify_tiles (checking), and view_tiles (display)
all build the exact same tiles from it.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import trimesh


@dataclass
class TileParams:
    """Tile geometry parameters, expressed in scaled voxels."""
    tile_thickness_voxels: int = 1   # T: bevel depth / skin thickness
    peg_size_voxels: int = 3         # p: peg cross-section (square)
    peg_depth_voxels: int = 3        # d: how far the peg sticks into the base


@dataclass
class Tile:
    """One generated tile, in its true assembled position."""
    mesh: "trimesh.Trimesh"
    voxel: Tuple[int, int, int]
    axis: int   # face normal axis (0=x, 1=y, 2=z)
    sign: int   # +1 or -1
    number: int = 0                      # unique assembly id written on the face
    color_code: int = 0                  # paint color-code written on the face
    number_mesh: "trimesh.Trimesh" = None  # flush two-color inlay (None if unlabeled)


DEFAULT_EMBOSS_DEPTH_MM = 0.6  # depth of the flush two-color number inlay

# Print-tolerance defaults (mm). The base peg holes stay at their exact voxel size; the
# peg is shrunk so it actually slides in on an FDM printer.
DEFAULT_PEG_CLEARANCE_MM = 0.1        # shrink peg per side -> 2x this diametral slop
DEFAULT_PEG_DEPTH_CLEARANCE_MM = 0.2  # make peg shorter than its hole so it can't bottom out


# The 6 face directions as (axis, sign).
FACE_DIRECTIONS = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]


def params_from_layers(layers) -> TileParams:
    """Build TileParams from the params carried on an ExpectedLayers instance."""
    return TileParams(
        tile_thickness_voxels=int(layers.tile_thickness_voxels),
        peg_size_voxels=int(layers.peg_size_voxels),
        peg_depth_voxels=int(layers.peg_depth_voxels),
    )


def tile_color(tile: Tile) -> np.ndarray:
    """Deterministic, pleasant-ish RGB color for a tile (per face)."""
    x, y, z = tile.voxel
    h = ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
         ^ ((tile.axis * 2 + (tile.sign > 0)) * 2654435761)) & 0xFFFFFFFF
    rng = np.random.default_rng(h)
    return rng.uniform(0.30, 0.90, size=3)


def _add_quad(tris, a, b, c, d, ref_normal):
    """Append a quad (perimeter a-b-c-d) as two triangles wound so the face normal
    roughly agrees with ``ref_normal``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    nrm = np.cross(b - a, c - a)
    if np.dot(nrm, ref_normal) < 0.0:
        a, b, c, d = d, c, b, a
    tris.append((a, b, c))
    tris.append((a, c, d))


def build_face_tile(C, uhat, vhat, w, L, T, peg_u0, peg_v0, peg_size, peg_depth):
    """Build one tile mesh.

    Parameters are in millimeters. ``C`` is the min-(u,v) corner of the outer face on
    the model surface; ``uhat``/``vhat`` are unit tangents; ``w`` is the unit inward
    direction (into the base, opposite the outward normal). ``L`` is the face size,
    ``T`` the bevel/skin thickness, ``peg_u0``/``peg_v0`` the peg offset from ``C``
    along u/v, ``peg_size`` the peg side, ``peg_depth`` how far it protrudes.
    """
    C = np.asarray(C, dtype=float)
    uhat = np.asarray(uhat, dtype=float)
    vhat = np.asarray(vhat, dtype=float)
    w = np.asarray(w, dtype=float)
    n = -w  # outward normal

    # Outer face (full L x L square on the surface).
    P00 = C
    P10 = C + uhat * L
    P11 = C + uhat * L + vhat * L
    P01 = C + vhat * L

    # Inner face ring corners (inset T on each side, depth T inward).
    Q00 = C + uhat * T + vhat * T + w * T
    Q10 = C + uhat * (L - T) + vhat * T + w * T
    Q11 = C + uhat * (L - T) + vhat * (L - T) + w * T
    Q01 = C + uhat * T + vhat * (L - T) + w * T

    # Peg footprint (on the inner face plane, depth T).
    pu1 = peg_u0 + peg_size
    pv1 = peg_v0 + peg_size
    R00 = C + uhat * peg_u0 + vhat * peg_v0 + w * T
    R10 = C + uhat * pu1 + vhat * peg_v0 + w * T
    R11 = C + uhat * pu1 + vhat * pv1 + w * T
    R01 = C + uhat * peg_u0 + vhat * pv1 + w * T

    # Peg bottom (depth T + peg_depth).
    S00 = R00 + w * peg_depth
    S10 = R10 + w * peg_depth
    S11 = R11 + w * peg_depth
    S01 = R01 + w * peg_depth

    tris = []
    # Flat outer face.
    _add_quad(tris, P00, P10, P11, P01, n)
    # 45-degree beveled side walls.
    _add_quad(tris, P00, P01, Q01, Q00, -uhat)
    _add_quad(tris, P10, P11, Q11, Q10, uhat)
    _add_quad(tris, P00, P10, Q10, Q00, -vhat)
    _add_quad(tris, P01, P11, Q11, Q01, vhat)
    # Inner face ring (frame between inner square and peg footprint).
    _add_quad(tris, Q00, Q10, R10, R00, w)
    _add_quad(tris, Q10, Q11, R11, R10, w)
    _add_quad(tris, Q11, Q01, R01, R11, w)
    _add_quad(tris, Q01, Q00, R00, R01, w)
    # Peg walls.
    _add_quad(tris, R00, R10, S10, S00, -vhat)
    _add_quad(tris, R10, R11, S11, S10, uhat)
    _add_quad(tris, R11, R01, S01, S11, vhat)
    _add_quad(tris, R01, R00, S00, S01, -uhat)
    # Peg bottom cap.
    _add_quad(tris, S00, S10, S11, S01, w)

    tri_arr = np.array(tris, dtype=float)  # (M, 3, 3)
    verts = tri_arr.reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.merge_vertices()
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def generate_face_tiles(layers, params: TileParams = None,
                        with_labels: bool = False,
                        emboss_depth_mm: float = DEFAULT_EMBOSS_DEPTH_MM,
                        peg_clearance_mm: float = DEFAULT_PEG_CLEARANCE_MM,
                        peg_depth_clearance_mm: float = DEFAULT_PEG_DEPTH_CLEARANCE_MM) -> List[Tile]:
    """Generate one tile per exposed face of every surface voxel.

    Iterates ``layers.initial_voxels`` (the original, unscaled solid). A face is
    exposed when the neighboring original voxel is empty or out of bounds. Output
    ordering is deterministic so all callers reproduce identical tiles.

    ``peg_clearance_mm`` shrinks the peg cross-section per side (the base hole keeps its
    exact size) and ``peg_depth_clearance_mm`` shortens the peg so it cannot bottom out
    before the tile seats flush on the base - both for real-world print fit.

    When ``with_labels`` is True, each tile gets a unique assembly number and a paint
    color-code written flush on its outer face (a two-color inlay), and ``tile.mesh``
    is the tile body with the number cavity removed.
    """
    if params is None:
        params = params_from_layers(layers)

    vox = layers.initial_voxels
    SF = int(layers.scale_factor_int)
    S = float(layers.voxel_size_mm)
    if SF <= 0 or not np.any(vox):
        return []

    color_code_of = getattr(layers, "voxel_color_code", None)
    label_mod = None
    if with_labels:
        import tile_labels as label_mod

    L = SF * S
    T = params.tile_thickness_voxels * S
    peg_size = params.peg_size_voxels * S
    peg_depth = params.peg_depth_voxels * S

    if not (0 < T < L / 2.0):
        raise ValueError(f"tile_thickness ({T}mm) must be >0 and < L/2 ({L / 2.0}mm). "
                         "Reduce --erosion_voxels or it is too large for the voxel size.")
    inner = L - 2.0 * T
    if peg_size > inner:
        raise ValueError(f"peg_size ({peg_size}mm) exceeds inner face ({inner}mm). "
                         "Reduce --peg_size_voxels.")

    # Peg offset within the face, aligned to the same integer voxel box the base hole
    # is carved from. The peg is then shrunk (and shortened) by the print clearances so
    # it fits the exact-size hole on a real printer; it stays centered in the hole.
    start_vox = (SF // 2) - (params.peg_size_voxels // 2)
    peg_off = start_vox * S
    peg_off_fit = peg_off + peg_clearance_mm
    peg_size_fit = max(0.2, peg_size - 2.0 * peg_clearance_mm)
    peg_depth_fit = max(0.2, peg_depth - peg_depth_clearance_mm)

    dims = vox.shape
    tiles: List[Tile] = []
    next_number = 1
    for (x, y, z) in np.argwhere(vox):
        block_min = np.array([x, y, z], dtype=float) * SF * S
        cc = int(color_code_of[x, y, z]) if color_code_of is not None else 0
        for axis, sign in FACE_DIRECTIONS:
            nb = [int(x), int(y), int(z)]
            nb[axis] += sign
            exposed = (nb[axis] < 0 or nb[axis] >= dims[axis]
                       or not vox[nb[0], nb[1], nb[2]])
            if not exposed:
                continue

            ua = (axis + 1) % 3
            va = (axis + 2) % 3
            uhat = np.zeros(3); uhat[ua] = 1.0
            vhat = np.zeros(3); vhat[va] = 1.0
            n = np.zeros(3); n[axis] = float(sign)
            w = -n

            C = block_min.copy()
            C[axis] = block_min[axis] + (L if sign > 0 else 0.0)
            # C[ua], C[va] already equal block_min on u/v axes.

            mesh = build_face_tile(C, uhat, vhat, w, L, T, peg_off_fit, peg_off_fit,
                                   peg_size_fit, peg_depth_fit)
            tile = Tile(mesh=mesh, voxel=(int(x), int(y), int(z)),
                        axis=axis, sign=sign, number=next_number, color_code=cc)
            next_number += 1

            if label_mod is not None:
                poly = label_mod.label_polygon([tile.number, tile.color_code], L)
                body, number_mesh = label_mod.apply_label(
                    mesh, poly, C, uhat, vhat, w, L, emboss_depth_mm)
                tile.mesh = body
                tile.number_mesh = number_mesh

            tiles.append(tile)
    return tiles
