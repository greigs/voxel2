"""Verify that the flat, mitered, per-face skin tiles form a perfect colorable skin.

With the flat mitered tiles (one per exposed voxel face: full-face outer square,
45-degree inward-beveled side walls, central registration peg), "perfect fit" is now
an OUTER-SURFACE property plus a no-collision property:

  * Surface coverage: there is exactly one full-face tile per exposed voxel face, so
    the visible surface is tiled with no gaps and no double-cover (by construction).
  * Tile-vs-tile: no two tile bodies physically intersect (the inward bevels and
    miters guarantee they only touch at zero-volume seams).
  * Tile-vs-base: tile bodies only enter the base through their peg holes, so no
    tile body volume overlaps the base body.

Tiles are generated analytically by ``tiles.generate_face_tiles`` (the same code the
writer uses), and the base is re-voxelized from ``<name>_processed.stl``. Both are
sampled at fine-cell centers (exact point-in-mesh) on a common grid.

Exit code is 0 when every check passes, non-zero otherwise.
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

import tiles as tiles_mod
from process_vox import (
    compute_expected_layers,
    save_vox_file,
    DEFAULT_STL_VOXEL_SIZE_MM,
    DEFAULT_GAP_MM,
    DEFAULT_PEG_DEPTH_VOXELS,
    FACE_DIRECTIONS,
)

# Per-axis sub-cell sampling offset. Deliberately asymmetric and chosen so no pair of
# offsets has an integer sum or difference: this keeps cell centers off the axis-aligned
# faces AND off the 45-degree miter seams (planes x +/- z = k*S), avoiding double-counting
# tiles that merely touch at a zero-volume seam.
SAMPLE_OFFSET = np.array([0.31, 0.53, 0.79])


def downsample_any(fine, n):
    """Reduce a fine 3D array to scaled-voxel resolution; a voxel is True if any of
    its ``n^3`` fine cells is True."""
    sx, sy, sz = (np.array(fine.shape) // n)
    trimmed = fine[: sx * n, : sy * n, : sz * n]
    return trimmed.reshape(sx, n, sy, n, sz, n).any(axis=(1, 3, 5))


def occupied_fine_indices(mesh, pitch, fine_shape):
    """Return integer indices of fine cells whose centers fall inside ``mesh``.

    Cell centers are tested exactly with ``mesh.contains`` (point-in-mesh). Sampling
    at centers - never on cube boundary planes - avoids boundary ambiguity. Only cells
    within the mesh bounding box are tested, chunked along X to bound memory.
    """
    if mesh is None or mesh.is_empty:
        return np.empty((0, 3), dtype=int)

    lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
    fs = np.array(fine_shape)
    off = SAMPLE_OFFSET
    i_lo = np.clip(np.floor(lo / pitch - off).astype(int), 0, fs - 1)
    i_hi = np.clip(np.ceil(hi / pitch - off).astype(int), 0, fs - 1)
    if np.any(i_hi < i_lo):
        return np.empty((0, 3), dtype=int)

    ys = np.arange(i_lo[1], i_hi[1] + 1)
    zs = np.arange(i_lo[2], i_hi[2] + 1)
    yz = np.stack(np.meshgrid(ys, zs, indexing="ij"), axis=-1).reshape(-1, 2)

    collected = []
    for xi in range(i_lo[0], i_hi[0] + 1):
        pts = np.empty((len(yz), 3), dtype=float)
        pts[:, 0] = (xi + off[0]) * pitch
        pts[:, 1] = (yz[:, 0] + off[1]) * pitch
        pts[:, 2] = (yz[:, 1] + off[2]) * pitch
        inside = mesh.contains(pts)
        if np.any(inside):
            sel = yz[inside]
            col = np.empty((len(sel), 3), dtype=int)
            col[:, 0] = xi
            col[:, 1:] = sel
            collected.append(col)

    if collected:
        return np.concatenate(collected, axis=0)
    return np.empty((0, 3), dtype=int)


def scatter(indices, shape, dtype=bool):
    """Build an array of ``shape`` with ``True``/+1 at the given indices."""
    out = np.zeros(shape, dtype=dtype)
    if len(indices):
        if dtype == bool:
            out[tuple(indices.T)] = True
        else:
            np.add.at(out, tuple(indices.T), 1)
    return out


def count_exposed_faces(vox):
    """Count exposed faces of a boolean voxel array (one tile is expected per face)."""
    dims = vox.shape
    n = 0
    for (x, y, z) in np.argwhere(vox):
        for axis, sign in FACE_DIRECTIONS:
            nb = [int(x), int(y), int(z)]
            nb[axis] += sign
            if nb[axis] < 0 or nb[axis] >= dims[axis] or not vox[nb[0], nb[1], nb[2]]:
                n += 1
    return n


def build_tile_occupancy(tile_list, pitch, fine_shape):
    """Voxelize each tile body onto the fine grid.

    Returns ``(tile_count, per_tile_idx)`` where ``tile_count`` is a per-cell hit
    count (>1 means tiles overlap) and ``per_tile_idx`` is the list of occupied
    indices per tile.
    """
    tile_count = np.zeros(fine_shape, dtype=np.int32)
    per_tile_idx = []
    for t in tile_list:
        idx = occupied_fine_indices(t.mesh, pitch, fine_shape)
        per_tile_idx.append(idx)
        if len(idx):
            np.add.at(tile_count, tuple(idx.T), 1)
    return tile_count, per_tile_idx


def voxelize_base(processed_path, layers, pitch, fine_shape):
    """Re-voxelize the base STL onto the scaled fine grid (aligned via crop offset)."""
    mesh = trimesh.load_mesh(processed_path)
    crop = layers.crop_min_coords if layers.crop_min_coords is not None else (0, 0, 0)
    mesh.apply_translation(np.array(crop, dtype=float) * layers.voxel_size_mm)
    idx = occupied_fine_indices(mesh, pitch, fine_shape)
    return scatter(idx, fine_shape, dtype=bool)


def sample_coords(mask, limit=20):
    """Return up to ``limit`` [x,y,z] indices where ``mask`` is True (for reports)."""
    idx = np.argwhere(mask)
    return idx[:limit].tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Verify flat mitered skin tiles cover the surface with no gaps or overlaps."
    )
    parser.add_argument("input_vox_file", help="Path to the source .vox file.")
    parser.add_argument("--scale_factor", type=float, default=10.0)
    parser.add_argument("--erosion_voxels", type=int, default=1)
    parser.add_argument("--peg_size_voxels", type=int, default=None)
    parser.add_argument("--peg_depth_voxels", type=int, default=DEFAULT_PEG_DEPTH_VOXELS)
    parser.add_argument("--subdiv", type=int, default=2,
                        help="Sub-voxel sampling resolution per scaled voxel (default: 2).")
    parser.add_argument("--processed_stl", default=None,
                        help="Override path to the base STL (default: <name>_processed.stl).")
    parser.add_argument("--report", default=None, help="Optional path to write a JSON report.")
    parser.add_argument("--debug_vox", action="store_true",
                        help="Write <name>_verify_overlaps.vox for inspection.")
    args = parser.parse_args()

    if args.subdiv < 1:
        print("Error: --subdiv must be >= 1.")
        return 2

    input_path = args.input_vox_file
    if not os.path.exists(input_path):
        print(f"Error: Input .vox file '{input_path}' not found.")
        return 2

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.dirname(input_path) or "."
    processed_path = args.processed_stl or os.path.join(out_dir, f"{base_name}_processed.stl")
    if not os.path.exists(processed_path):
        print(f"Error: base STL not found: '{processed_path}'. Run process_vox.py first.")
        return 2

    print(f"Regenerating base + tile geometry from '{input_path}' "
          f"(scale={args.scale_factor}, erosion={args.erosion_voxels})...")
    layers = compute_expected_layers(
        input_path, args.scale_factor, args.erosion_voxels,
        stl_voxel_size_mm=DEFAULT_STL_VOXEL_SIZE_MM, gap_mm=DEFAULT_GAP_MM,
        peg_size_voxels=args.peg_size_voxels, peg_depth_voxels=args.peg_depth_voxels)

    scaled = layers.scaled_solid
    if not np.any(scaled):
        print("Error: source model is empty after scaling; nothing to verify.")
        return 2

    N = args.subdiv
    S = layers.voxel_size_mm
    pitch = S / N
    fine_shape = tuple(int(d) * N for d in scaled.shape)
    print(f"Scaled grid: {scaled.shape}, fine grid: {fine_shape} (subdiv={N}, pitch={pitch:.4f}mm).")

    print("Generating flat mitered tiles...")
    tile_list = tiles_mod.generate_face_tiles(layers)
    n_exposed = count_exposed_faces(layers.initial_voxels)
    print(f"  {len(tile_list)} tiles, {n_exposed} exposed faces.")

    print(f"Re-voxelizing base STL '{processed_path}'...")
    base_occ = voxelize_base(processed_path, layers, pitch, fine_shape)

    print("Voxelizing tiles...")
    tile_count, _ = build_tile_occupancy(tile_list, pitch, fine_shape)
    tile_occ = tile_count > 0

    # --- Checks ---------------------------------------------------------------
    overlap_tt = tile_count > 1            # tile-vs-tile body intersection
    overlap_tb = tile_occ & base_occ       # tile body intersects base body
    coverage_ok = (len(tile_list) == n_exposed) and (n_exposed > 0)

    def count(mask):
        return int(np.count_nonzero(mask))

    cell_volume = float(N) ** 3
    n_tt = count(overlap_tt)
    n_tb = count(overlap_tb)

    failed = (not coverage_ok) or n_tt > 0 or n_tb > 0

    print("\n================ Verification summary ================")
    print(f"{'Check':<36}{'value':>14}   status")
    print("-" * 64)
    cov_status = "OK" if coverage_ok else "FAIL"
    print(f"{'Surface coverage (tiles==faces)':<36}{str(len(tile_list)) + '/' + str(n_exposed):>14}   {cov_status}")
    print(f"{'Tile vs tile overlap (cells)':<36}{n_tt:>14}   {'OK' if n_tt == 0 else 'FAIL'}")
    print(f"{'Tile vs base overlap (cells)':<36}{n_tb:>14}   {'OK' if n_tb == 0 else 'FAIL'}")
    print("-" * 64)
    print(f"(approx voxel-equivalents: tt={n_tt / cell_volume:.1f}, tb={n_tb / cell_volume:.1f})")
    print("=" * 54)
    overall = "PASSED" if not failed else "FAILED"
    print(f"Overall: {overall} -- tiles {'form' if not failed else 'do NOT form'} "
          f"a gapless, non-overlapping skin.\n")

    if args.report:
        report = {
            "input_vox": os.path.abspath(input_path),
            "processed_stl": os.path.abspath(processed_path),
            "params": {
                "scale_factor": args.scale_factor,
                "erosion_voxels": args.erosion_voxels,
                "tile_thickness_voxels": layers.tile_thickness_voxels,
                "peg_size_voxels": layers.peg_size_voxels,
                "peg_depth_voxels": layers.peg_depth_voxels,
                "subdiv": N,
                "voxel_size_mm": S,
                "gap_mm": layers.gap_mm,
                "scale_factor_int": layers.scale_factor_int,
            },
            "scaled_shape": list(scaled.shape),
            "fine_shape": list(fine_shape),
            "n_tiles": len(tile_list),
            "n_exposed_faces": n_exposed,
            "counts_fine_cells": {
                "tile_vs_tile_overlap": n_tt,
                "tile_vs_base_overlap": n_tb,
            },
            "samples_fine_indices": {
                "tile_vs_tile_overlap": sample_coords(overlap_tt),
                "tile_vs_base_overlap": sample_coords(overlap_tb),
            },
            "coverage_ok": bool(coverage_ok),
            "passed": not failed,
        }
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON report to '{args.report}'.")

    if args.debug_vox:
        overlaps_vox = downsample_any(overlap_tt | overlap_tb, N)
        overlaps_path = os.path.join(out_dir, f"{base_name}_verify_overlaps.vox")
        if np.any(overlaps_vox):
            print(f"Writing overlap voxels to '{overlaps_path}'...")
            save_vox_file(overlaps_path, overlaps_vox, layers.palette)
        else:
            print("No overlap voxels to write.")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
