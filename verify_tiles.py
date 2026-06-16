"""Verify that the per-voxel skin tiles form a perfect, colorable skin over the base.

The generator in ``process_vox.py`` produces:
  * ``<name>_processed.stl``    -- the eroded core ("base"), cropped to content.
  * ``<name>_gapped_diff.stl``  -- the skin, one chunk per original voxel, laid out
                                   with a small ``gap_mm`` between voxel cells so each
                                   cell can be printed/colored independently.

This script regenerates the expected base/skin/solid voxel layers directly from the
source ``.vox`` (via ``process_vox.compute_expected_layers``), then re-voxelizes the
*actual* STL outputs back onto a common fine grid (each scaled voxel subdivided
``--subdiv`` times). Removing the gap from every tile, it then checks that:

  * tiles leave no GAPS (every expected-skin cell is covered),
  * tiles do not OVERLAP each other,
  * tiles do not OVERLAP the base,
  * base + tiles exactly equal the full scaled solid (no missing/extra material),
  * each tile stays within its own original voxel cell (per-cube colorability).

Exit code is 0 when every check passes, non-zero otherwise.
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

from process_vox import (
    compute_expected_layers,
    save_vox_file,
    DEFAULT_STL_VOXEL_SIZE_MM,
    DEFAULT_GAP_MM,
)


def upsample(arr, n):
    """Repeat a 3D boolean array ``n`` times along each axis."""
    return np.repeat(np.repeat(np.repeat(arr, n, axis=0), n, axis=1), n, axis=2)


def downsample_any(fine, n):
    """Reduce a fine 3D array to scaled-voxel resolution; a voxel is True if any of
    its ``n^3`` fine cells is True."""
    sx, sy, sz = (np.array(fine.shape) // n)
    trimmed = fine[: sx * n, : sy * n, : sz * n]
    return trimmed.reshape(sx, n, sy, n, sz, n).any(axis=(1, 3, 5))


def occupied_fine_indices(mesh, pitch, fine_shape):
    """Return integer indices of fine cells whose centers fall inside ``mesh``.

    Cell centers are tested exactly with ``mesh.contains`` (point-in-mesh). Sampling
    at centers - never on cube boundary planes - avoids the boundary ambiguity that
    plagues surface voxelization when ``pitch`` equals the cube size. Only cells
    within the mesh bounding box are tested, chunked along X to bound memory.
    """
    if mesh is None or mesh.is_empty:
        return np.empty((0, 3), dtype=int)

    lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
    fs = np.array(fine_shape)
    # Fine cell i has center (i + 0.5) * pitch.
    i_lo = np.clip(np.floor(lo / pitch - 0.5).astype(int), 0, fs - 1)
    i_hi = np.clip(np.ceil(hi / pitch - 0.5).astype(int), 0, fs - 1)
    if np.any(i_hi < i_lo):
        return np.empty((0, 3), dtype=int)

    ys = np.arange(i_lo[1], i_hi[1] + 1)
    zs = np.arange(i_lo[2], i_hi[2] + 1)
    yz = np.stack(np.meshgrid(ys, zs, indexing="ij"), axis=-1).reshape(-1, 2)

    collected = []
    for xi in range(i_lo[0], i_hi[0] + 1):
        pts = np.empty((len(yz), 3), dtype=float)
        pts[:, 0] = (xi + 0.5) * pitch
        pts[:, 1] = (yz[:, 0] + 0.5) * pitch
        pts[:, 2] = (yz[:, 1] + 0.5) * pitch
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


def voxelize_base(processed_path, layers, pitch, fine_shape):
    """Re-voxelize the base STL onto the scaled fine grid."""
    mesh = trimesh.load_mesh(processed_path)
    crop = layers.crop_min_coords if layers.crop_min_coords is not None else (0, 0, 0)
    # The base STL starts at its own origin; shift it back to where the cropped core
    # sits inside the scaled grid.
    mesh.apply_translation(np.array(crop, dtype=float) * layers.voxel_size_mm)
    idx = occupied_fine_indices(mesh, pitch, fine_shape)
    return scatter(idx, fine_shape, dtype=bool)


def voxelize_tiles(gapped_path, layers, pitch, fine_shape, subdiv):
    """Re-voxelize the gapped skin STL, un-gapping each tile component.

    Returns ``(tile_count, leaks)`` where ``tile_count`` is a per-cell hit count
    (values > 1 indicate tile-vs-tile overlap) and ``leaks`` lists components that
    bled outside their own original voxel cell.
    """
    gap_mesh = trimesh.load_mesh(gapped_path)
    components = gap_mesh.split(only_watertight=False)
    if len(components) == 0:
        components = [gap_mesh]

    SF = layers.scale_factor_int
    S = layers.voxel_size_mm
    gap = layers.gap_mm
    pitch_block = SF * S + gap  # gapped spacing between original voxels

    tile_count = np.zeros(fine_shape, dtype=np.int32)
    leaks = []

    for comp in components:
        if comp.is_empty:
            continue
        min_corner = np.asarray(comp.bounds[0])
        # Each component lives entirely within one original voxel's gapped block.
        sm = np.floor(min_corner / pitch_block + 1e-6).astype(int)
        sm = np.clip(sm, 0, None)

        assembled = comp.copy()
        assembled.apply_translation(-sm.astype(float) * gap)  # remove the gap offset

        idx = occupied_fine_indices(assembled, pitch, fine_shape)
        if len(idx) == 0:
            continue
        np.add.at(tile_count, tuple(idx.T), 1)

        # Leakage: this tile's cells must stay within its sm block at fine resolution.
        blk_lo = sm * SF * subdiv
        blk_hi = (sm + 1) * SF * subdiv
        outside = np.any((idx < blk_lo) | (idx >= blk_hi), axis=1)
        if np.any(outside):
            leaks.append({"voxel": sm.tolist(), "leaked_cells": int(outside.sum())})

    return tile_count, leaks


def sample_coords(mask, limit=20):
    """Return up to ``limit`` [x,y,z] indices where ``mask`` is True (for reports)."""
    idx = np.argwhere(mask)
    return idx[:limit].tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Verify per-voxel skin tiles fit the base model with no gaps or overlaps."
    )
    parser.add_argument("input_vox_file", help="Path to the source .vox file.")
    parser.add_argument("--scale_factor", type=float, default=10.0,
                        help="Scale factor used to generate the STLs (default: 10.0).")
    parser.add_argument("--erosion_voxels", type=int, default=1,
                        help="Erosion layers used to generate the STLs (default: 1).")
    parser.add_argument("--subdiv", type=int, default=2,
                        help="Sub-voxel sampling resolution per scaled voxel (default: 2).")
    parser.add_argument("--processed_stl", default=None,
                        help="Override path to the base STL (default: <name>_processed.stl).")
    parser.add_argument("--gapped_stl", default=None,
                        help="Override path to the gapped skin STL (default: <name>_gapped_diff.stl).")
    parser.add_argument("--report", default=None,
                        help="Optional path to write a JSON report.")
    parser.add_argument("--debug_vox", action="store_true",
                        help="Write <name>_verify_gaps.vox / _verify_overlaps.vox for inspection.")
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
    gapped_path = args.gapped_stl or os.path.join(out_dir, f"{base_name}_gapped_diff.stl")

    for label, path in (("base", processed_path), ("gapped skin", gapped_path)):
        if not os.path.exists(path):
            print(f"Error: {label} STL not found: '{path}'. "
                  "Run process_vox.py first or pass an explicit path.")
            return 2

    print(f"Regenerating expected voxel layers from '{input_path}' "
          f"(scale={args.scale_factor}, erosion={args.erosion_voxels})...")
    layers = compute_expected_layers(
        input_path, args.scale_factor, args.erosion_voxels,
        stl_voxel_size_mm=DEFAULT_STL_VOXEL_SIZE_MM, gap_mm=DEFAULT_GAP_MM)

    scaled = layers.scaled_solid
    if not np.any(scaled):
        print("Error: source model is empty after scaling; nothing to verify.")
        return 2

    N = args.subdiv
    S = layers.voxel_size_mm
    pitch = S / N
    fine_shape = tuple(int(d) * N for d in scaled.shape)

    print(f"Scaled grid: {scaled.shape}, fine grid: {fine_shape} (subdiv={N}, pitch={pitch:.4f}mm).")

    # Expected layers upsampled to the fine grid.
    scaled_fine = upsample(scaled, N)
    skin_fine = upsample(layers.skin, N)
    base_expected_fine = upsample(layers.base_aligned, N)

    print(f"Re-voxelizing base STL '{processed_path}'...")
    base_occ = voxelize_base(processed_path, layers, pitch, fine_shape)

    print(f"Re-voxelizing gapped skin STL '{gapped_path}'...")
    tile_count, leaks = voxelize_tiles(gapped_path, layers, pitch, fine_shape, N)
    tile_occ = tile_count > 0

    # --- Checks ---------------------------------------------------------------
    gaps = skin_fine & ~tile_occ            # expected skin not covered by any tile
    overlap_tt = tile_count > 1             # tile-vs-tile overlap
    overlap_tb = tile_occ & base_occ        # tile-vs-base overlap
    assembled = base_occ | tile_occ
    missing = scaled_fine & ~assembled      # solid not covered by base or tiles
    extra = assembled & ~scaled_fine        # material outside the solid
    base_mismatch = base_occ ^ base_expected_fine  # informational: base mesh vs expected

    cell_volume = float(N) ** 3  # fine cells per scaled voxel, for human-readable counts

    def count(mask):
        return int(np.count_nonzero(mask))

    results = {
        "gaps_between_or_under_tiles": count(gaps),
        "tile_vs_tile_overlap": count(overlap_tt),
        "tile_vs_base_overlap": count(overlap_tb),
        "solid_not_covered": count(missing),
        "material_outside_solid": count(extra),
        "tile_leakage_components": len(leaks),
    }
    # base_mismatch is informational only (does not fail the run).
    info = {"base_mesh_vs_expected_mismatch_cells": count(base_mismatch)}

    failed = any(v > 0 for v in results.values())

    print("\n================ Verification summary ================")
    print(f"{'Check':<34}{'fine cells':>12}{'~voxels':>10}   status")
    print("-" * 70)
    check_order = [
        ("Gaps (skin uncovered)", "gaps_between_or_under_tiles"),
        ("Tile vs tile overlap", "tile_vs_tile_overlap"),
        ("Tile vs base overlap", "tile_vs_base_overlap"),
        ("Solid not covered", "solid_not_covered"),
        ("Material outside solid", "material_outside_solid"),
    ]
    for label, key in check_order:
        c = results[key]
        status = "OK" if c == 0 else "FAIL"
        print(f"{label:<34}{c:>12}{c / cell_volume:>10.1f}   {status}")
    leak_status = "OK" if len(leaks) == 0 else "FAIL"
    print(f"{'Tile leakage (per-cube color)':<34}{len(leaks):>12}{'':>10}   {leak_status}")
    print("-" * 70)
    print(f"(info) base mesh vs expected mismatch cells: {info['base_mesh_vs_expected_mismatch_cells']}")
    print("=" * 54)
    overall = "PASSED" if not failed else "FAILED"
    print(f"Overall: {overall} -- tiles {'form' if not failed else 'do NOT form'} "
          f"a perfect skin over the base.\n")

    if leaks:
        print("Leaking tiles (component bled outside its own voxel cell):")
        for lk in leaks[:10]:
            print(f"  voxel {lk['voxel']}: {lk['leaked_cells']} cells")
        if len(leaks) > 10:
            print(f"  ... and {len(leaks) - 10} more")

    # --- Optional JSON report -------------------------------------------------
    if args.report:
        report = {
            "input_vox": os.path.abspath(input_path),
            "processed_stl": os.path.abspath(processed_path),
            "gapped_stl": os.path.abspath(gapped_path),
            "params": {
                "scale_factor": args.scale_factor,
                "erosion_voxels": args.erosion_voxels,
                "subdiv": N,
                "voxel_size_mm": S,
                "gap_mm": layers.gap_mm,
                "scale_factor_int": layers.scale_factor_int,
            },
            "scaled_shape": list(scaled.shape),
            "fine_shape": list(fine_shape),
            "counts_fine_cells": results,
            "info": info,
            "samples_fine_indices": {
                "gaps": sample_coords(gaps),
                "tile_vs_tile_overlap": sample_coords(overlap_tt),
                "tile_vs_base_overlap": sample_coords(overlap_tb),
                "solid_not_covered": sample_coords(missing),
                "material_outside_solid": sample_coords(extra),
            },
            "leaks": leaks,
            "passed": not failed,
        }
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON report to '{args.report}'.")

    # --- Optional debug .vox --------------------------------------------------
    if args.debug_vox:
        palette = layers.palette
        gaps_vox = downsample_any(gaps, N)
        overlaps_vox = downsample_any(overlap_tt | overlap_tb, N)
        gaps_path = os.path.join(out_dir, f"{base_name}_verify_gaps.vox")
        overlaps_path = os.path.join(out_dir, f"{base_name}_verify_overlaps.vox")
        if np.any(gaps_vox):
            print(f"Writing gap voxels to '{gaps_path}'...")
            save_vox_file(gaps_path, gaps_vox, palette)
        else:
            print("No gap voxels to write.")
        if np.any(overlaps_vox):
            print(f"Writing overlap voxels to '{overlaps_path}'...")
            save_vox_file(overlaps_path, overlaps_vox, palette)
        else:
            print("No overlap voxels to write.")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
