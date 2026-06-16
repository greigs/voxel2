"""Interactive viewer for inspecting how the skin tiles fit over the base model.

Loads the base (``<name>_processed.stl``) and the per-voxel skin tiles
(``<name>_gapped_diff.stl``), removes the inter-tile gap so everything sits in its
true assembled position, and opens a PyVista window with:

  * an "Explode" slider that pushes every tile radially outward from the model
    center (0 = fully assembled / perfect fit, higher = exploded apart), and
  * automatic highlighting (in red) of any tiles that overlap another tile or the
    base, using the same voxel-overlap logic as ``verify_tiles.py``.

Use ``--screenshot out.png --explode 0.0`` to render a single frame off-screen
(useful on headless machines or for quick checks).
"""

import argparse
import os
import sys

import numpy as np
import trimesh
import pyvista as pv

from process_vox import compute_expected_layers, DEFAULT_STL_VOXEL_SIZE_MM, DEFAULT_GAP_MM
from verify_tiles import occupied_fine_indices, scatter


def color_for_voxel(sm):
    """Deterministic pleasant-ish color for an original voxel coordinate."""
    h = (int(sm[0]) * 73856093) ^ (int(sm[1]) * 19349663) ^ (int(sm[2]) * 83492791)
    rng = np.random.default_rng(h & 0xFFFFFFFF)
    return rng.uniform(0.30, 0.90, size=3)


def ungap_components(gap_mesh, layers):
    """Split the gapped skin mesh into components and translate each back to its
    true assembled position. Returns a list of (mesh, sm_voxel) tuples."""
    components = gap_mesh.split(only_watertight=False)
    if len(components) == 0:
        components = [gap_mesh]

    SF = layers.scale_factor_int
    S = layers.voxel_size_mm
    gap = layers.gap_mm
    pitch_block = SF * S + gap

    assembled = []
    for comp in components:
        if comp.is_empty:
            continue
        sm = np.floor(np.asarray(comp.bounds[0]) / pitch_block + 1e-6).astype(int)
        sm = np.clip(sm, 0, None)
        a = comp.copy()
        a.apply_translation(-sm.astype(float) * gap)
        assembled.append((a, sm))
    return assembled


def detect_overlaps(assembled, base_mesh, layers, subdiv):
    """Return a boolean list flagging which assembled tiles overlap another tile or
    the base, plus aggregate cell counts. Uses exact center-sampling on a fine grid.
    """
    S = layers.voxel_size_mm
    pitch = S / subdiv
    fine_shape = tuple(int(d) * subdiv for d in layers.scaled_solid.shape)

    base_idx = occupied_fine_indices(base_mesh, pitch, fine_shape)
    base_occ = scatter(base_idx, fine_shape, dtype=bool)

    tile_count = np.zeros(fine_shape, dtype=np.int32)
    comp_indices = []
    for a, _sm in assembled:
        idx = occupied_fine_indices(a, pitch, fine_shape)
        comp_indices.append(idx)
        if len(idx):
            np.add.at(tile_count, tuple(idx.T), 1)

    overlap_cell = tile_count > 1
    flags = []
    n_tt = 0
    n_tb = 0
    for idx in comp_indices:
        if len(idx) == 0:
            flags.append(False)
            continue
        tup = tuple(idx.T)
        hits_tt = bool(np.any(overlap_cell[tup]))
        hits_tb = bool(np.any(base_occ[tup]))
        n_tt += int(hits_tt)
        n_tb += int(hits_tb)
        flags.append(hits_tt or hits_tb)

    stats = {
        "overlap_cells_tile_tile": int(np.count_nonzero(overlap_cell)),
        "overlap_cells_tile_base": int(np.count_nonzero(base_occ & (tile_count > 0))),
        "tiles_overlapping": int(sum(flags)),
    }
    return flags, stats


def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer: explode the skin tiles and highlight overlaps."
    )
    parser.add_argument("input_vox_file", help="Path to the source .vox file.")
    parser.add_argument("--scale_factor", type=float, default=10.0)
    parser.add_argument("--erosion_voxels", type=int, default=1)
    parser.add_argument("--subdiv", type=int, default=1,
                        help="Sub-voxel resolution for overlap detection (default: 1).")
    parser.add_argument("--max_explode", type=float, default=2.0,
                        help="Maximum explosion multiplier on the slider (default: 2.0).")
    parser.add_argument("--processed_stl", default=None)
    parser.add_argument("--gapped_stl", default=None)
    parser.add_argument("--no-base", action="store_true", help="Do not draw the base model.")
    parser.add_argument("--no-overlap-check", action="store_true",
                        help="Skip overlap detection (faster startup).")
    parser.add_argument("--screenshot", default=None,
                        help="Render one frame off-screen to this PNG and exit.")
    parser.add_argument("--explode", type=float, default=0.0,
                        help="Explosion value for --screenshot (default: 0.0 = assembled).")
    args = parser.parse_args()

    input_path = args.input_vox_file
    if not os.path.exists(input_path):
        print(f"Error: Input .vox file '{input_path}' not found.")
        return 2

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.dirname(input_path) or "."
    processed_path = args.processed_stl or os.path.join(out_dir, f"{base_name}_processed.stl")
    gapped_path = args.gapped_stl or os.path.join(out_dir, f"{base_name}_gapped_diff.stl")

    if not os.path.exists(gapped_path):
        print(f"Error: gapped skin STL not found: '{gapped_path}'. Run process_vox.py first.")
        return 2
    draw_base = not args.no_base and os.path.exists(processed_path)
    if not args.no_base and not os.path.exists(processed_path):
        print(f"Warning: base STL '{processed_path}' not found; showing tiles only.")

    print(f"Regenerating layout metadata from '{input_path}'...")
    layers = compute_expected_layers(
        input_path, args.scale_factor, args.erosion_voxels,
        stl_voxel_size_mm=DEFAULT_STL_VOXEL_SIZE_MM, gap_mm=DEFAULT_GAP_MM)

    base_mesh = None
    if draw_base or not args.no_overlap_check:
        if os.path.exists(processed_path):
            base_mesh = trimesh.load_mesh(processed_path)
            crop = layers.crop_min_coords if layers.crop_min_coords is not None else (0, 0, 0)
            base_mesh.apply_translation(np.array(crop, dtype=float) * layers.voxel_size_mm)

    print(f"Loading and un-gapping tiles from '{gapped_path}'...")
    gap_mesh = trimesh.load_mesh(gapped_path)
    assembled = ungap_components(gap_mesh, layers)
    print(f"  {len(assembled)} tile components.")

    flags = [False] * len(assembled)
    if not args.no_overlap_check and base_mesh is not None:
        print(f"Detecting overlaps (subdiv={args.subdiv})...")
        flags, stats = detect_overlaps(assembled, base_mesh, layers, args.subdiv)
        print(f"  {stats['tiles_overlapping']} tiles overlap another tile or the base.")

    # Model center for radial explosion (use the assembled bounding box center).
    all_lo = []
    all_hi = []
    for a, _ in assembled:
        all_lo.append(a.bounds[0])
        all_hi.append(a.bounds[1])
    if base_mesh is not None:
        all_lo.append(base_mesh.bounds[0])
        all_hi.append(base_mesh.bounds[1])
    lo = np.min(np.array(all_lo), axis=0)
    hi = np.max(np.array(all_hi), axis=0)
    center = (lo + hi) / 2.0

    off_screen = args.screenshot is not None
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")

    if draw_base and base_mesh is not None:
        base_actor = plotter.add_mesh(pv.wrap(base_mesh), color=(0.75, 0.75, 0.78),
                                      name="base", opacity=1.0)
    else:
        base_actor = None

    n_overlap = int(sum(flags))
    tile_actors = []  # (actor, direction_vector)
    for i, (a, sm) in enumerate(assembled):
        direction = np.asarray(a.centroid) - center
        if flags[i]:
            color = (0.90, 0.10, 0.10)
        else:
            color = color_for_voxel(sm)
        actor = plotter.add_mesh(pv.wrap(a), color=color, name=f"tile_{i}")
        tile_actors.append((actor, direction))

    def set_explode(value):
        for actor, direction in tile_actors:
            actor.SetPosition(*(direction * value))

    title_lines = [
        f"{len(assembled)} tiles | {n_overlap} overlapping (red)",
        "Slider: Explode (0 = assembled)",
    ]
    plotter.add_text("\n".join(title_lines), font_size=10, color="black", name="hud")

    if not off_screen:
        plotter.add_slider_widget(
            set_explode, [0.0, args.max_explode], value=0.0, title="Explode",
            pointa=(0.30, 0.92), pointb=(0.70, 0.92),
        )

        if base_actor is not None:
            def toggle_base(state):
                base_actor.SetVisibility(bool(state))
            plotter.add_checkbox_button_widget(toggle_base, value=True, position=(10, 10),
                                               size=25, color_on=(0.2, 0.2, 0.8))
            plotter.add_text("Toggle base", position=(45, 12), font_size=8, color="black",
                             name="base_label")

    set_explode(args.explode if off_screen else 0.0)

    if off_screen:
        plotter.show(screenshot=args.screenshot, auto_close=True)
        print(f"Wrote screenshot to '{args.screenshot}' (explode={args.explode}).")
    else:
        print("Opening interactive viewer. Drag the 'Explode' slider; red = overlapping.")
        plotter.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
