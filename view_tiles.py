"""Interactive viewer for inspecting how the flat mitered skin tiles fit the base.

Builds the base (from ``<name>_processed.stl``) and the exact per-face tiles directly
from ``tiles.generate_face_tiles`` (the same geometry the writer and verifier use), so
each tile keeps its own identity for robust coloring (no merge-at-miter problem). Opens
a PyVista window with:

  * an "Explode" slider that pushes every tile radially outward from the model center
    (0 = fully assembled / perfect fit, higher = exploded apart), and
  * automatic highlighting (in red) of any tile that overlaps another tile or the base,
    using the same fine-grid overlap logic as ``verify_tiles.py``.

Use ``--screenshot out.png --explode 0.0`` to render a single frame off-screen.
"""

import argparse
import os
import sys

import numpy as np
import trimesh
import pyvista as pv

import tiles as tiles_mod
from process_vox import (
    compute_expected_layers,
    DEFAULT_STL_VOXEL_SIZE_MM,
    DEFAULT_GAP_MM,
    DEFAULT_PEG_DEPTH_VOXELS,
)
from verify_tiles import occupied_fine_indices, scatter


def detect_overlaps(tile_list, base_mesh, layers, subdiv):
    """Return a per-tile boolean list flagging tiles that overlap another tile or the
    base, plus aggregate cell counts. Uses exact off-seam center-sampling."""
    S = layers.voxel_size_mm
    pitch = S / subdiv
    fine_shape = tuple(int(d) * subdiv for d in layers.scaled_solid.shape)

    base_occ = None
    if base_mesh is not None:
        base_idx = occupied_fine_indices(base_mesh, pitch, fine_shape)
        base_occ = scatter(base_idx, fine_shape, dtype=bool)

    tile_count = np.zeros(fine_shape, dtype=np.int32)
    tile_indices = []
    for t in tile_list:
        idx = occupied_fine_indices(t.mesh, pitch, fine_shape)
        tile_indices.append(idx)
        if len(idx):
            np.add.at(tile_count, tuple(idx.T), 1)

    overlap_cell = tile_count > 1
    flags = []
    for idx in tile_indices:
        if len(idx) == 0:
            flags.append(False)
            continue
        tup = tuple(idx.T)
        hits = bool(np.any(overlap_cell[tup]))
        if not hits and base_occ is not None:
            hits = bool(np.any(base_occ[tup]))
        flags.append(hits)

    stats = {
        "overlap_cells_tile_tile": int(np.count_nonzero(overlap_cell)),
        "overlap_cells_tile_base": int(np.count_nonzero(base_occ & (tile_count > 0)))
        if base_occ is not None else 0,
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
    parser.add_argument("--peg_size_voxels", type=int, default=None)
    parser.add_argument("--peg_depth_voxels", type=int, default=DEFAULT_PEG_DEPTH_VOXELS)
    parser.add_argument("--subdiv", type=int, default=1,
                        help="Sub-voxel resolution for overlap detection (default: 1).")
    parser.add_argument("--max_explode", type=float, default=2.0,
                        help="Maximum explosion multiplier on the slider (default: 2.0).")
    parser.add_argument("--processed_stl", default=None)
    parser.add_argument("--no-base", action="store_true", help="Do not draw the base model.")
    parser.add_argument("--no_numbers", action="store_true",
                        help="Do not build/show the flat number inlays.")
    parser.add_argument("--number_color", default="black",
                        help="Color for the number inlays (default: black).")
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

    draw_base = not args.no_base and os.path.exists(processed_path)
    if not args.no_base and not os.path.exists(processed_path):
        print(f"Warning: base STL '{processed_path}' not found; showing tiles only.")

    print(f"Regenerating base + tile geometry from '{input_path}'...")
    layers = compute_expected_layers(
        input_path, args.scale_factor, args.erosion_voxels,
        stl_voxel_size_mm=DEFAULT_STL_VOXEL_SIZE_MM, gap_mm=DEFAULT_GAP_MM,
        peg_size_voxels=args.peg_size_voxels, peg_depth_voxels=args.peg_depth_voxels)

    base_mesh = None
    if (draw_base or not args.no_overlap_check) and os.path.exists(processed_path):
        base_mesh = trimesh.load_mesh(processed_path)
        crop = layers.crop_min_coords if layers.crop_min_coords is not None else (0, 0, 0)
        base_mesh.apply_translation(np.array(crop, dtype=float) * layers.voxel_size_mm)

    with_labels = not args.no_numbers
    print(f"Generating flat mitered tiles{' with number inlays' if with_labels else ''}...")
    tile_list = tiles_mod.generate_face_tiles(layers, with_labels=with_labels)
    print(f"  {len(tile_list)} tiles.")
    if not tile_list:
        print("No tiles to show.")
        return 2

    flags = [False] * len(tile_list)
    if not args.no_overlap_check:
        print(f"Detecting overlaps (subdiv={args.subdiv})...")
        flags, stats = detect_overlaps(tile_list, base_mesh, layers, args.subdiv)
        print(f"  {stats['tiles_overlapping']} tiles overlap another tile or the base.")

    # Model center for radial explosion.
    all_lo = [t.mesh.bounds[0] for t in tile_list]
    all_hi = [t.mesh.bounds[1] for t in tile_list]
    if base_mesh is not None:
        all_lo.append(base_mesh.bounds[0])
        all_hi.append(base_mesh.bounds[1])
    lo = np.min(np.array(all_lo), axis=0)
    hi = np.max(np.array(all_hi), axis=0)
    center = (lo + hi) / 2.0

    off_screen = args.screenshot is not None
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")

    base_actor = None
    if draw_base and base_mesh is not None:
        base_actor = plotter.add_mesh(pv.wrap(base_mesh), color=(0.75, 0.75, 0.78),
                                      name="base", opacity=1.0)

    n_overlap = int(sum(flags))
    tile_actors = []  # (actor, direction_vector)
    for i, t in enumerate(tile_list):
        direction = np.asarray(t.mesh.centroid) - center
        color = (0.90, 0.10, 0.10) if flags[i] else tuple(tiles_mod.tile_color(t))
        actor = plotter.add_mesh(pv.wrap(t.mesh), color=color, name=f"tile_{i}")
        tile_actors.append((actor, direction))
        if t.number_mesh is not None:
            num_actor = plotter.add_mesh(pv.wrap(t.number_mesh), color=args.number_color,
                                         name=f"num_{i}")
            tile_actors.append((num_actor, direction))

    def set_explode(value):
        for actor, direction in tile_actors:
            actor.SetPosition(*(direction * value))

    title_lines = [
        f"{len(tile_list)} tiles | {n_overlap} overlapping (red)",
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
