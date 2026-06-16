import argparse
from scipy.ndimage import zoom, binary_erosion, generate_binary_structure
from pyvox.models import Vox, Model
from pyvox.parser import VoxParser
from pyvox.writer import VoxWriter
from stl import mesh
import numpy as np # Ensure numpy is imported
import os # For path operations
import trimesh # For mesh processing
from typing import NamedTuple, Optional, Tuple # For the shared ground-truth helper
from process_stl_parts import process_mesh_object # For advanced mesh processing
import tiles as tiles_mod # Flat mitered per-face tile generator

CUTOUT_SIZE = 1 # Legacy: size of the cube cut from each surface (kept for reference)
DEFAULT_STL_VOXEL_SIZE_MM = 1.25 # Side length of one scaled voxel in the output STL
DEFAULT_GAP_MM = 0.1 # Gap inserted between per-voxel tiles in the gapped difference STL
DEFAULT_PEG_DEPTH_VOXELS = 3 # How far the registration peg sinks into the base (scaled voxels)

# Face directions as (axis, sign), shared with tiles.py.
FACE_DIRECTIONS = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]


def default_peg_size_voxels(scale_factor_int):
    """Default registration peg cross-section: roughly one third of the voxel face."""
    return max(1, int(round(scale_factor_int / 3.0)))


class ExpectedLayers(NamedTuple):
    """Ground-truth voxel layers regenerated from a .vox source.

    All boolean arrays except ``base_cropped`` share the scaled-grid shape so the
    base, skin and full solid can be compared cell-for-cell.
    """
    initial_voxels: np.ndarray            # original (unscaled) solid
    palette: list                          # source palette
    scaled_solid: np.ndarray               # full scaled solid
    cutout_voxels: np.ndarray              # voxels removed as peg holes from the base
    base_cropped: np.ndarray               # eroded core with peg holes, cropped (matches _processed.stl)
    crop_min_coords: Optional[Tuple[int, int, int]]  # min corner of base_cropped within scaled grid
    base_aligned: np.ndarray               # eroded core (with holes) placed back into the scaled grid
    skin: np.ndarray                       # difference = scaled_solid AND NOT base_aligned (debug only)
    scale_factor_int: int                  # integer scale factor (SF)
    voxel_size_mm: float                   # STL voxel size in mm (S)
    gap_mm: float                          # tile gap in mm
    tile_thickness_voxels: int             # T: bevel depth / skin thickness (scaled voxels)
    peg_size_voxels: int                   # p: peg cross-section (scaled voxels)
    peg_depth_voxels: int                  # d: peg depth (scaled voxels)
    voxel_color_code: np.ndarray           # per-(original)-voxel paint color-code (0 empty)
    color_legend: dict                     # color-code -> source palette index


def carve_peg_holes(base_uncropped, initial_voxels, scale_factor_int,
                    tile_thickness_voxels, peg_size_voxels, peg_depth_voxels):
    """Carve a square peg hole into the base under each exposed voxel face.

    Operates in the scaled grid (uncropped). For each exposed face, removes a
    ``peg_size`` x ``peg_size`` x ``peg_depth`` box centered on the face, starting at
    the base surface (``tile_thickness`` voxels below the model surface) and going
    deeper. Returns ``(carved_base, holes_mask)``.
    """
    SF = scale_factor_int
    base = base_uncropped.copy()
    holes = np.zeros_like(base, dtype=bool)
    if SF <= 0 or not np.any(initial_voxels) or base.shape != tuple(np.array(initial_voxels.shape) * SF):
        return base, holes

    dims = initial_voxels.shape
    shape = base.shape
    start = (SF // 2) - (peg_size_voxels // 2)
    T = tile_thickness_voxels
    d = peg_depth_voxels

    for (x, y, z) in np.argwhere(initial_voxels):
        block_min = np.array([x, y, z]) * SF
        for axis, sign in FACE_DIRECTIONS:
            nb = [int(x), int(y), int(z)]
            nb[axis] += sign
            exposed = (nb[axis] < 0 or nb[axis] >= dims[axis]
                       or not initial_voxels[nb[0], nb[1], nb[2]])
            if not exposed:
                continue

            ua = (axis + 1) % 3
            va = (axis + 2) % 3
            lo = [0, 0, 0]
            hi = [shape[0], shape[1], shape[2]]
            lo[ua] = block_min[ua] + start
            hi[ua] = lo[ua] + peg_size_voxels
            lo[va] = block_min[va] + start
            hi[va] = lo[va] + peg_size_voxels
            if sign > 0:
                top = block_min[axis] + SF
                hi[axis] = top - T
                lo[axis] = hi[axis] - d
            else:
                bot = block_min[axis]
                lo[axis] = bot + T
                hi[axis] = lo[axis] + d

            lo = [max(0, int(v)) for v in lo]
            hi = [min(int(s), int(v)) for s, v in zip(shape, hi)]
            if lo[0] < hi[0] and lo[1] < hi[1] and lo[2] < hi[2]:
                sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
                base[sl] = False
                holes[sl] = True

    return base, holes


def load_vox_full(filepath):
    """Loads the first model of a .vox file into ``(bool array, color_index array, palette)``.

    ``color_index`` has the same shape as the bool array; cells hold the source palette
    index (0 where empty)."""
    vox_data_container = VoxParser(filepath).parse()

    if not vox_data_container.models:
        print(f"Input .vox file '{filepath}' contains no models. Treating as empty (0,0,0).")
        palette = vox_data_container.palette or [(128, 128, 128, 255)]
        return np.zeros((0, 0, 0), dtype=bool), np.zeros((0, 0, 0), dtype=np.int32), palette

    model = vox_data_container.models[0]
    model_size = model.size

    if not (isinstance(model_size, tuple) and len(model_size) == 3 and
            all(isinstance(dim, int) and dim >= 0 for dim in model_size)):
        raise ValueError(f"Invalid model size format for model in '{filepath}': {model_size}. "
                         "Expected 3 non-negative integers.")

    voxel_data_bool = np.zeros(model_size, dtype=bool)
    color_index = np.zeros(model_size, dtype=np.int32)
    if model.voxels:
        for x, y, z, c_index in model.voxels:
            if 0 <= x < model_size[0] and 0 <= y < model_size[1] and 0 <= z < model_size[2]:
                voxel_data_bool[x, y, z] = True
                color_index[x, y, z] = int(c_index)
            else:
                print(f"Warning: Voxel at ({x},{y},{z}) is outside the defined model size {model_size}. Skipping.")

    palette = vox_data_container.palette or [(128, 128, 128, 255)]
    return voxel_data_bool, color_index, palette


def load_vox_to_bool_array(filepath):
    """Loads the first model of a .vox file into a (bool array, palette) pair."""
    voxel_data_bool, _color_index, palette = load_vox_full(filepath)
    return voxel_data_bool, palette


def _palette_rgb(palette, index):
    """Best-effort lookup of an (r,g,b) 0-255 tuple for a MagicaVoxel palette index."""
    if not palette:
        return (180, 180, 180)
    # MagicaVoxel color indices are 1-based into a 256-entry palette.
    for cand in (index - 1, index):
        if 0 <= cand < len(palette):
            entry = palette[cand]
            try:
                return (int(entry[0]), int(entry[1]), int(entry[2]))
            except Exception:
                break
    return (180, 180, 180)


def build_color_codes(color_index, solid_mask):
    """Map the distinct palette indices used by solid voxels to compact codes 1..K.

    Returns ``(code_array, legend)`` where ``code_array`` matches the input shape
    (0 = empty) and ``legend`` maps code -> source palette index."""
    code_array = np.zeros(color_index.shape, dtype=np.int32)
    if not np.any(solid_mask):
        return code_array, {}
    used = sorted({int(v) for v in np.unique(color_index[solid_mask])})
    idx_to_code = {idx: i + 1 for i, idx in enumerate(used)}
    legend = {code: idx for idx, code in idx_to_code.items()}
    for idx, code in idx_to_code.items():
        code_array[solid_mask & (color_index == idx)] = code
    return code_array, legend


def compute_expected_layers(vox_path, scale_factor, erosion_voxels,
                            stl_voxel_size_mm=DEFAULT_STL_VOXEL_SIZE_MM,
                            gap_mm=DEFAULT_GAP_MM,
                            peg_size_voxels=None,
                            peg_depth_voxels=DEFAULT_PEG_DEPTH_VOXELS):
    """Regenerates the expected base/solid voxel layers from a .vox file.

    Pipeline: scale -> erode by ``erosion_voxels`` (== tile thickness T) -> carve a
    matching square peg hole under each exposed face -> crop. The flat mitered tiles
    themselves are generated analytically by ``tiles.generate_face_tiles`` from the
    returned layers, so this just needs to produce the base (with holes) plus the
    shared geometry parameters.
    """
    initial_voxel_data_bool, initial_color_index, original_palette = load_vox_full(vox_path)
    voxel_color_code, color_legend = build_color_codes(
        initial_color_index, initial_voxel_data_bool)

    scaled_voxel_data = scale_voxels(initial_voxel_data_bool, scale_factor)
    int_sf = int(round(scale_factor))

    tile_thickness_voxels = int(erosion_voxels)
    if peg_size_voxels is None:
        peg_size_voxels = default_peg_size_voxels(int_sf)
    peg_size_voxels = int(peg_size_voxels)
    peg_depth_voxels = int(peg_depth_voxels)

    # Clamp peg geometry so two perpendicular pegs on the SAME voxel can never collide
    # in the shared interior. A peg starts at depth T (the base surface) and reaches
    # depth T + d from its face; the perpendicular peg's footprint begins at voxel
    # offset ``start = SF//2 - p//2`` from its own face, i.e. ``SF - start - p`` from
    # ours. Requiring T + d <= SF - start - p keeps them apart.
    if int_sf > 0:
        start_v = (int_sf // 2) - (peg_size_voxels // 2)
        if start_v < tile_thickness_voxels:
            peg_size_voxels = max(1, int_sf - 2 * tile_thickness_voxels)
            start_v = (int_sf // 2) - (peg_size_voxels // 2)
        max_peg_depth = (int_sf - start_v - peg_size_voxels) - tile_thickness_voxels
        max_peg_depth = max(0, max_peg_depth)
        if peg_depth_voxels > max_peg_depth:
            print(f"Note: peg_depth_voxels {peg_depth_voxels} would let perpendicular pegs "
                  f"collide; clamping to {max_peg_depth}.")
            peg_depth_voxels = max_peg_depth

    # Base body = eroded scaled solid (faces recede by the tile thickness T).
    if erosion_voxels > 0 and np.any(scaled_voxel_data):
        base_uncropped = erode_voxels(scaled_voxel_data, erosion_voxels)
    else:
        base_uncropped = scaled_voxel_data.copy()

    # Carve matching peg holes under each exposed face.
    cutout_voxels = np.zeros_like(scaled_voxel_data, dtype=bool)
    if np.any(base_uncropped) and int_sf > 0:
        base_uncropped, cutout_voxels = carve_peg_holes(
            base_uncropped, initial_voxel_data_bool, int_sf,
            tile_thickness_voxels, peg_size_voxels, peg_depth_voxels)

    crop_min_coords = None
    if np.any(base_uncropped):
        base_cropped, crop_min_coords = crop_voxel_data(base_uncropped)
    else:
        base_cropped = base_uncropped

    # Place the (possibly cropped) base back into the scaled grid.
    base_aligned = np.zeros_like(scaled_voxel_data, dtype=bool)
    if np.any(base_cropped):
        if crop_min_coords is not None:
            xmin, ymin, zmin = crop_min_coords
            dxc, dyc, dzc = base_cropped.shape
            sx = slice(xmin, xmin + dxc)
            sy = slice(ymin, ymin + dyc)
            sz = slice(zmin, zmin + dzc)
            if (sx.start >= 0 and sx.stop <= scaled_voxel_data.shape[0] and
                    sy.start >= 0 and sy.stop <= scaled_voxel_data.shape[1] and
                    sz.start >= 0 and sz.stop <= scaled_voxel_data.shape[2]):
                base_aligned[sx, sy, sz] = base_cropped
        elif base_cropped.shape == scaled_voxel_data.shape:
            base_aligned = base_cropped.copy()

    skin = np.logical_and(scaled_voxel_data, np.logical_not(base_aligned))

    return ExpectedLayers(
        initial_voxels=initial_voxel_data_bool,
        palette=original_palette,
        scaled_solid=scaled_voxel_data,
        cutout_voxels=cutout_voxels,
        base_cropped=base_cropped,
        crop_min_coords=crop_min_coords,
        base_aligned=base_aligned,
        skin=skin,
        scale_factor_int=int_sf,
        voxel_size_mm=stl_voxel_size_mm,
        gap_mm=gap_mm,
        tile_thickness_voxels=tile_thickness_voxels,
        peg_size_voxels=int(peg_size_voxels),
        peg_depth_voxels=int(peg_depth_voxels),
        voxel_color_code=voxel_color_code,
        color_legend=color_legend,
    )

def scale_voxels(voxel_data_bool, scale_factor):
    """
    Scales the voxel data by ensuring each original voxel becomes a solid
    sf_int x sf_int x sf_int block, where sf_int is the rounded integer scale_factor.
    """
    if not voxel_data_bool.any() or scale_factor <= 0:
        # If input is empty or scale factor is non-positive,
        # return an empty array or handle as appropriate.
        # For simplicity, if scale_factor is non-positive, treat as no scaling or empty.
        # Or, more robustly, calculate scaled shape even if empty.
        if scale_factor <= 0:
            sf_int = 0
        else:
            sf_int = int(round(scale_factor))
            if sf_int <= 0: # Handles cases like scale_factor = 0.1 rounding to 0
                sf_int = 0
        
        orig_dx, orig_dy, orig_dz = voxel_data_bool.shape
        scaled_shape = (orig_dx * sf_int, orig_dy * sf_int, orig_dz * sf_int)
        return np.zeros(scaled_shape, dtype=bool)

    sf_int = int(round(scale_factor))
    if sf_int <= 0: # e.g. scale_factor was < 0.5 and rounded to 0
        orig_dx, orig_dy, orig_dz = voxel_data_bool.shape
        return np.zeros((orig_dx * 0, orig_dy * 0, orig_dz * 0), dtype=bool)


    orig_dx, orig_dy, orig_dz = voxel_data_bool.shape
    
    # Calculate new dimensions
    scaled_dx = orig_dx * sf_int
    scaled_dy = orig_dy * sf_int
    scaled_dz = orig_dz * sf_int

    # Create the new scaled array, initialized to False
    scaled_data_bool = np.zeros((scaled_dx, scaled_dy, scaled_dz), dtype=bool)

    for x_orig in range(orig_dx):
        for y_orig in range(orig_dy):
            for z_orig in range(orig_dz):
                if voxel_data_bool[x_orig, y_orig, z_orig]:
                    # This voxel is True, so create a block in the scaled array
                    x_start_scaled = x_orig * sf_int
                    y_start_scaled = y_orig * sf_int
                    z_start_scaled = z_orig * sf_int
                    
                    scaled_data_bool[
                        x_start_scaled : x_start_scaled + sf_int,
                        y_start_scaled : y_start_scaled + sf_int,
                        z_start_scaled : z_start_scaled + sf_int
                    ] = True
    
    return scaled_data_bool

def erode_voxels(voxel_data_bool, erosion_voxels):
    """Erodes the voxel data using a 3x3x3 structuring element."""
    # generate_binary_structure(rank, connectivity)
    # rank=3 for 3D. connectivity=3 means 26 neighbors (corners included)
    # Alternatively, np.ones((3,3,3), dtype=bool) could be used for a solid cube.
    struct = generate_binary_structure(3, 3) 
    eroded_data = binary_erosion(voxel_data_bool, structure=struct, iterations=erosion_voxels) # CORRECTED: voxel_data_bool, erosion_voxels
    return eroded_data

def crop_voxel_data(voxel_data_bool):
    """Crops the voxel data to the smallest bounding box containing all True voxels.
    Returns the cropped data, and the min coordinates (x_min, y_min, z_min) of the crop.
    Returns an empty array and None for coords if input is empty.
    """
    if not np.any(voxel_data_bool):
        return np.zeros((0, 0, 0), dtype=bool), None

    true_indices = np.argwhere(voxel_data_bool)
    if true_indices.size == 0: # Should be caught by np.any, but as a safeguard
        return np.zeros((0,0,0), dtype=bool), None

    x_min, y_min, z_min = true_indices.min(axis=0)
    x_max, y_max, z_max = true_indices.max(axis=0)

    cropped_data = voxel_data_bool[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    return cropped_data, (x_min, y_min, z_min)

def apply_surface_cutouts(original_voxels_bool, scaled_voxels_bool, scale_factor_int, cutout_dim):
    """
    For each voxel in original_voxels_bool, if it has an exposed surface,
    cut out a (cutout_dim x cutout_dim x cutout_dim) cube from the corresponding
    position on its surface in the scaled_voxels_bool.
    Assumes scale_factor_int is the integer factor used for scaling.
    Returns two arrays: the modified scaled data, and an array of only the cutout voxels.
    """
    if not original_voxels_bool.any() or not scaled_voxels_bool.any() or scale_factor_int <= 0 or cutout_dim <= 0:
        return scaled_voxels_bool.copy(), np.zeros_like(scaled_voxels_bool, dtype=bool)

    modified_scaled_data = scaled_voxels_bool.copy()
    cutouts_only_data = np.zeros_like(scaled_voxels_bool, dtype=bool) # To store only the voxels that are cut out

    sm_dx, sm_dy, sm_dz = original_voxels_bool.shape
    sc_dx, sc_dy, sc_dz = scaled_voxels_bool.shape # Dimensions of the scaled array

    SF = scale_factor_int # Alias for brevity
    # Offset to center the cutout_dim block within an SF-sized segment
    offset = (SF // 2) - (cutout_dim // 2)

    for smx in range(sm_dx):
        for smy in range(sm_dy):
            for smz in range(sm_dz):
                if not original_voxels_bool[smx, smy, smz]:
                    continue

                # Define centered slices for the other two dimensions when a face is on X, Y, or Z plane
                
                # For X-face cutouts, Y and Z are centered on the smallxel's scaled footprint
                y_centered_start_on_x_face = smy * SF + offset
                y_centered_end_on_x_face = y_centered_start_on_x_face + cutout_dim
                z_centered_start_on_x_face = smz * SF + offset
                z_centered_end_on_x_face = z_centered_start_on_x_face + cutout_dim
                
                y_slice_for_x_face = slice(max(0, y_centered_start_on_x_face), min(sc_dy, y_centered_end_on_x_face))
                z_slice_for_x_face = slice(max(0, z_centered_start_on_x_face), min(sc_dz, z_centered_end_on_x_face))

                # For Y-face cutouts, X and Z are centered
                x_centered_start_on_y_face = smx * SF + offset
                x_centered_end_on_y_face = x_centered_start_on_y_face + cutout_dim
                # z_centered_start/end_on_y_face is same as z_centered_start/end_on_x_face
                
                x_slice_for_y_face = slice(max(0, x_centered_start_on_y_face), min(sc_dx, x_centered_end_on_y_face))
                z_slice_for_y_face = z_slice_for_x_face # Same Z centering logic

                # For Z-face cutouts, X and Y are centered
                # x_centered_start/end_on_z_face is same as x_centered_start/end_on_y_face
                # y_centered_start/end_on_z_face is same as y_centered_start/end_on_x_face

                x_slice_for_z_face = x_slice_for_y_face # Same X centering logic
                y_slice_for_z_face = y_slice_for_x_face # Same Y centering logic

                # Check and process -X face of smallxel (smx,smy,smz)
                if smx == 0 or not original_voxels_bool[smx - 1, smy, smz]:
                    x_surf_start = smx * SF
                    x_surf_end = x_surf_start + cutout_dim
                    current_x_slice = slice(max(0, x_surf_start), min(sc_dx, x_surf_end))
                    if current_x_slice.start < current_x_slice.stop and y_slice_for_x_face.start < y_slice_for_x_face.stop and z_slice_for_x_face.start < z_slice_for_x_face.stop:
                        region_to_cut = (current_x_slice, y_slice_for_x_face, z_slice_for_x_face)
                        # Update cutouts_only_data: mark voxels in the current region_to_cut as True
                        # if they are solid in modified_scaled_data (before this cut) OR
                        # if they were already marked as cut from a previous overlapping operation.
                        cutouts_only_data[region_to_cut] = np.logical_or(
                            cutouts_only_data[region_to_cut],
                            modified_scaled_data[region_to_cut]
                        )
                        modified_scaled_data[region_to_cut] = False
                
                # Check and process +X face
                if smx == sm_dx - 1 or not original_voxels_bool[smx + 1, smy, smz]:
                    x_surf_start = (smx + 1) * SF - cutout_dim
                    x_surf_end = (smx + 1) * SF
                    current_x_slice = slice(max(0, x_surf_start), min(sc_dx, x_surf_end))
                    if current_x_slice.start < current_x_slice.stop and y_slice_for_x_face.start < y_slice_for_x_face.stop and z_slice_for_x_face.start < z_slice_for_x_face.stop:
                        region_to_cut = (current_x_slice, y_slice_for_x_face, z_slice_for_x_face)
                        cutouts_only_data[region_to_cut] = np.logical_or(
                            cutouts_only_data[region_to_cut],
                            modified_scaled_data[region_to_cut]
                        )
                        modified_scaled_data[region_to_cut] = False

                # Check and process -Y face
                if smy == 0 or not original_voxels_bool[smx, smy - 1, smz]:
                    y_surf_start = smy * SF
                    y_surf_end = y_surf_start + cutout_dim
                    current_y_slice = slice(max(0, y_surf_start), min(sc_dy, y_surf_end))
                    if x_slice_for_y_face.start < x_slice_for_y_face.stop and current_y_slice.start < current_y_slice.stop and z_slice_for_y_face.start < z_slice_for_y_face.stop:
                        region_to_cut = (x_slice_for_y_face, current_y_slice, z_slice_for_y_face)
                        cutouts_only_data[region_to_cut] = np.logical_or(
                            cutouts_only_data[region_to_cut],
                            modified_scaled_data[region_to_cut]
                        )
                        modified_scaled_data[region_to_cut] = False

                # Check and process +Y face
                if smy == sm_dy - 1 or not original_voxels_bool[smx, smy + 1, smz]:
                    y_surf_start = (smy + 1) * SF - cutout_dim
                    y_surf_end = (smy + 1) * SF
                    current_y_slice = slice(max(0, y_surf_start), min(sc_dy, y_surf_end))
                    if x_slice_for_y_face.start < x_slice_for_y_face.stop and current_y_slice.start < current_y_slice.stop and z_slice_for_y_face.start < z_slice_for_y_face.stop:
                        region_to_cut = (x_slice_for_y_face, current_y_slice, z_slice_for_y_face)
                        cutouts_only_data[region_to_cut] = np.logical_or(
                            cutouts_only_data[region_to_cut],
                            modified_scaled_data[region_to_cut]
                        )
                        modified_scaled_data[region_to_cut] = False
                
                # Check and process -Z face
                if smz == 0 or not original_voxels_bool[smx, smy, smz - 1]:
                    z_surf_start = smz * SF
                    z_surf_end = z_surf_start + cutout_dim
                    current_z_slice = slice(max(0, z_surf_start), min(sc_dz, z_surf_end))
                    if x_slice_for_z_face.start < x_slice_for_z_face.stop and y_slice_for_z_face.start < y_slice_for_z_face.stop and current_z_slice.start < current_z_slice.stop:
                        region_to_cut = (x_slice_for_z_face, y_slice_for_z_face, current_z_slice)
                        cutouts_only_data[region_to_cut] = np.logical_or(
                            cutouts_only_data[region_to_cut],
                            modified_scaled_data[region_to_cut]
                        )
                        modified_scaled_data[region_to_cut] = False

                # Check and process +Z face
                if smz == sm_dz - 1 or not original_voxels_bool[smx, smy, smz + 1]:
                    z_surf_start = (smz + 1) * SF - cutout_dim
                    z_surf_end = (smz + 1) * SF
                    current_z_slice = slice(max(0, z_surf_start), min(sc_dz, z_surf_end))
                    if x_slice_for_z_face.start < x_slice_for_z_face.stop and y_slice_for_z_face.start < y_slice_for_z_face.stop and current_z_slice.start < current_z_slice.stop:
                        region_to_cut = (x_slice_for_z_face, y_slice_for_z_face, current_z_slice)
                        cutouts_only_data[region_to_cut] = np.logical_or(
                            cutouts_only_data[region_to_cut],
                            modified_scaled_data[region_to_cut]
                        )
                        modified_scaled_data[region_to_cut] = False
                        
    return modified_scaled_data, cutouts_only_data

def generate_triangles_for_voxel_block(voxel_data_bool, output_voxel_size_mm, global_offset_mm):
    """Generates STL triangles for a block of voxels with a global offset."""
    if not np.any(voxel_data_bool):
        return []

    dx, dy, dz = voxel_data_bool.shape
    s = float(output_voxel_size_mm)
    offset_x, offset_y, offset_z = global_offset_mm
    
    block_triangles = []

    for x_local in range(dx):
        for y_local in range(dy):
            for z_local in range(dz):
                if voxel_data_bool[x_local, y_local, z_local]:
                    base_vx = offset_x + float(x_local) * s
                    base_vy = offset_y + float(y_local) * s
                    base_vz = offset_z + float(z_local) * s
                    
                    v = [
                        (base_vx,    base_vy,    base_vz),                     # 0
                        (base_vx + s,base_vy,    base_vz),                 # 1
                        (base_vx + s,base_vy + s,base_vz),             # 2
                        (base_vx,    base_vy + s,base_vz),                 # 3
                        (base_vx,    base_vy,    base_vz + s),             # 4
                        (base_vx + s,base_vy,    base_vz + s),             # 5
                        (base_vx + s,base_vy + s,base_vz + s),         # 6
                        (base_vx,    base_vy + s,base_vz + s)              # 7
                    ]

                    # -X face
                    if x_local == 0 or not voxel_data_bool[x_local - 1, y_local, z_local]:
                        block_triangles.append([v[0], v[4], v[7]])
                        block_triangles.append([v[0], v[7], v[3]])
                    # +X face
                    if x_local == dx - 1 or not voxel_data_bool[x_local + 1, y_local, z_local]:
                        block_triangles.append([v[1], v[2], v[6]])
                        block_triangles.append([v[1], v[6], v[5]])
                    # -Y face
                    if y_local == 0 or not voxel_data_bool[x_local, y_local - 1, z_local]:
                        block_triangles.append([v[0], v[1], v[5]])
                        block_triangles.append([v[0], v[5], v[4]])
                    # +Y face
                    if y_local == dy - 1 or not voxel_data_bool[x_local, y_local + 1, z_local]:
                        block_triangles.append([v[3], v[7], v[6]])
                        block_triangles.append([v[3], v[6], v[2]])
                    # -Z face
                    if z_local == 0 or not voxel_data_bool[x_local, y_local, z_local - 1]:
                        block_triangles.append([v[0], v[2], v[1]])
                        block_triangles.append([v[0], v[3], v[2]])
                    # +Z face
                    if z_local == dz - 1 or not voxel_data_bool[x_local, y_local, z_local + 1]:
                        block_triangles.append([v[4], v[5], v[6]])
                        block_triangles.append([v[4], v[6], v[7]])
    return block_triangles

def save_gapped_difference_stl(filepath, 
                               original_smallxels_bool, 
                               overall_difference_voxels_scaled, 
                               scale_factor_int, 
                               stl_voxel_size_mm, 
                               gap_mm):
    """Saves an STL of objects from difference_voxels, gapped by original grid,
    with each object processed by process_mesh_object."""
    if not np.any(overall_difference_voxels_scaled) or not np.any(original_smallxels_bool):
        print(f"No difference voxels or original voxels to process for '{filepath}'. Skipping.")
        return False # Indicate no file saved

    sm_dx, sm_dy, sm_dz = original_smallxels_bool.shape
    SF = scale_factor_int
    S_vmm = stl_voxel_size_mm

    all_final_triangles = []
    total_input_blocks = 0 

    for smx in range(sm_dx):
        for smy in range(sm_dy):
            for smz in range(sm_dz):
                if not original_smallxels_bool[smx, smy, smz]:
                    continue

                sub_diff_block = overall_difference_voxels_scaled[
                    smx*SF : (smx+1)*SF,
                    smy*SF : (smy+1)*SF,
                    smz*SF : (smz+1)*SF
                ]

                if not np.any(sub_diff_block):
                    continue
                
                total_input_blocks += 1

                # 1. Generate initial triangles for the sub_diff_block (at origin, correct scale)
                initial_block_triangles = generate_triangles_for_voxel_block(
                    sub_diff_block,
                    S_vmm,
                    (0.0, 0.0, 0.0) # Generate at local origin
                )

                if not initial_block_triangles:
                    # print(f"Block at sm({smx},{smy},{smz}) generated no initial triangles. Skipping.") # Optional: verbose logging
                    continue

                # 2. Convert triangles to Trimesh object
                all_block_vertices_list = []
                all_block_faces_list = []
                vertex_to_index_map = {}
                next_vertex_idx = 0

                for triangle in initial_block_triangles:
                    current_face_indices = []
                    for vertex_coords in triangle:
                        vertex_tuple = tuple(vertex_coords) # Ensure hashable for dict key
                        if vertex_tuple not in vertex_to_index_map:
                            vertex_to_index_map[vertex_tuple] = next_vertex_idx
                            all_block_vertices_list.append(list(vertex_coords))
                            next_vertex_idx += 1
                        current_face_indices.append(vertex_to_index_map[vertex_tuple])
                    all_block_faces_list.append(current_face_indices)
                
                if not all_block_vertices_list or not all_block_faces_list:
                    # print(f"Block at sm({smx},{smy},{smz}) resulted in no vertices/faces for Trimesh. Skipping.") # Optional
                    continue

                try:
                    initial_trimesh_obj = trimesh.Trimesh(
                        vertices=np.array(all_block_vertices_list),
                        faces=np.array(all_block_faces_list)
                    )
                    # Check if the created mesh is valid enough for processing
                    if initial_trimesh_obj.is_empty or initial_trimesh_obj.volume < 1e-9:
                        # print(f"Block at sm({smx},{smy},{smz}) created an empty or non-volumetric Trimesh object. Skipping processing for this block.") # Optional
                        continue
                except Exception as e:
                    print(f"Error creating Trimesh object for block at sm({smx},{smy},{smz}): {e}. Skipping.")
                    continue
                
                # 3. Process the Trimesh object
                part_base_name = f"{os.path.splitext(os.path.basename(filepath))[0]}_sm{smx}_{smy}_{smz}"
                
                processed_meshes, cut_type = process_mesh_object(
                    initial_trimesh_obj,
                    part_base_name,
                    run_three_wall_flag=False # Allow auto-detection of cut type
                )

                # 4. Add triangles from processed meshes to all_final_triangles, applying global offset
                if processed_meshes:
                    block_origin_x_mm = smx * (SF * S_vmm + gap_mm)
                    block_origin_y_mm = smy * (SF * S_vmm + gap_mm)
                    block_origin_z_mm = smz * (SF * S_vmm + gap_mm)
                    current_block_global_offset_mm = np.array([block_origin_x_mm, block_origin_y_mm, block_origin_z_mm])

                    for proc_mesh in processed_meshes:
                        if proc_mesh and not proc_mesh.is_empty:
                            # Ensure proc_mesh.vertices and proc_mesh.faces are valid
                            if proc_mesh.vertices is not None and len(proc_mesh.vertices) > 0 and \
                               proc_mesh.faces is not None and len(proc_mesh.faces) > 0:
                                
                                transformed_vertices = proc_mesh.vertices + current_block_global_offset_mm
                                for face_indices_in_proc_mesh in proc_mesh.faces:
                                    all_final_triangles.append(transformed_vertices[face_indices_in_proc_mesh].tolist())
                            # else: # Optional: verbose logging for invalid processed parts
                                # print(f"Processed mesh part from {part_base_name} (cut type: {cut_type}) is invalid (no vertices/faces). Skipping this part.")
                # else: # Optional: verbose logging for blocks that yield no processed meshes
                    # print(f"Processing of block {part_base_name} (sm({smx},{smy},{smz})) with process_mesh_object (cut type: {cut_type}) yielded no meshes.")


    if not all_final_triangles:
        print(f"No actual mesh objects to save in '{filepath}' after processing {total_input_blocks} input blocks. Skipping.")
        return False

    num_triangles = len(all_final_triangles)
    final_mesh_obj = mesh.Mesh(np.zeros(num_triangles, dtype=mesh.Mesh.dtype))
    for i, triangle_vertices in enumerate(all_final_triangles):
        final_mesh_obj.vectors[i] = triangle_vertices
    
    final_mesh_obj.save(filepath)
    print(f"Saved gapped difference STL to '{filepath}' from {total_input_blocks} input blocks, resulting in {num_triangles} triangles. Gap: {gap_mm}mm.")
    return True

def save_vox_file(filepath, voxel_data_bool, original_palette):
    """Saves the boolean voxel data to a .vox file, clipping if dimensions exceed 255."""
    
    max_dim = 255 # Max coordinate value for a single model, so size is max_dim + 1 = 256
    original_shape = voxel_data_bool.shape
    # Clipped shape ensures dimensions are at most 256 (indices 0-255)
    clipped_shape = tuple(min(s, max_dim + 1) for s in original_shape)
    
    data_to_save = voxel_data_bool
    
    if original_shape != clipped_shape:
        print(f"Warning: Model dimensions {original_shape} exceed the .vox single model limit of ({max_dim+1},{max_dim+1},{max_dim+1}).")
        print(f"Clipping .vox output to {clipped_shape}.")
        # Slice the array to fit within the max dimensions
        data_to_save = voxel_data_bool[:clipped_shape[0], :clipped_shape[1], :clipped_shape[2]]

    voxels_list_for_model = []
    current_palette = list(original_palette) 

    if not current_palette:
        current_palette.append((128, 128, 128, 255)) 
    
    if len(current_palette) > 256:
        print(f"Warning: Original palette has {len(current_palette)} colors. Clipping to 256 for .vox output.")
        current_palette = current_palette[:256]
    elif not current_palette: 
        current_palette.append((128, 128, 128, 255))

    color_index_to_use = 1 # 1-based index for the first color in the palette

    model_size_tuple = data_to_save.shape 

    for x_coord in range(model_size_tuple[0]):
        for y_coord in range(model_size_tuple[1]):
            for z_coord in range(model_size_tuple[2]):
                if data_to_save[x_coord, y_coord, z_coord]:
                    # Ensure coordinates are within byte range (0-255)
                    # This should be guaranteed by the clipping of data_to_save
                    voxels_list_for_model.append((x_coord, y_coord, z_coord, color_index_to_use))

    new_model_instance = Model(size=model_size_tuple, voxels=voxels_list_for_model)
    vox_container_to_save = Vox(models=[new_model_instance], palette=current_palette)
    
    if not voxels_list_for_model and np.any(model_size_tuple):
        print(f"Note: The model for '{filepath}\' is empty after processing/clipping, but has non-zero size {model_size_tuple}. Saving an empty model.")
    elif not np.any(model_size_tuple):
        print(f"Note: The model for '{filepath}\' has zero size {model_size_tuple}. Saving an empty model.")
    else:
        print(f"Saving {len(voxels_list_for_model)} voxels to '{filepath}\' with model size {model_size_tuple} and palette size {len(current_palette)}.")

    writer = VoxWriter(filepath, vox_container_to_save)
    writer.write()

def save_stl_file(filepath, voxel_data_bool, output_voxel_size_mm=1.0):
    """Converts boolean voxel data to an STL mesh and saves it.
    Each voxel in voxel_data_bool is represented as a cube of side length output_voxel_size_mm.
    Only external faces are included to create a hollow mesh.
    """
    if not np.any(voxel_data_bool):
        print(f"No voxels to save to STL file '{filepath}'. Skipping.")
        return

    dx, dy, dz = voxel_data_bool.shape
    
    # List to store all the triangles for the external faces
    all_triangles = []

    s = float(output_voxel_size_mm) # size of one voxel in mm

    for x in range(dx):
        for y in range(dy):
            for z in range(dz):
                if voxel_data_bool[x, y, z]:
                    # This is a solid voxel, check its 6 faces
                    # Calculate base coordinates for this voxel, scaled by output_voxel_size_mm
                    base_x, base_y, base_z = float(x) * s, float(y) * s, float(z) * s
                    
                    # Define the 8 vertices of the cube for voxel (x, y, z)
                    # scaled by output_voxel_size_mm
                    v = [
                        (base_x, base_y, base_z),                     # 0: bottom-left-front
                        (base_x + s, base_y, base_z),                 # 1: bottom-right-front
                        (base_x + s, base_y + s, base_z),             # 2: bottom-right-back
                        (base_x, base_y + s, base_z),                 # 3: bottom-left-back
                        (base_x, base_y, base_z + s),                 # 4: top-left-front
                        (base_x + s, base_y, base_z + s),             # 5: top-right-front
                        (base_x + s, base_y + s, base_z + s),         # 6: top-right-back
                        (base_x, base_y + s, base_z + s)              # 7: top-left-back
                    ]

                    # Check -X face (left)
                    if x == 0 or not voxel_data_bool[x - 1, y, z]:
                        all_triangles.append([v[0], v[4], v[7]])
                        all_triangles.append([v[0], v[7], v[3]])
                    
                    # Check +X face (right)
                    if x == dx - 1 or not voxel_data_bool[x + 1, y, z]:
                        all_triangles.append([v[1], v[2], v[6]]) # Corrected winding
                        all_triangles.append([v[1], v[6], v[5]]) # Corrected winding

                    # Check -Y face (front)
                    if y == 0 or not voxel_data_bool[x, y - 1, z]:
                        all_triangles.append([v[0], v[1], v[5]])
                        all_triangles.append([v[0], v[5], v[4]])

                    # Check +Y face (back)
                    if y == dy - 1 or not voxel_data_bool[x, y + 1, z]:
                        all_triangles.append([v[3], v[7], v[6]]) # Corrected winding as per user request
                        all_triangles.append([v[3], v[6], v[2]]) # Corrected winding as per user request

                    # Check -Z face (bottom)
                    if z == 0 or not voxel_data_bool[x, y, z - 1]:
                        all_triangles.append([v[0], v[2], v[1]])
                        all_triangles.append([v[0], v[3], v[2]])
                        
                    # Check +Z face (top)
                    if z == dz - 1 or not voxel_data_bool[x, y, z + 1]:
                        all_triangles.append([v[4], v[5], v[6]])
                        all_triangles.append([v[4], v[6], v[7]])

    if not all_triangles:
        print(f"No external faces found to save to STL file '{filepath}'. Skipping.")
        return

    # Create the mesh object
    num_triangles = len(all_triangles)
    stl_mesh_obj = mesh.Mesh(np.zeros(num_triangles, dtype=mesh.Mesh.dtype))
    for i, triangle_vertices in enumerate(all_triangles):
        stl_mesh_obj.vectors[i] = triangle_vertices
    
    stl_mesh_obj.save(filepath)
    num_voxels = np.sum(voxel_data_bool)
    dim_x_mm = dx * s
    dim_y_mm = dy * s
    dim_z_mm = dz * s
    print(f"Saved hollow STL file to '{filepath}' with {num_voxels} voxels ({num_triangles} triangles representing external faces). Each voxel is {s}mm sided. STL dimensions: {dim_x_mm:.2f}mm x {dim_y_mm:.2f}mm x {dim_z_mm:.2f}mm.")


def _dir_name(axis, sign):
    return {0: "x", 1: "y", 2: "z"}[axis] + ("p" if sign > 0 else "n")


def save_tiles_stl(filepath, tile_list, layers, gap_mm, tiles_dir=None,
                   numbers_path=None, manifest_path=None):
    """Write the flat mitered tiles to a combined gapped STL.

    Each tile is offset by ``sm * gap_mm`` so per-voxel groups are spread apart for the
    exploded printable layout (matching what verify/view un-gap). If tiles carry number
    inlays, the inlays are written (aligned, same gap) to ``numbers_path`` for a second
    AMS filament, and a CSV ``manifest_path`` maps each number to its location/color.
    Per-tile STLs (body + number) are dumped to ``tiles_dir`` when given.
    """
    if not tile_list:
        print(f"No tiles to write for '{filepath}'. Skipping.")
        return False

    placed_bodies = []
    placed_numbers = []
    manifest_rows = []
    have_numbers = any(t.number_mesh is not None for t in tile_list)

    if tiles_dir and not os.path.exists(tiles_dir):
        os.makedirs(tiles_dir)

    for t in tile_list:
        offset = np.asarray(t.voxel, dtype=float) * gap_mm
        body = t.mesh.copy()
        body.apply_translation(offset)
        placed_bodies.append(body)

        num = None
        if t.number_mesh is not None:
            num = t.number_mesh.copy()
            num.apply_translation(offset)
            placed_numbers.append(num)

        manifest_rows.append((t.number, t.voxel[0], t.voxel[1], t.voxel[2],
                              _dir_name(t.axis, t.sign), t.color_code))

        if tiles_dir:
            stem = f"tile_{t.number:04d}_{t.voxel[0]}_{t.voxel[1]}_{t.voxel[2]}_{_dir_name(t.axis, t.sign)}"
            t.mesh.export(os.path.join(tiles_dir, stem + "_body.stl"))
            if t.number_mesh is not None:
                t.number_mesh.export(os.path.join(tiles_dir, stem + "_number.stl"))

    trimesh.util.concatenate(placed_bodies).export(filepath)
    print(f"Saved {len(tile_list)} tile bodies to '{filepath}'. Gap: {gap_mm}mm." +
          (f" Per-tile STLs in '{tiles_dir}'." if tiles_dir else ""))

    if have_numbers and numbers_path and placed_numbers:
        trimesh.util.concatenate(placed_numbers).export(numbers_path)
        print(f"Saved {len(placed_numbers)} number inlays to '{numbers_path}' "
              "(print in the AMS text color).")

    if manifest_path:
        import csv
        with open(manifest_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["number", "voxel_x", "voxel_y", "voxel_z", "face", "color_code"])
            w.writerows(manifest_rows)
        print(f"Wrote tile manifest to '{manifest_path}'.")

    return True


def save_color_legend(layers, json_path, png_path=None):
    """Write the color-code legend (code -> source color) as JSON and an optional PNG.

    The PNG is a swatch chart so you can match each color-code to a real paint."""
    legend = layers.color_legend or {}
    palette = layers.palette
    rows = []
    for code in sorted(legend):
        rgb = _palette_rgb(palette, legend[code])
        rows.append({"color_code": int(code), "palette_index": int(legend[code]),
                     "rgb": list(rgb), "hex": "#%02X%02X%02X" % rgb})

    import json as _json
    with open(json_path, "w") as f:
        _json.dump({"colors": rows}, f, indent=2)
    print(f"Wrote color legend to '{json_path}' ({len(rows)} colors).")

    if png_path and rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n = len(rows)
            fig, ax = plt.subplots(figsize=(3.2, 0.5 * n + 0.6))
            for i, r in enumerate(rows):
                y = n - 1 - i
                ax.add_patch(plt.Rectangle((0, y), 1, 0.85,
                                           color=tuple(c / 255.0 for c in r["rgb"])))
                ax.text(1.2, y + 0.42, f"code {r['color_code']}  {r['hex']}",
                        va="center", fontsize=10)
            ax.set_xlim(0, 4)
            ax.set_ylim(0, n)
            ax.axis("off")
            ax.set_title("Tile paint color codes")
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"Wrote color legend image to '{png_path}'.")
        except Exception as e:
            print(f"Could not write legend PNG ({e}).")


def main():
    parser = argparse.ArgumentParser(description="Scale, erode .vox files, and save as .vox and .stl.")
    parser.add_argument("input_vox_file", help="Path to the input .vox file (e.g., scene.vox).")
    parser.add_argument("--scale_factor", type=float, default=10.0, help="Factor to scale voxels by (default: 10.0).")
    parser.add_argument("--erosion_voxels", type=int, default=1, help="Voxel layers to erode = tile thickness T (default: 1).")
    parser.add_argument("--peg_size_voxels", type=int, default=None, help="Registration peg cross-section in scaled voxels (default: ~SF/3).")
    parser.add_argument("--peg_depth_voxels", type=int, default=DEFAULT_PEG_DEPTH_VOXELS, help=f"Registration peg depth in scaled voxels (default: {DEFAULT_PEG_DEPTH_VOXELS}).")
    parser.add_argument("--tiles_dir", default=None, help="If set, also write one STL per tile (body + number) into this directory.")
    parser.add_argument("--no_numbers", action="store_true", help="Disable the flat two-color number inlays on tiles.")
    parser.add_argument("--emboss_depth_mm", type=float, default=tiles_mod.DEFAULT_EMBOSS_DEPTH_MM, help=f"Depth of the flush number inlay layer in mm (default: {tiles_mod.DEFAULT_EMBOSS_DEPTH_MM}).")
    parser.add_argument("--peg_clearance_mm", type=float, default=tiles_mod.DEFAULT_PEG_CLEARANCE_MM, help=f"Peg-to-hole lateral clearance per side, mm (default: {tiles_mod.DEFAULT_PEG_CLEARANCE_MM}).")
    parser.add_argument("--peg_depth_clearance_mm", type=float, default=tiles_mod.DEFAULT_PEG_DEPTH_CLEARANCE_MM, help=f"How much shorter the peg is than its hole, mm (default: {tiles_mod.DEFAULT_PEG_DEPTH_CLEARANCE_MM}).")

    args = parser.parse_args()

    input_path = args.input_vox_file
    scale_factor = args.scale_factor
    erosion_amount = args.erosion_voxels

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    # Determine output file paths
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path)
    if not output_dir: # If input_path is just a filename
        output_dir = "." 

    output_vox_path = os.path.join(output_dir, f"{base_name}_processed.vox")
    output_stl_path = os.path.join(output_dir, f"{base_name}_processed.stl")
    output_cutouts_vox_path = os.path.join(output_dir, f"{base_name}_cutouts.vox")
    output_cutouts_stl_path = os.path.join(output_dir, f"{base_name}_cutouts.stl")
    output_scaled_vox_path = os.path.join(output_dir, f"{base_name}_scaled.vox") # New path for scaled .vox
    output_scaled_stl_path = os.path.join(output_dir, f"{base_name}_scaled.stl") # New path for scaled .stl
    output_gapped_diff_stl_path = os.path.join(output_dir, f"{base_name}_gapped_diff.stl") # New path
    output_numbers_stl_path = os.path.join(output_dir, f"{base_name}_numbers.stl")
    output_manifest_path = os.path.join(output_dir, f"{base_name}_tiles_manifest.csv")
    output_legend_json_path = os.path.join(output_dir, f"{base_name}_color_legend.json")
    output_legend_png_path = os.path.join(output_dir, f"{base_name}_color_legend.png")

    STL_VOXEL_SIZE_MM = DEFAULT_STL_VOXEL_SIZE_MM
    GAP_MM = DEFAULT_GAP_MM # Gap for the new gapped difference STL

    try:
        print(f"Loading '{input_path}'...")
        # Regenerate all voxel layers via the shared helper so the generator and
        # the verifier use identical math.
        layers = compute_expected_layers(
            input_path, scale_factor, erosion_amount,
            stl_voxel_size_mm=STL_VOXEL_SIZE_MM, gap_mm=GAP_MM,
            peg_size_voxels=args.peg_size_voxels, peg_depth_voxels=args.peg_depth_voxels)

        initial_voxel_data_bool = layers.initial_voxels
        original_palette = layers.palette
        scaled_voxel_data = layers.scaled_solid
        cutout_voxels_to_save = layers.cutout_voxels
        processed_voxel_data = layers.base_cropped
        crop_min_coords = layers.crop_min_coords
        difference_voxels = layers.skin
        int_sf = layers.scale_factor_int

        print(f"Original dimensions: {initial_voxel_data_bool.shape}")
        if np.sum(initial_voxel_data_bool) == 0:
            print("Warning: The input .vox model is empty (contains no set voxels).")
        print(f"Scaled dimensions: {scaled_voxel_data.shape}")

        # Save the scaled data before cutouts or erosion
        if np.any(scaled_voxel_data):
            print(f"Saving scaled .vox file to '{output_scaled_vox_path}'...")
            save_vox_file(output_scaled_vox_path, scaled_voxel_data, original_palette)
            print(f"Saving scaled model as .stl file to '{output_scaled_stl_path}'...")
            save_stl_file(output_scaled_stl_path, scaled_voxel_data, output_voxel_size_mm=STL_VOXEL_SIZE_MM)
        else:
            print(f"Skipping save of scaled model as it is empty ('{output_scaled_vox_path}', '{output_scaled_stl_path}').")

        if np.sum(processed_voxel_data) == 0:
            print("Warning: All voxels were removed after scaling and/or erosion.")

        print(f"Saving processed .vox file to '{output_vox_path}'...")
        save_vox_file(output_vox_path, processed_voxel_data, original_palette)

        print(f"Saving processed model as .stl file to '{output_stl_path}'...")
        save_stl_file(output_stl_path, processed_voxel_data, output_voxel_size_mm=STL_VOXEL_SIZE_MM)

        # Generate the flat mitered per-face skin tiles and write the gapped STL.
        gapped_stl_saved = False
        with_labels = not args.no_numbers
        if np.any(scaled_voxel_data):
            print(f"Generating flat mitered per-face tiles{' with number inlays' if with_labels else ''}...")
            tile_list = tiles_mod.generate_face_tiles(
                layers, with_labels=with_labels, emboss_depth_mm=args.emboss_depth_mm,
                peg_clearance_mm=args.peg_clearance_mm,
                peg_depth_clearance_mm=args.peg_depth_clearance_mm)
            print(f"  {len(tile_list)} tiles generated.")
            if tile_list:
                print(f"Saving tiles STL to '{output_gapped_diff_stl_path}'...")
                gapped_stl_saved = save_tiles_stl(
                    output_gapped_diff_stl_path, tile_list, layers, GAP_MM,
                    tiles_dir=args.tiles_dir,
                    numbers_path=output_numbers_stl_path if with_labels else None,
                    manifest_path=output_manifest_path)
                if with_labels:
                    save_color_legend(layers, output_legend_json_path, output_legend_png_path)
            else:
                print("No exposed faces; no tiles to save.")
        else:
            print("Scaled voxel data is empty. Skipping tiles STL.")

        print(f"\nProcessing complete for '{input_path}'.")
        print(f"Output .vox: {output_vox_path}")
        print(f"Output .stl: {output_stl_path}")
        if np.any(scaled_voxel_data):
            print(f"Output Scaled .vox: {output_scaled_vox_path}")
            print(f"Output Scaled .stl: {output_scaled_stl_path}")
            
        if np.any(cutout_voxels_to_save):
            print(f"Output Cutouts .vox: {output_cutouts_vox_path}")
            print(f"Output Cutouts .stl: {output_cutouts_stl_path}")
        
        if 'gapped_stl_saved' in locals() and gapped_stl_saved:
             print(f"Output Gapped Difference .stl: {output_gapped_diff_stl_path}")
            

    except ImportError as e:
        print(f"Error: A required Python package is missing: {e}")
        print("Please ensure you have installed all dependencies:")
        print("pip install numpy scipy py-vox-io numpy-stl")
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found during processing.")    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    main()
