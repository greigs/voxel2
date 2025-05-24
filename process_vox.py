import argparse
import numpy as np
from scipy.ndimage import zoom, binary_erosion, generate_binary_structure
from pyvox.models import Vox, Model
from pyvox.parser import VoxParser
from pyvox.writer import VoxWriter
from stl import mesh
import os
from skimage.measure import marching_cubes # ADDED: For Marching Cubes

CUTOUT_SIZE = 2 # Size of the cube to cut from each surface at the scaled resolution

def scale_voxels(voxel_data_bool, scale_factor):
    """Scales the voxel data by a given factor."""
    # order=0 for nearest-neighbor interpolation, suitable for binary voxel data
    scaled_data = zoom(voxel_data_bool, scale_factor, order=0) # CORRECTED: voxel_data_bool, scale_factor
    return scaled_data.astype(bool)

def erode_voxels(voxel_data_bool, erosion_voxels):
    """Erodes the voxel data using a 3x3x3 structuring element."""
    # generate_binary_structure(rank, connectivity)
    # rank=3 for 3D. connectivity=3 means 26 neighbors (corners included)
    # Alternatively, np.ones((3,3,3), dtype=bool) could be used for a solid cube.
    struct = generate_binary_structure(3, 3) 
    eroded_data = binary_erosion(voxel_data_bool, structure=struct, iterations=erosion_voxels) # CORRECTED: voxel_data_bool, erosion_voxels
    return eroded_data

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
                        # Record the voxels to be cut before actually cutting them
                        region_to_cut = (current_x_slice, y_slice_for_x_face, z_slice_for_x_face)
                        cutouts_only_data[region_to_cut] = np.logical_and(cutouts_only_data[region_to_cut], True) # Keep existing cutouts if overlap
                        cutouts_only_data[region_to_cut] = np.logical_or(cutouts_only_data[region_to_cut], modified_scaled_data[region_to_cut])
                        modified_scaled_data[region_to_cut] = False
                
                # Check and process +X face
                if smx == sm_dx - 1 or not original_voxels_bool[smx + 1, smy, smz]:
                    x_surf_start = (smx + 1) * SF - cutout_dim
                    x_surf_end = (smx + 1) * SF
                    current_x_slice = slice(max(0, x_surf_start), min(sc_dx, x_surf_end))
                    if current_x_slice.start < current_x_slice.stop and y_slice_for_x_face.start < y_slice_for_x_face.stop and z_slice_for_x_face.start < z_slice_for_x_face.stop:
                        region_to_cut = (current_x_slice, y_slice_for_x_face, z_slice_for_x_face)
                        cutouts_only_data[region_to_cut] = np.logical_and(cutouts_only_data[region_to_cut], True)
                        cutouts_only_data[region_to_cut] = np.logical_or(cutouts_only_data[region_to_cut], modified_scaled_data[region_to_cut])
                        modified_scaled_data[region_to_cut] = False

                # Check and process -Y face
                if smy == 0 or not original_voxels_bool[smx, smy - 1, smz]:
                    y_surf_start = smy * SF
                    y_surf_end = y_surf_start + cutout_dim
                    current_y_slice = slice(max(0, y_surf_start), min(sc_dy, y_surf_end))
                    if x_slice_for_y_face.start < x_slice_for_y_face.stop and current_y_slice.start < current_y_slice.stop and z_slice_for_y_face.start < z_slice_for_y_face.stop:
                        region_to_cut = (x_slice_for_y_face, current_y_slice, z_slice_for_y_face)
                        cutouts_only_data[region_to_cut] = np.logical_and(cutouts_only_data[region_to_cut], True)
                        cutouts_only_data[region_to_cut] = np.logical_or(cutouts_only_data[region_to_cut], modified_scaled_data[region_to_cut])
                        modified_scaled_data[region_to_cut] = False

                # Check and process +Y face
                if smy == sm_dy - 1 or not original_voxels_bool[smx, smy + 1, smz]:
                    y_surf_start = (smy + 1) * SF - cutout_dim
                    y_surf_end = (smy + 1) * SF
                    current_y_slice = slice(max(0, y_surf_start), min(sc_dy, y_surf_end))
                    if x_slice_for_y_face.start < x_slice_for_y_face.stop and current_y_slice.start < current_y_slice.stop and z_slice_for_y_face.start < z_slice_for_y_face.stop:
                        region_to_cut = (x_slice_for_y_face, current_y_slice, z_slice_for_y_face)
                        cutouts_only_data[region_to_cut] = np.logical_and(cutouts_only_data[region_to_cut], True)
                        cutouts_only_data[region_to_cut] = np.logical_or(cutouts_only_data[region_to_cut], modified_scaled_data[region_to_cut])
                        modified_scaled_data[region_to_cut] = False
                
                # Check and process -Z face
                if smz == 0 or not original_voxels_bool[smx, smy, smz - 1]:
                    z_surf_start = smz * SF
                    z_surf_end = z_surf_start + cutout_dim
                    current_z_slice = slice(max(0, z_surf_start), min(sc_dz, z_surf_end))
                    if x_slice_for_z_face.start < x_slice_for_z_face.stop and y_slice_for_z_face.start < y_slice_for_z_face.stop and current_z_slice.start < current_z_slice.stop:
                        region_to_cut = (x_slice_for_z_face, y_slice_for_z_face, current_z_slice)
                        cutouts_only_data[region_to_cut] = np.logical_and(cutouts_only_data[region_to_cut], True)
                        cutouts_only_data[region_to_cut] = np.logical_or(cutouts_only_data[region_to_cut], modified_scaled_data[region_to_cut])
                        modified_scaled_data[region_to_cut] = False

                # Check and process +Z face
                if smz == sm_dz - 1 or not original_voxels_bool[smx, smy, smz + 1]:
                    z_surf_start = (smz + 1) * SF - cutout_dim
                    z_surf_end = (smz + 1) * SF
                    current_z_slice = slice(max(0, z_surf_start), min(sc_dz, z_surf_end))
                    if x_slice_for_z_face.start < x_slice_for_z_face.stop and y_slice_for_z_face.start < y_slice_for_z_face.stop and current_z_slice.start < current_z_slice.stop:
                        region_to_cut = (x_slice_for_z_face, y_slice_for_z_face, current_z_slice)
                        cutouts_only_data[region_to_cut] = np.logical_and(cutouts_only_data[region_to_cut], True)
                        cutouts_only_data[region_to_cut] = np.logical_or(cutouts_only_data[region_to_cut], modified_scaled_data[region_to_cut])
                        modified_scaled_data[region_to_cut] = False
                        
    return modified_scaled_data, cutouts_only_data

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

def save_stl_file(filepath, voxel_data_bool):
    """Converts boolean voxel data to an STL mesh using Marching Cubes and saves it."""
    if np.sum(voxel_data_bool) == 0:
        print(f"No voxels to save to STL file '{filepath}'. Skipping.")
        return

    # Convert boolean voxel data to float for marching_cubes. Pad to avoid issues at boundaries.
    # Marching cubes expects values, so True becomes 1.0 and False becomes 0.0.
    # Padding ensures that surfaces on the edge of the volume are closed.
    data_float = np.pad(voxel_data_bool.astype(float), pad_width=1, mode='constant', constant_values=0.0)

    # Apply marching cubes to extract surface
    # level=0.5 is appropriate for binary data (0s and 1s)
    # spacing ensures the model is scaled correctly if voxels aren't unit cubes (default is 1,1,1)
    try:
        verts, faces, normals, values = marching_cubes(data_float, level=0.5, spacing=(1.0, 1.0, 1.0))
    except Exception as e:
        if "No surfaces found" in str(e) or "isolevel" in str(e).lower(): # skimage specific error messages
             print(f"Warning: Marching cubes found no surfaces in '{filepath}' (possibly empty or too sparse after processing). Skipping STL generation.")
             return
        raise # Re-raise other exceptions

    if verts.size == 0 or faces.size == 0:
        print(f"Warning: Marching cubes resulted in an empty mesh for '{filepath}'. Skipping STL generation.")
        return

    # Adjust vertices due to padding: subtract 1 from all coordinates
    verts -= 1

    # Create the STL mesh object
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3): # For each vertex in the face
            stl_mesh.vectors[i][j] = verts[f[j], :]
            
    stl_mesh.save(filepath)
    print(f"Saved STL file to '{filepath}' using Marching Cubes.")


def main():
    parser = argparse.ArgumentParser(description="Scale, erode .vox files, and save as .vox and .stl.")
    parser.add_argument("input_vox_file", help="Path to the input .vox file (e.g., scene.vox).")
    parser.add_argument("--scale_factor", type=float, default=10.0, help="Factor to scale voxels by (default: 10.0).")
    parser.add_argument("--erosion_voxels", type=int, default=1, help="Number of voxel layers to erode after scaling (default: 2).")
    
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

    try:
        print(f"Loading \'{input_path}\'...")
        vox_data_container = VoxParser(input_path).parse()

        if not vox_data_container.models:
            print(f"Input .vox file \'{input_path}\' contains no models. Processing as an empty scene with shape (0,0,0).")
            # Define a shape that indicates emptiness but is still a 3D array for downstream processing
            initial_voxel_data_bool = np.zeros((0,0,0), dtype=bool) 
            original_palette = vox_data_container.palette
            if not original_palette:
                original_palette = [(128,128,128,255)] # Default palette
        else:
            # Assuming we work with the first model in the file
            model = vox_data_container.models[0]
            model_size = model.size # This is (sx, sy, sz)
            
            if not (isinstance(model_size, tuple) and len(model_size) == 3 and all(isinstance(dim, int) and dim >= 0 for dim in model_size)):
                raise ValueError(f"Invalid model size format for model in '{input_path}': {model_size}. Expected 3 non-negative integers.")

            initial_voxel_data_bool = np.zeros(model_size, dtype=bool)
            
            if model.voxels: # Ensure model.voxels is not None
                for x, y, z, color_index in model.voxels:
                    # Voxel coordinates are relative to the model's space.
                    # Ensure they are within the bounds defined by model_size.
                    if 0 <= x < model_size[0] and \
                       0 <= y < model_size[1] and \
                       0 <= z < model_size[2]:
                        initial_voxel_data_bool[x, y, z] = True 
                    else:
                        print(f"Warning: Voxel at ({x},{y},{z}) is outside the defined model size {model_size}. Skipping.")
            
            original_palette = vox_data_container.palette
            if not original_palette:
                original_palette = [(128,128,128,255)] # Default palette
        
        print(f"Original dimensions: {initial_voxel_data_bool.shape}")
        if np.sum(initial_voxel_data_bool) == 0:
            print("Warning: The input .vox model is empty (contains no set voxels).")

        print(f"Scaling by {scale_factor}x...")
        scaled_voxel_data = scale_voxels(initial_voxel_data_bool, scale_factor)
        print(f"Scaled dimensions: {scaled_voxel_data.shape}")

        # Save the scaled data before cutouts or erosion
        if np.any(scaled_voxel_data):
            print(f"Saving scaled .vox file to '{output_scaled_vox_path}'...")
            save_vox_file(output_scaled_vox_path, scaled_voxel_data, original_palette)
            print(f"Saving scaled model as .stl file to '{output_scaled_stl_path}'...")
            save_stl_file(output_scaled_stl_path, scaled_voxel_data)
        else:
            print(f"Skipping save of scaled model as it is empty ('{output_scaled_vox_path}', '{output_scaled_stl_path}').")

        # Apply surface cutouts
        int_sf = int(round(scale_factor))
        data_for_erosion = scaled_voxel_data # Default to scaled_voxel_data
        cutout_voxels_to_save = np.zeros_like(scaled_voxel_data, dtype=bool) # Initialize empty cutouts

        if initial_voxel_data_bool.any() and scaled_voxel_data.any() and int_sf > 0 and CUTOUT_SIZE > 0:
            num_voxels_before_cutout = np.sum(scaled_voxel_data)
            print(f"Applying {CUTOUT_SIZE}x{CUTOUT_SIZE}x{CUTOUT_SIZE} surface cutouts from original smallxel surfaces (using effective scale factor {int_sf})...")
            data_for_erosion, cutout_voxels_to_save = apply_surface_cutouts(
                initial_voxel_data_bool,
                scaled_voxel_data,
                int_sf,
                CUTOUT_SIZE
            )
            num_voxels_after_cutout = np.sum(data_for_erosion)
            print(f"Dimensions after cutouts: {data_for_erosion.shape}") # Shape should not change
            if num_voxels_after_cutout == num_voxels_before_cutout:
                 print("Note: Cutout process did not remove any additional voxels (e.g., no exposed surfaces, cutouts fell outside, or target areas already empty).")
            else:
                 print(f"Note: Cutout process removed {num_voxels_before_cutout - num_voxels_after_cutout} voxels.")
        else:
            if not (initial_voxel_data_bool.any() and scaled_voxel_data.any()):
                print("Skipping surface cutouts as initial or scaled model is empty.")
            elif int_sf <= 0:
                print(f"Skipping surface cutouts as integer scale factor ({int_sf}) is not positive.")
            elif CUTOUT_SIZE <= 0:
                print(f"Skipping surface cutouts as CUTOUT_SIZE ({CUTOUT_SIZE}) is not positive.")
        
        if erosion_amount > 0:
            print(f"Eroding by {erosion_amount} voxel layers...")
            processed_voxel_data = erode_voxels(data_for_erosion, erosion_amount)
            print(f"Dimensions after erosion: {processed_voxel_data.shape}")
        else:
            processed_voxel_data = data_for_erosion # Skip erosion
            print("Skipping erosion step as erosion_amount is 0 or less.")
        
        if np.sum(processed_voxel_data) == 0:
            print("Warning: All voxels were removed after scaling and/or erosion.")

        print(f"Saving processed .vox file to '{output_vox_path}'...")
        save_vox_file(output_vox_path, processed_voxel_data, original_palette)

        print(f"Saving processed model as .stl file to '{output_stl_path}'...")
        save_stl_file(output_stl_path, processed_voxel_data)

        if np.any(cutout_voxels_to_save):
            print(f"Saving cutout voxels to .vox file: '{output_cutouts_vox_path}'...")
            save_vox_file(output_cutouts_vox_path, cutout_voxels_to_save, original_palette)
            print(f"Saving cutout voxels as .stl file: '{output_cutouts_stl_path}'...")
            save_stl_file(output_cutouts_stl_path, cutout_voxels_to_save)
        else:
            print(f"No cutout voxels to save for '{output_cutouts_vox_path}' and '{output_cutouts_stl_path}'.")

        print(f"\\nProcessing complete for \'{input_path}\'.")
        print(f"Output .vox: {output_vox_path}")
        print(f"Output .stl: {output_stl_path}")
        if np.any(scaled_voxel_data):
            print(f"Output Scaled .vox: {output_scaled_vox_path}")
            print(f"Output Scaled .stl: {output_scaled_stl_path}")
        if np.any(cutout_voxels_to_save):
            print(f"Output Cutouts .vox: {output_cutouts_vox_path}")
            print(f"Output Cutouts .stl: {output_cutouts_stl_path}")

    except ImportError as e:
        print(f"Error: A required Python package is missing: {e}")
        print("Please ensure you have installed all dependencies:")
        print("pip install numpy scipy py-vox-io numpy-stl scikit-image") # MODIFIED: Added scikit-image
    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        print(f"Error: Input file '{input_path}' not found during processing.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
