import argparse
import numpy as np
from pyvox.models import Vox, Model
from pyvox.parser import VoxParser
from pyvox.writer import VoxWriter
import os

def load_vox_to_bool_array(filepath):
    """
    Loads the first model from a .vox file into a 3D NumPy boolean array.
    Returns: (voxel_data_bool, palette, model_size) or (None, None, None) on error.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return None, None, None
    try:
        vox_container = VoxParser(filepath).parse()
        if not vox_container.models:
            print(f"Warning: No models found in '{filepath}'. Treating as empty model of size (0,0,0).")
            return np.zeros((0, 0, 0), dtype=bool), [(128, 128, 128, 255)], (0, 0, 0)

        model = vox_container.models[0]  # Use the first model
        model_size = model.size

        # Validate model_size format, allowing zero dimensions
        if not (isinstance(model_size, tuple) and len(model_size) == 3 and
                all(isinstance(dim, int) and dim >= 0 for dim in model_size)):
            print(f"Error: Invalid model size format {model_size} in '{filepath}'. Expected 3 non-negative integers.")
            return None, None, None

        voxel_data_bool = np.zeros(model_size, dtype=bool)

        if model.voxels:
            for x, y, z, _color_index in model.voxels:
                if 0 <= x < model_size[0] and \
                   0 <= y < model_size[1] and \
                   0 <= z < model_size[2]:
                    voxel_data_bool[x, y, z] = True
                else:
                    print(f"Warning: Voxel at ({x},{y},{z}) in '{filepath}' is outside its defined model size {model_size}. Skipping.")
        
        palette = vox_container.palette
        if not palette: # Ensure there's a default palette
            palette = [(128, 128, 128, 255)] 

        return voxel_data_bool, palette, model_size
    except Exception as e:
        print(f"Error parsing .vox file '{filepath}': {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def save_bool_array_to_vox(filepath, voxel_data_bool, palette):
    """
    Saves a 3D NumPy boolean array as a .vox file.
    """
    max_dim_val = 255  # Max coordinate value for a single model, so size is max_dim_val + 1 = 256
    original_shape = voxel_data_bool.shape
    
    # Ensure dimensions are positive for saving, even if empty.
    # A model size of (0,N,M) is valid for pyvox if there are no voxels.
    # However, for consistency, let's ensure shape is at least (0,0,0)
    current_shape = tuple(max(0, s) for s in original_shape)

    data_to_save = voxel_data_bool
    
    # Check if any dimension exceeds .vox limit (size 256, so max index 255)
    if any(s > max_dim_val + 1 for s in current_shape):
        print(f"Warning: Model dimensions {current_shape} exceed the .vox single model limit of ({max_dim_val+1},{max_dim_val+1},{max_dim_val+1}).")
        clipped_shape = tuple(min(s, max_dim_val + 1) for s in current_shape)
        print(f"Clipping .vox output to {clipped_shape}.")
        data_to_save = voxel_data_bool[:clipped_shape[0], :clipped_shape[1], :clipped_shape[2]]
        model_size_for_vox_model = clipped_shape
    else:
        model_size_for_vox_model = current_shape

    voxels_list_for_model = []
    
    # Ensure palette is a list and handle potential issues
    current_palette = list(palette) if palette else []

    if not current_palette:
        current_palette.append((128, 128, 128, 255))  # Default gray
    
    if len(current_palette) > 256:
        print(f"Warning: Palette has {len(current_palette)} colors. Clipping to 256 for .vox output.")
        current_palette = current_palette[:256]
    
    # Voxel color index is 1-based, refers to palette[index-1]
    color_index_to_use = 1 
    if not current_palette: # Should not happen due to above default, but as a safeguard
         # This case implies an issue, as .vox needs a palette.
         # Forcing a default if somehow still empty.
        current_palette.append((128,128,128,255))


    dx, dy, dz = data_to_save.shape
    for x_coord in range(dx):
        for y_coord in range(dy):
            for z_coord in range(dz):
                if data_to_save[x_coord, y_coord, z_coord]:
                    voxels_list_for_model.append((x_coord, y_coord, z_coord, color_index_to_use))

    new_model_instance = Model(size=model_size_for_vox_model, voxels=voxels_list_for_model)
    vox_container_to_save = Vox(models=[new_model_instance], palette=current_palette)
    
    if not voxels_list_for_model and np.any(s > 0 for s in model_size_for_vox_model):
        print(f"Note: The model for '{filepath}' is empty after processing, but has non-zero size {model_size_for_vox_model}. Saving an empty model.")
    elif all(s == 0 for s in model_size_for_vox_model): # All dimensions are zero
        print(f"Note: The model for '{filepath}' has zero size {model_size_for_vox_model}. Saving an empty model structure.")
    else:
        print(f"Saving {len(voxels_list_for_model)} voxels to '{filepath}' with model size {model_size_for_vox_model} and palette size {len(current_palette)}.")

    try:
        writer = VoxWriter(filepath, vox_container_to_save)
        writer.write()
    except Exception as e:
        print(f"Error writing .vox file '{filepath}': {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Subtracts voxels of one .vox file from another .vox file.")
    parser.add_argument("target_vox_file", help="Path to the .vox file to subtract voxels from.")
    parser.add_argument("subtraction_vox_file", help="Path to the .vox file whose voxels will be removed from the target.")
    parser.add_argument("output_vox_file", help="Path to save the resulting .vox file.")
    args = parser.parse_args()

    print(f"Loading target file: {args.target_vox_file}")
    target_array, target_palette, target_size = load_vox_to_bool_array(args.target_vox_file)
    if target_array is None:
        print(f"Failed to load target .vox file: {args.target_vox_file}")
        return

    print(f"Loading subtraction file: {args.subtraction_vox_file}")
    subtraction_array, _, subtraction_size = load_vox_to_bool_array(args.subtraction_vox_file)
    if subtraction_array is None:
        print(f"Failed to load subtraction .vox file: {args.subtraction_vox_file}")
        return

    result_array = target_array.copy()
    voxels_subtracted_count = 0

    s_dx, s_dy, s_dz = subtraction_size
    t_dx, t_dy, t_dz = target_size

    print(f"Target model dimensions: {target_size}")
    print(f"Subtraction model dimensions: {subtraction_size}")

    for x in range(s_dx):
        for y in range(s_dy):
            for z in range(s_dz):
                if subtraction_array[x, y, z]:  # If voxel exists in subtraction model
                    # Check if this coordinate is within the bounds of the target model
                    if 0 <= x < t_dx and 0 <= y < t_dy and 0 <= z < t_dz:
                        if result_array[x, y, z]:  # If voxel was set in target model
                            result_array[x, y, z] = False
                            voxels_subtracted_count += 1
    
    print(f"Number of voxels subtracted: {voxels_subtracted_count}")

    print(f"Saving result to: {args.output_vox_file}")
    save_bool_array_to_vox(args.output_vox_file, result_array, target_palette)
    print("Subtraction process complete.")

if __name__ == "__main__":
    main()
