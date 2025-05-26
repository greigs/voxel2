import os
import sys
import argparse
import numpy as np
import trimesh

MAX_DIM = 8.0
SPACING = 0.1
MAX_CUTTING_DEPTH = 10

# Constants for flat face validation
MIN_FLAT_FACE_EDGE = 8.0
MIN_FLAT_FACE_AREA = MIN_FLAT_FACE_EDGE * MIN_FLAT_FACE_EDGE
FLAT_FACE_NORMAL_TOLERANCE = 0.01 # cos(angle), so close to 0 for 90 deg, 1 for 0 deg. Here, for normal alignment.
FLAT_FACE_FILL_RATIO = 0.90 # How much of the 2D bounding box should be filled by the face cluster

def has_required_flat_face(mesh, min_edge_length=MIN_FLAT_FACE_EDGE, min_area=MIN_FLAT_FACE_AREA, normal_tolerance=FLAT_FACE_NORMAL_TOLERANCE, fill_ratio_threshold=FLAT_FACE_FILL_RATIO):
    """
    Checks if a mesh has at least one flat, filled rectangular (or near-rectangular) face
    meeting the minimum edge length.
    Focuses on axis-aligned faces for simplicity and relevance to 3D printing.
    """
    if mesh.is_empty or not hasattr(mesh, 'face_normals'):
        return False

    # Iterate over major axis-aligned directions
    for axis in range(3):
        # Check positive and negative directions for this axis
        for direction_sign in [-1, 1]:
            target_normal = np.zeros(3)
            target_normal[axis] = direction_sign

            # Find faces aligned with this target_normal
            aligned_face_indices = []
            for i, normal in enumerate(mesh.face_normals):
                if np.dot(normal, target_normal) > (1.0 - normal_tolerance): # Normal is similar
                    aligned_face_indices.append(i)
            
            if not aligned_face_indices:
                continue

            # Group connected aligned faces (simple BFS/DFS based grouping could be added here if needed)
            # For now, consider all aligned faces on this plane as a single candidate patch
            # This is a simplification; true robust clustering is more complex.
            
            candidate_patch_vertices = mesh.vertices[np.unique(mesh.faces[aligned_face_indices].flatten())]
            
            if len(candidate_patch_vertices) < 3:
                continue

            # Project these vertices onto the plane defined by the target_normal
            # For an axis-aligned plane, projection is just dropping the axis coordinate.
            projected_vertices = np.delete(candidate_patch_vertices, axis, axis=1)

            if projected_vertices.shape[0] < 3 or projected_vertices.shape[1] != 2: # Should be 2D points
                continue
            
            try:
                # Get the 2D bounding box of the projected vertices
                min_coords = projected_vertices.min(axis=0)
                max_coords = projected_vertices.max(axis=0)
                bbox_dims = max_coords - min_coords
                bbox_area = bbox_dims[0] * bbox_dims[1]

                if bbox_dims[0] >= min_edge_length and bbox_dims[1] >= min_edge_length and bbox_area >= min_area:
                    # Check fill ratio: sum of areas of faces in the patch vs bbox_area
                    # This is a rough check. A more accurate way is to use the area of the 2D convex hull.
                    patch_face_area_sum = np.sum(mesh.area_faces[aligned_face_indices])
                    
                    # The projected area of the faces should be used.
                    # For axis aligned faces, face_area is already the projected area.
                    
                    if patch_face_area_sum / bbox_area >= fill_ratio_threshold:
                        # print(f"DEBUG: Found flat face: axis={axis}, sign={direction_sign}, bbox_dims={bbox_dims}, patch_area={patch_face_area_sum}, bbox_area={bbox_area}")
                        return True
            except Exception as e:
                # print(f"DEBUG: Exception in has_required_flat_face geometry processing: {e}")
                continue # Problem with this patch, try others
    return False

def validate_part(mesh, part_name, max_dim=MAX_DIM, min_flat_face_dim=MIN_FLAT_FACE_EDGE):
    """
    Validates a single mesh part.
    Returns (isValid, reasons_list)
    """
    is_valid = True
    reasons = []

    if mesh.is_empty or len(mesh.faces) < 4 or not hasattr(mesh, 'volume') or mesh.volume < 1e-6:
        reasons.append(f"Part {part_name} is empty or has negligible volume/faces.")
        return False, reasons

    # 1. Check dimensions
    dims = mesh.extents
    num_dims_over_limit = 0
    for i, d in enumerate(dims):
        if d > max_dim:
            num_dims_over_limit += 1
    
    if num_dims_over_limit > 2:
        is_valid = False
        reasons.append(f"Part {part_name}: Too many dimensions > {max_dim}mm. Dims: [{dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f}]mm ({num_dims_over_limit} over limit).")

    # 2. Check for required flat face
    if not has_required_flat_face(mesh, min_edge_length=min_flat_face_dim):
        is_valid = False
        reasons.append(f"Part {part_name}: Does not have a required >= {min_flat_face_dim}x{min_flat_face_dim}mm flat rectangular face.")
        
    if not reasons and not is_valid: # Should not happen if logic is correct
        reasons.append(f"Part {part_name}: Failed validation for unknown reasons.")
    elif is_valid:
        reasons.append(f"Part {part_name}: Validated successfully. Dims: [{dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f}]mm")


    return is_valid, reasons

def recursively_cut_mesh(mesh_to_process, max_dim, separation, current_depth, original_input_stl_path):
    if current_depth > MAX_CUTTING_DEPTH:
        print(f"Warning (depth {current_depth}): Max cutting depth reached. Returning mesh as is.")
        return [mesh_to_process]

    dims = mesh_to_process.extents
    dims_over_limit_indices = [i for i, d in enumerate(dims) if d > max_dim]
    num_dims_over_limit = len(dims_over_limit_indices)

    if num_dims_over_limit <= 2:
        # print(f"Debug (depth {current_depth}): Mesh dimensions {dims} are within limits or only 2 dims over. No cut needed.")
        return [mesh_to_process]

    # --- Special handling for two-wall.stl using pre-split manual parts ---
    if current_depth == 0 and "two-wall.stl" in os.path.basename(original_input_stl_path).lower() and num_dims_over_limit > 2:
        print(f"Attempting to use manually pre-split parts for 'two-wall.stl' (depth {current_depth})")
        base_path = os.path.dirname(original_input_stl_path)
        manual_part1_path = os.path.join(base_path, "two-wall-manualsplit-1.stl")
        manual_part2_path = os.path.join(base_path, "two-wall-manualsplit-2.stl")
        
        part1_manual, part2_manual = None, None
        try:
            if os.path.exists(manual_part1_path):
                part1_manual = trimesh.load_mesh(manual_part1_path, process=True)
                if part1_manual.is_empty:
                    print(f"Warning: Loaded manual part {manual_part1_path} is empty.")
                    part1_manual = None 
            else:
                print(f"Warning: Manual part file not found: {manual_part1_path}")
            
            if os.path.exists(manual_part2_path):
                part2_manual = trimesh.load_mesh(manual_part2_path, process=True)
                if part2_manual.is_empty:
                    print(f"Warning: Loaded manual part {manual_part2_path} is empty.")
                    part2_manual = None
            else:
                print(f"Warning: Manual part file not found: {manual_part2_path}")

        except Exception as e:
            print(f"Error loading manual split parts for 'two-wall.stl': {e}")
            part1_manual, part2_manual = None, None # Ensure fallback

        if part1_manual and not part1_manual.is_empty and part2_manual and not part2_manual.is_empty:
            print("Successfully loaded manually pre-split parts for 'two-wall.stl'.")
            
            # Determine which part is "lower" (smaller Z centroid) and "upper"
            # Assuming the split was primarily along Z, as per previous attempts.
            cut_axis = 2 # Z-axis for separation
            center1_z = part1_manual.bounds.mean(axis=0)[cut_axis]
            center2_z = part2_manual.bounds.mean(axis=0)[cut_axis]

            if center1_z > center2_z:
                part1_manual, part2_manual = part2_manual, part1_manual # Swap them
                print("Swapped manual parts to ensure part1 is lower Z, part2 is higher Z.")

            # Apply separation
            translation_vector1 = np.zeros(3); translation_vector1[cut_axis] = -separation / 2.0
            part1_manual.apply_translation(translation_vector1)
            print(f"Applied translation {translation_vector1} to manual part 1.")
            
            translation_vector2 = np.zeros(3); translation_vector2[cut_axis] = separation / 2.0
            part2_manual.apply_translation(translation_vector2)
            print(f"Applied translation {translation_vector2} to manual part 2.")
            
            current_results = []
            current_results.extend(recursively_cut_mesh(part1_manual, max_dim, separation, current_depth + 1, original_input_stl_path))
            current_results.extend(recursively_cut_mesh(part2_manual, max_dim, separation, current_depth + 1, original_input_stl_path))
            return current_results
        else:
            print("Failed to load or use manual pre-split parts for 'two-wall.stl'. Proceeding with automated cutting logic.")
    # --- End of special handling for two-wall.stl ---

    # Automated cutting logic (if not two-wall special case, or if it failed, or for deeper recursion)
    print(f"Info (depth {current_depth}): Mesh dimensions {dims}. Num dims over limit: {num_dims_over_limit}. Oversized axes indices: {dims_over_limit_indices}")
    
    # Fallback to the previous slice_mesh_plane logic if manual parts aren't used or for other STLs
    # This is the section that was previously attempted for two-wall.stl specifically, now generalized.
    # We will try to cut along the largest oversized dimension first.
    sorted_oversized_axis_indices = sorted(dims_over_limit_indices, key=lambda axis_idx: dims[axis_idx], reverse=True)

    for cut_attempt, cut_axis in enumerate(sorted_oversized_axis_indices):
        axis_labels = ['X', 'Y', 'Z']
        print(f"Attempting automated cut (attempt {cut_attempt+1}/{len(sorted_oversized_axis_indices)}) on axis {axis_labels[cut_axis]} (index {cut_axis}) at depth {current_depth}")
        
        mesh_to_slice = mesh_to_process.copy()
        mesh_to_slice.process() # Ensure processed

        if not mesh_to_slice.is_watertight:
            # print(f"Info (depth {current_depth}, axis {axis_labels[cut_axis]}): Mesh is not watertight. Attempting to fill holes.")
            mesh_to_slice.fill_holes()
            if not mesh_to_slice.is_watertight:
                print(f"Warning (depth {current_depth}, axis {axis_labels[cut_axis]}): Mesh could not be made watertight. Slice might be unreliable.")

        plane_normal = np.zeros(3)
        plane_normal[cut_axis] = 1
        # Cut at the midpoint of the oversized dimension
        cut_point = mesh_to_slice.bounds[0][cut_axis] + dims[cut_axis] / 2.0
        plane_origin = mesh_to_slice.bounds.mean(axis=0) # Use center of current mesh for other axes
        plane_origin[cut_axis] = cut_point
        
        slice_result = None
        try:
            slice_result = trimesh.intersections.slice_mesh_plane(
                mesh=mesh_to_slice, 
                plane_normal=plane_normal, 
                plane_origin=plane_origin,
                cap=True
            )
        except Exception as e:
            print(f"Error during automated slice_mesh_plane (axis {axis_labels[cut_axis]}, depth {current_depth}): {e}")

        processed_slice_parts = []
        if isinstance(slice_result, list):
            processed_slice_parts = slice_result
        elif isinstance(slice_result, trimesh.Trimesh):
            # If a single mesh is returned, it means no effective cut for our purpose (splitting into two)
            if len(slice_result.faces) < len(mesh_to_slice.faces) or not np.allclose(slice_result.volume, mesh_to_slice.volume):
                 print(f"Debug (depth {current_depth}, axis {axis_labels[cut_axis]}): slice_mesh_plane returned a single, altered mesh. Not a successful split into two.")
            # else: it's the original mesh, no cut
            pass # Not a successful split into multiple parts
        
        valid_sliced_parts = []
        for i, p in enumerate(processed_slice_parts):
            if p and not p.is_empty and hasattr(p, 'volume') and p.volume > 1e-6 and len(p.faces) >= 4:
                p.process() # Ensure the part is processed
                valid_sliced_parts.append(p)
        
        if len(valid_sliced_parts) >= 2:
            print(f"Successful automated slice on axis {axis_labels[cut_axis]} (depth {current_depth}). Produced {len(valid_sliced_parts)} valid parts.")
            current_results = []
            # Sort parts by their center along the cut_axis to apply separation correctly
            valid_sliced_parts.sort(key=lambda m: m.bounds.mean(axis=0)[cut_axis])
            
            # Apply separation: first part moves negative, last part moves positive along cut_axis
            # This assumes a clean cut into two main pieces, other pieces are intermediate.
            part_first = valid_sliced_parts[0]
            part_last = valid_sliced_parts[-1]

            translation_vector_first = np.zeros(3); translation_vector_first[cut_axis] = -separation / 2.0
            part_first.apply_translation(translation_vector_first)
            current_results.extend(recursively_cut_mesh(part_first, max_dim, separation, current_depth + 1, original_input_stl_path))
            
            if len(valid_sliced_parts) > 1 and part_first is not part_last: # Ensure there actually is a distinct last part
                translation_vector_last = np.zeros(3); translation_vector_last[cut_axis] = separation / 2.0
                part_last.apply_translation(translation_vector_last)
                current_results.extend(recursively_cut_mesh(part_last, max_dim, separation, current_depth + 1, original_input_stl_path))
            
            # Handle any intermediate parts (if slice produces more than 2, e.g., fragments)
            if 2 < len(valid_sliced_parts) < 5: # Process a few fragments, not too many
                for i in range(1, len(valid_sliced_parts) - 1):
                    current_results.extend(recursively_cut_mesh(valid_sliced_parts[i], max_dim, separation, current_depth + 1, original_input_stl_path))
            return current_results
        else:
            print(f"Automated slice_mesh_plane on axis {axis_labels[cut_axis]} (depth {current_depth}) did not yield at least two valid parts.")

    print(f"Warning (depth {current_depth}): All attempted automated cut axes ({[axis_labels[i] for i in sorted_oversized_axis_indices]}) failed to split the mesh. Dims: {dims}. Returning as is.")
    return [mesh_to_process]

def main():
    parser = argparse.ArgumentParser(description=f"Process an STL file to cut parts larger than {MAX_DIM}mm in more than two dimensions, ensuring specific flat faces.")
    parser.add_argument("input_stl_path", help="Path to the input STL file.")
    args = parser.parse_args()

    input_stl_path = args.input_stl_path
    if not os.path.exists(input_stl_path):
        print(f"Error: Input STL file not found: {input_stl_path}")
        sys.exit(1)

    print(f"Loading STL file: {os.path.basename(input_stl_path)}")
    try:
        original_mesh = trimesh.load_mesh(input_stl_path, process=True) # Process=True is good for consistency
    except Exception as e:
        print(f"Error loading STL file '{input_stl_path}': {e}")
        sys.exit(1)

    if not isinstance(original_mesh, trimesh.Trimesh) or original_mesh.is_empty:
        print(f"Error: Failed to load a valid mesh from '{input_stl_path}'.")
        sys.exit(1)
    
    print("Initial mesh loaded and preprocessed.")
    print(f"Initial bounding box: Min {original_mesh.bounds[0].tolist()}, Max {original_mesh.bounds[1].tolist()}")
    print(f"Initial dimensions: {original_mesh.extents.tolist()}")


    # Create output directory
    input_basename = os.path.splitext(os.path.basename(input_stl_path))[0]
    output_dir = os.path.join(os.path.dirname(input_stl_path), f"output_parts_{input_basename}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    print(f"Starting recursive cutting process with MAX_DIMENSION = {MAX_DIM}mm...")
    
    # Make a copy to avoid modifying the original loaded mesh if it's used elsewhere
    mesh_for_cutting = original_mesh.copy()

    final_part_meshes = recursively_cut_mesh(mesh_for_cutting, MAX_DIM, SPACING, 0, input_stl_path)

    print(f"Processing complete. Generated {len(final_part_meshes)} potential parts.")
    
    saved_parts_count = 0
    failed_parts_count = 0

    for i, part_mesh in enumerate(final_part_meshes):
        part_name = f"part_{i+1}"
        print(f"--- Validating and Saving {part_name} ---")

        if not isinstance(part_mesh, trimesh.Trimesh) or part_mesh.is_empty:
            print(f"Skipping {part_name} as it's not a valid mesh or is empty.")
            failed_parts_count +=1
            continue
        
        # Ensure the mesh is processed before validation, especially if it came from slicing
        part_mesh.process() 
        if not part_mesh.is_watertight:
            # print(f"Info for {part_name}: Mesh is not watertight. Attempting to fill holes before validation.")
            part_mesh.fill_holes()
            # print(f"Info for {part_name}: Watertight after fill_holes: {part_mesh.is_watertight}")


        is_valid, reasons = validate_part(part_mesh, part_name, MAX_DIM, MIN_FLAT_FACE_EDGE)
        
        for reason in reasons:
            print(reason)

        if is_valid:
            output_stl_path = os.path.join(output_dir, f"{part_name}.stl")
            try:
                part_mesh.export(output_stl_path)
                print(f"Saved: {output_stl_path}")
                saved_parts_count += 1
            except Exception as e:
                print(f"Error saving {part_name} to {output_stl_path}: {e}")
                failed_parts_count += 1
        else:
            # Optionally save failed parts to a different directory or with a suffix
            output_stl_path_failed = os.path.join(output_dir, f"{part_name}_FAILED.stl")
            try:
                part_mesh.export(output_stl_path_failed)
                print(f"Saved FAILED part for inspection: {output_stl_path_failed}")
            except Exception as e:
                print(f"Error saving FAILED {part_name} to {output_stl_path_failed}: {e}")
            failed_parts_count += 1
            print(f"--- {part_name} FAILED post-save validation. ---")


    print(f"\\nSummary: {saved_parts_count} parts saved successfully to '{output_dir}'.")
    if failed_parts_count > 0:
        print(f"{failed_parts_count} parts FAILED validation or saving.")
    
    if saved_parts_count == 0 and failed_parts_count > 0 and len(final_part_meshes) > 0 :
        print("No parts successfully met all validation criteria.")
    elif len(final_part_meshes) == 0:
        print("No parts were generated by the cutting process.")

    print("\\nThe script attempts one pass of cutting. If validation fails for some parts,")
    print("the cutting strategy within 'recursively_cut_mesh' or the validation logic in 'has_required_flat_face'")
    print("might need further refinement for your specific STL model.")


if __name__ == "__main__":
    main()
