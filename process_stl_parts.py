import sys
print("DEBUG_PS: Python interpreter started, sys imported.", flush=True)

try:
    import os
    print("DEBUG_PS: os imported.", flush=True)
    import argparse
    print("DEBUG_PS: argparse imported.", flush=True)
    import numpy as np
    print("DEBUG_PS: numpy imported.", flush=True)
    import trimesh
    print("DEBUG_PS: trimesh imported.", flush=True)
    import math
    print("DEBUG_PS: math imported.", flush=True)
    import logging
    print("DEBUG_PS: logging imported.", flush=True)
    import time
    print("DEBUG_PS: time imported.", flush=True)
    import shutil
    print("DEBUG_PS: shutil imported.", flush=True)
    from scipy.spatial import ConvexHull, Delaunay
    print("DEBUG_PS: scipy.spatial imported.", flush=True)
    from collections import deque
    print("DEBUG_PS: collections.deque imported.", flush=True)
    print("DEBUG_PS: All imports successful.", flush=True)
except Exception as e:
    print(f"DEBUG_PS: Exception during import: {e}", flush=True)
    sys.exit(1) # Exit if imports fail

# Global parameters (adjust as needed)
MAX_DIM = 150.0  # Maximum dimension in mm for any side of the part
MIN_FLAT_FACE_DIMENSION_MM = 8.0 # Minimum length for an edge of the required flat face
MIN_FLAT_FACE_ASPECT_RATIO = 1.0 # Minimum aspect ratio for a face to be considered (e.g., 1.0 for square-like)
FLAT_FACE_ANGLE_TOLERANCE_DEGREES = 1.0 # Max angle deviation for normals to be considered coplanar
MIN_CUTTING_DEPTH = 0
MAX_CUTTING_DEPTH = 10  # Example: Limit recursion to 10 levels
MIN_VOLUME_FOR_SUBSTANTIAL_PART_ANGLED = 1.0 # Example: 1 mm^3, adjust as needed
SPACING_MM = 5.0 # Default spacing for cuts if applicable elsewhere
SEPARATION_MM = 1.0 # Separation distance for cut pieces
FLAT_FACE_NORMAL_TOLERANCE = 1e-3 
FLAT_FACE_FILL_RATIO = 0.80 # Relaxed from 0.85
NORMAL_SIMILARITY_TOLERANCE = 1e-4 # For comparing normals to find unique ones
MAX_DIM_FOR_MULTIAXIS_OVERSIZE_MM = 8.0 # New rule: max dimension if more than two axes exceed this
MAX_DIM_FOR_MULTIAXIS_MM = 10.0 # Define the missing constant
OUTPUT_DIR = "" # Initialize globally, will be set in main

MIN_VOLUME_CUT_PIECE_MM3 = 0.5  # Minimum volume for a piece to be considered substantial after a cut
MIN_FACES_CUT_PIECE = 4      # Minimum faces for a piece
MIN_VOLUME_FOR_SUBSTANTIAL_PART_ANGLED = 1.0 # For angled cuts

# Core helper functions for validation and cutting logic START
# (This block replaces stubs or missing definitions for these functions)

def has_required_flat_face(part_mesh, min_flat_face_dim=8.0):
    """
    Checks if a mesh has at least one flat facet (group of coplanar faces)
    that is approximately rectangular and has dimensions of at least
    min_flat_face_dim x min_flat_face_dim.
    """
    if part_mesh is None or part_mesh.is_empty or len(part_mesh.faces) == 0:
        return False, "Mesh is empty or has no faces."

    if not part_mesh.is_watertight:
        part_mesh.fill_holes()
        if not part_mesh.is_watertight:
            return False, "Mesh not watertight, cannot reliably find flat faces."

    # Accessing part_mesh.facets will trigger their computation if not already done.
    # Ensure facets are computed.
    if not hasattr(part_mesh, 'facets') or not part_mesh.facets:
        try:
            part_mesh.process() # This should compute facets
        except Exception as e:
            return False, f"Error processing mesh for facets: {e}"
        if not hasattr(part_mesh, 'facets') or not part_mesh.facets:
            return False, "No facets found in mesh even after process()."


    min_area_required = min_flat_face_dim * min_flat_face_dim
    fill_ratio_threshold = 0.90 # How much of the 2D bounding box should be filled by the facet area

    for facet_idx in range(len(part_mesh.facets)):
        facet_face_indices = part_mesh.facets[facet_idx]
        if not isinstance(facet_face_indices, (list, np.ndarray)) or len(facet_face_indices) == 0:
            continue

        actual_facet_area = np.sum(part_mesh.area_faces[facet_face_indices])

        if actual_facet_area < min_area_required * 0.8: # Quick area check
            continue

        facet_vertex_indices = np.unique(part_mesh.faces[facet_face_indices].flatten())
        facet_vertices = part_mesh.vertices[facet_vertex_indices]

        if len(facet_vertices) < 3:
            continue

        facet_normal = part_mesh.facets_normal[facet_idx]
        plane_origin = facet_vertices[0]
        # x_axis_direction, y_axis_direction = trimesh.geometry.plane_basis(normal=facet_normal) # OLD, caused AttributeError
        transform_to_xy = trimesh.geometry.plane_transform(origin=plane_origin, normal=facet_normal)
        x_axis_direction = transform_to_xy[0, :3] # First row of rotation part of the transform
        y_axis_direction = transform_to_xy[1, :3] # Second row of rotation part of the transform
        
        vectors_from_origin = facet_vertices - plane_origin
        projected_coords_x = np.dot(vectors_from_origin, x_axis_direction)
        projected_coords_y = np.dot(vectors_from_origin, y_axis_direction)
        
        projected_vertices_2d = np.column_stack((projected_coords_x, projected_coords_y))
        
        min_2d = projected_vertices_2d.min(axis=0)
        max_2d = projected_vertices_2d.max(axis=0)
        dims_2d = max_2d - min_2d

        if dims_2d[0] >= min_flat_face_dim and dims_2d[1] >= min_flat_face_dim:
            bbox_area_2d = dims_2d[0] * dims_2d[1]
            if bbox_area_2d == 0:
                continue
            
            fill_ratio = actual_facet_area / bbox_area_2d
            
            if fill_ratio >= fill_ratio_threshold:
                print(f"DEBUG_HRFF: Found valid flat face on facet {facet_idx} for mesh being validated. Area {actual_facet_area:.2f}, Dims {dims_2d[0]:.2f}x{dims_2d[1]:.2f}, Fill ratio {fill_ratio:.2f}") # MODIFIED: Uncommented and ensured it prints useful info
                return True, "Found required flat face."

    return False, f"No flat rectangular face >= {min_flat_face_dim}x{min_flat_face_dim}mm with fill ratio >= {fill_ratio_threshold} found."

def validate_part(mesh, part_name, max_dim=150.0, min_flat_face_dim=8.0):
    """
    Validates a mesh piece based on several criteria including substantiality and presence of a required flat face.
    """
    validation_passed = True
    messages = []

    if mesh is None or mesh.is_empty:
        return False, [f"Part '{part_name}' is empty or None."]

    # Substantiality checks (using global constants for volume/face count)
    if not hasattr(mesh, 'volume') or mesh.volume < MIN_VOLUME_CUT_PIECE_MM3:
        messages.append(f"Volume {mesh.volume if hasattr(mesh, 'volume') else 'N/A'} < {MIN_VOLUME_CUT_PIECE_MM3}")
        validation_passed = False
    if len(mesh.faces) < MIN_FACES_CUT_PIECE:
        messages.append(f"Faces {len(mesh.faces)} < {MIN_FACES_CUT_PIECE}")
        validation_passed = False

    # Watertight check (critical for reliable facet detection and other properties)
    if not mesh.is_watertight:
        mesh.fill_holes() # Attempt to make it watertight
        if not mesh.is_watertight:
            messages.append("Not watertight even after fill_holes()")
            validation_passed = False # Crucial for flat face detection

    # Flat face check (only if mesh is basically valid and watertight)
    if validation_passed: # Implying it's not empty, has min volume/faces, and is watertight
        has_face, face_msg = has_required_flat_face(mesh, min_flat_face_dim)
        if not has_face:
            messages.append(face_msg)
            validation_passed = False
        else: # MODIFIED: Added else block to record flat face success for verbose output
            messages.append(f"Passed flat face check: {face_msg}")


    final_status_message = f"Part '{part_name}' validation: {'Passed' if validation_passed else 'Failed'}."
    if messages: # Append specific reasons if any
        # MODIFIED: Ensure all messages are joined, even on pass, for clarity
        return validation_passed, [final_status_message + " Details: " + "; ".join(messages)]
    else: # No specific failure messages, but validation_passed might be True or False
        return validation_passed, [final_status_message]

def is_piece_substantial(piece_mesh):
    """Checks if a mesh piece is substantial enough to be considered valid after a cut."""
    if piece_mesh is None or piece_mesh.is_empty:
        return False
    # Using volume as the primary criterion for substantiality
    if hasattr(piece_mesh, 'volume') and piece_mesh.volume >= MIN_VOLUME_CUT_PIECE_MM3: 
        return True
    # If volume is not available or doesn't meet the threshold, check face count
    if len(piece_mesh.faces) >= MIN_FACES_CUT_PIECE:
        return True
    return False

def attempt_two_axis_cuts(mesh_to_cut, first_cut_axis, second_cut_axis, first_cut_origin, output_dir, base_filename):
    """
    Attempts two axis-aligned cuts.
    Returns a status code string and the three resulting pieces if successful.
    Status codes:
    - "SUCCESS"
    - "FIRST_CUT_FAILED_SLICE"
    - "FIRST_CUT_INEFFECTIVE_SUBSTANTIAL" (p1 or p2 not substantial)
    - "FIRST_CUT_INEFFECTIVE_VOLUME_RATIO" (one piece is almost the original volume)
    - "SECOND_CUT_FAILED_SLICE"
    - "SECOND_CUT_INEFFECTIVE_SUBSTANTIAL" (p2a or p2b not substantial)
    - "SECOND_CUT_INEFFECTIVE_VOLUME_RATIO"
    """
    print(f"DEBUG_ATAC: Attempting two-axis cuts. First axis: {first_cut_axis}, Second axis: {second_cut_axis}, First cut origin: {first_cut_origin}")
    mesh_copy = mesh_to_cut.copy()
    original_volume = mesh_copy.volume

    # First cut
    plane_normal1 = np.zeros(3)
    plane_normal1[first_cut_axis] = 1.0
    
    try:
        piece1 = trimesh.intersections.slice_mesh_plane(mesh_copy, plane_normal=plane_normal1, plane_origin=first_cut_origin, cap=True)
        remaining_piece = trimesh.intersections.slice_mesh_plane(mesh_copy, plane_normal=-plane_normal1, plane_origin=first_cut_origin, cap=True)
    except Exception as e:
        print(f"ERROR_ATAC: First cut failed during slice_mesh_plane: {e}")
        return "FIRST_CUT_FAILED_SLICE", None

    if not piece1 or piece1.is_empty or not remaining_piece or remaining_piece.is_empty:
        print("DEBUG_ATAC: First cut resulted in one or more empty pieces.")
        return "FIRST_CUT_FAILED_SLICE", None

    if not is_piece_substantial(piece1) or not is_piece_substantial(remaining_piece):
        print(f"DEBUG_ATAC: First cut - p1 substantial: {is_piece_substantial(piece1)}, remaining substantial: {is_piece_substantial(remaining_piece)}. One piece not substantial.")
        return "FIRST_CUT_INEFFECTIVE_SUBSTANTIAL", None

    volume_ratio_threshold = 0.98 
    if piece1.volume / original_volume > volume_ratio_threshold or \
       remaining_piece.volume / original_volume > volume_ratio_threshold:
        print(f"DEBUG_ATAC: First cut ineffective - volume ratio issue. p1_vol: {piece1.volume}, rem_vol: {remaining_piece.volume}, orig_vol: {original_volume}")
        return "FIRST_CUT_INEFFECTIVE_VOLUME_RATIO", None
    
    print(f"DEBUG_ATAC: First cut successful. Piece1 vol: {piece1.volume}, Remaining piece vol: {remaining_piece.volume}")

    # Second cut on the remaining_piece
    plane_normal2 = np.zeros(3)
    plane_normal2[second_cut_axis] = 1.0
    second_cut_origin = remaining_piece.center_mass 

    try:
        piece2a = trimesh.intersections.slice_mesh_plane(remaining_piece, plane_normal=plane_normal2, plane_origin=second_cut_origin, cap=True)
        piece2b = trimesh.intersections.slice_mesh_plane(remaining_piece, plane_normal=-plane_normal2, plane_origin=second_cut_origin, cap=True)
    except Exception as e:
        print(f"ERROR_ATAC: Second cut failed during slice_mesh_plane: {e}")
        return "SECOND_CUT_FAILED_SLICE", None

    if not piece2a or piece2a.is_empty or not piece2b or piece2b.is_empty:
        print("DEBUG_ATAC: Second cut resulted in one or more empty pieces.")
        return "SECOND_CUT_FAILED_SLICE", None

    if not is_piece_substantial(piece2a) or not is_piece_substantial(piece2b):
        print(f"DEBUG_ATAC: Second cut - p2a substantial: {is_piece_substantial(piece2a)}, p2b substantial: {is_piece_substantial(piece2b)}. One piece not substantial.")
        return "SECOND_CUT_INEFFECTIVE_SUBSTANTIAL", None

    original_volume_for_second_cut = remaining_piece.volume
    if piece2a.volume / original_volume_for_second_cut > volume_ratio_threshold or \
       piece2b.volume / original_volume_for_second_cut > volume_ratio_threshold:
        print(f"DEBUG_ATAC: Second cut ineffective - volume ratio issue. p2a_vol: {piece2a.volume}, p2b_vol: {piece2b.volume}, rem_vol: {original_volume_for_second_cut}")
        return "SECOND_CUT_INEFFECTIVE_VOLUME_RATIO", None

    print(f"DEBUG_ATAC: Second cut successful. Piece2a vol: {piece2a.volume}, Piece2b vol: {piece2b.volume}")
    
    return "SUCCESS", [piece1, piece2a, piece2b]

def perform_three_wall_corner_cuts(mesh, output_dir, original_filename_base):
    print(f"DEBUG_P3WCC: Starting three-wall corner cuts for {original_filename_base}", flush=True) # ADDED flush
    mesh_copy = mesh.copy() 
    bounds = mesh_copy.bounds
    if bounds is None or len(bounds) < 2:
        print(f"ERROR_P3WCC: Invalid bounds for {original_filename_base}. Cannot proceed.", flush=True)
        return []
    min_corner = bounds[0]
    max_corner = bounds[1]
    mesh_extents = mesh_copy.extents
    print(f"DEBUG_P3WCC: Mesh bounds: {bounds}, Extents: {mesh_extents}", flush=True) # ADDED flush and bounds print

    corners = [
        min_corner,
        [max_corner[0], min_corner[1], min_corner[2]],
        [min_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], max_corner[1], min_corner[2]],
        [max_corner[0], min_corner[1], max_corner[2]],
        [min_corner[0], max_corner[1], max_corner[2]],
        max_corner
    ]

    principal_axes = [0, 1, 2] 

    for corner_idx, corner_origin_bb in enumerate(corners):
        print(f"DEBUG_P3WCC: Iterating corner {corner_idx}: {corner_origin_bb}", flush=True) # ADDED flush
        for first_cut_axis in principal_axes:
            print(f"DEBUG_P3WCC:   Iterating first_cut_axis {first_cut_axis}", flush=True) # ADDED flush
            offset_direction_scalar = 1.0 if np.isclose(corner_origin_bb[first_cut_axis], min_corner[first_cut_axis]) else -1.0
            offset_distance = mesh_extents[first_cut_axis] * 0.25 if mesh_extents[first_cut_axis] > 1e-6 else 1.0
            effective_first_cut_origin_coords = np.copy(corner_origin_bb)
            effective_first_cut_origin_coords[first_cut_axis] += offset_direction_scalar * offset_distance
            
            realigned_first_cut = False
            filename_suffix_realignment_indicator = ""
            current_first_cut_origin_point = effective_first_cut_origin_coords

            if not (min_corner[first_cut_axis] <= effective_first_cut_origin_coords[first_cut_axis] <= max_corner[first_cut_axis]):
                 print(f"DEBUG_P3WCC: Offset from corner {corner_idx}, axis {first_cut_axis} resulted in origin {effective_first_cut_origin_coords} outside bounds. Using center_mass for this attempt's first cut.")
                 current_first_cut_origin_point = mesh_copy.center_mass
                 realigned_first_cut = True 
                 filename_suffix_realignment_indicator = "r"
            
            print(f"DEBUG_P3WCC: Trying Corner {corner_idx}, First Cut Axis: {first_cut_axis}, Initial First Cut Origin: {current_first_cut_origin_point}", flush=True) # ADDED flush

            for attempt_count in range(2): 
                if attempt_count == 1: 
                    if realigned_first_cut: 
                        continue 
                    print(f"DEBUG_P3WCC: Re-attempting first cut from center_mass for C{corner_idx}, Axis {first_cut_axis}")
                    current_first_cut_origin_point = mesh_copy.center_mass
                    realigned_first_cut = True
                    filename_suffix_realignment_indicator = "r"
                
                print(f"DEBUG_P3WCC:     Attempt {attempt_count + 1} (Realigned: {realigned_first_cut})", flush=True) # ADDED flush
                status_overall = "INIT" # To track status from attempt_two_axis_cuts through validation
                final_validated_pieces = []

                for second_cut_axis in principal_axes:
                    if second_cut_axis == first_cut_axis:
                        continue

                    print(f"DEBUG_P3WCC:       Trying Second Cut Axis: {second_cut_axis} (First cut origin: {current_first_cut_origin_point}, Realigned: {realigned_first_cut})", flush=True) # ADDED flush
                    
                    base_fn = f"{original_filename_base}_c{corner_idx}{filename_suffix_realignment_indicator}_ax{first_cut_axis}{second_cut_axis}"
                    
                    status, pieces = attempt_two_axis_cuts(
                        mesh_copy, 
                        first_cut_axis, 
                        second_cut_axis, 
                        current_first_cut_origin_point, 
                        output_dir, 
                        base_fn 
                    )
                    status_overall = status # Store status from cutting attempt

                    if status == "SUCCESS" and pieces and len(pieces) == 3:
                        p1_final, p2a_final, p2b_final = pieces
                        print(f"DEBUG_P3WCC: attempt_two_axis_cuts successful for C{corner_idx}{filename_suffix_realignment_indicator}, Axes {first_cut_axis},{second_cut_axis}. Proceeding to validate 3 pieces.")

                        all_pieces_fully_validated = True
                        validated_and_saved_pieces_for_this_attempt = []
                        temp_saved_filenames_for_this_attempt = []
                        
                        pieces_to_validate = [p1_final, p2a_final, p2b_final]
                        final_base_filename_for_parts = f"{original_filename_base}_c{corner_idx}{filename_suffix_realignment_indicator}_ax{first_cut_axis}{second_cut_axis}"

                        for i, piece_mesh_to_validate in enumerate(pieces_to_validate):
                            part_final_name_suffix = f"_p{i+1}"
                            part_final_filename = f"{final_base_filename_for_parts}{part_final_name_suffix}.stl"
                            part_final_path = os.path.join(output_dir, part_final_filename)
                            validation_part_desc = f"{original_filename_base}_C{corner_idx}{filename_suffix_realignment_indicator}_Ax{first_cut_axis}{second_cut_axis}_P{i+1}"

                            is_valid_final_piece, validation_msgs = validate_part(
                                piece_mesh_to_validate, 
                                validation_part_desc,
                                min_flat_face_dim=MIN_FLAT_FACE_DIMENSION_MM
                            )
                            if is_valid_final_piece:
                                print(f"DEBUG_P3WCC: Final piece {part_final_filename} PASSED validation.")
                                try:
                                    if piece_mesh_to_validate and not piece_mesh_to_validate.is_empty:
                                        piece_mesh_to_validate.export(part_final_path)
                                        print(f"Saved successfully validated corner cut part: {part_final_path}")
                                        validated_and_saved_pieces_for_this_attempt.append(piece_mesh_to_validate)
                                        temp_saved_filenames_for_this_attempt.append(part_final_path)
                                    else:
                                        print(f"ERROR_P3WCC: Validated piece {part_final_filename} is None or empty, cannot export.")
                                        all_pieces_fully_validated = False; break
                                except Exception as e_export_final:
                                    print(f"ERROR_P3WCC: Failed to export validated piece {part_final_filename}: {e_export_final}")
                                    all_pieces_fully_validated = False; break 
                            else:
                                print(f"DEBUG_P3WCC: Final piece {part_final_filename} (from {validation_part_desc}) FAILED validation: {'; '.join(validation_msgs)}. Strategy C{corner_idx}{filename_suffix_realignment_indicator}_Ax{first_cut_axis}{second_cut_axis} failed for this piece.")
                                all_pieces_fully_validated = False; break 

                        if all_pieces_fully_validated and len(validated_and_saved_pieces_for_this_attempt) == 3:
                            print(f"DEBUG_P3WCC: All 3 pieces for C{corner_idx}{filename_suffix_realignment_indicator}_Ax{first_cut_axis}{second_cut_axis} PASSED validation and were saved. Strategy successful.")
                            final_validated_pieces = validated_and_saved_pieces_for_this_attempt
                            status_overall = "VALIDATION_SUCCESS" # Mark overall success
                            return final_validated_pieces # Success!
                        else: # Validation failed for one or more pieces
                            print(f"DEBUG_P3WCC: Full validation of 3 pieces failed for C{corner_idx}{filename_suffix_realignment_indicator}_Ax{first_cut_axis}{second_cut_axis}. Cleaning up.")
                            status_overall = "VALIDATION_FAILED"
                            for f_path in temp_saved_filenames_for_this_attempt:
                                if os.path.exists(f_path):
                                    try: os.remove(f_path); print(f"DEBUG_P3WCC: Cleaned up: {f_path}")
                                    except Exception as e_clean: print(f"ERROR_P3WCC: Failed to clean up {f_path}: {e_clean}")
                            # Continue to the next second_cut_axis or realignment attempt
                    
                    elif status in ["FIRST_CUT_INEFFECTIVE_SUBSTANTIAL", "FIRST_CUT_INEFFECTIVE_VOLUME_RATIO"]:
                        print(f"DEBUG_P3WCC: First cut ineffective (status: {status}) for C{corner_idx}{filename_suffix_realignment_indicator}, Axis {first_cut_axis}.")
                        if attempt_count == 0: break # Break from second_cut_axis loop to go to realignment attempt
                    else: 
                        print(f"DEBUG_P3WCC: attempt_two_axis_cuts failed for C{corner_idx}{filename_suffix_realignment_indicator}, Axes {first_cut_axis},{second_cut_axis} with status: {status}. Trying next.")
                # End of second_cut_axis loop
                
                if status_overall == "VALIDATION_SUCCESS": # Should have returned if successful
                     pass # Should be unreachable due to return
                elif attempt_count == 0 and status_overall in ["FIRST_CUT_INEFFECTIVE_SUBSTANTIAL", "FIRST_CUT_INEFFECTIVE_VOLUME_RATIO"]:
                    print(f"DEBUG_P3WCC: Proceeding to realignment for C{corner_idx}, Axis {first_cut_axis} due to {status_overall}.")
                    continue # To the next attempt_count (realignment)
                elif status_overall == "FIRST_CUT_FAILED_SLICE":
                    print(f"DEBUG_P3WCC: First cut slice failed for C{corner_idx}, Axis {first_cut_axis}. No realignment for this, moving to next first_cut_axis.")
                    break # Break from attempt_count loop (skipping realignment for this first_cut_axis)
            # End of attempt_count loop (initial and realignment)
    # End of first_cut_axis loop
    # End of corner_idx loop

    print(f"DEBUG_P3WCC: No successful three-wall cutting strategy found for {original_filename_base} after checking all combinations.", flush=True) # ADDED flush
    return []

# Core helper functions for validation and cutting logic END

# def _evaluate_candidate_cut(original_mesh_to_slice, plane_normal, plane_origin, cut_description, current_depth):
# ... existing code ...
def main():
    print("DEBUG_MAIN: main() function started.", flush=True) # ADDED flush
    parser = argparse.ArgumentParser(description='Process STL parts for cutting.')
    parser.add_argument('input_stl', type=str, help='Path to the input STL file.')
    parser.add_argument('--output_dir', type=str, default='output_parts', help='Directory to save output parts.')
    parser.add_argument('--run_three_wall_corner_cuts', action='store_true', help='Flag to run three-wall corner cuts.')
    args = parser.parse_args()

    original_filename_base = os.path.splitext(os.path.basename(args.input_stl))[0]

    print(f"DEBUG_MAIN: Parsed arguments: input_stl={args.input_stl}, output_dir={args.output_dir}, run_three_wall_corner_cuts={args.run_three_wall_corner_cuts}", flush=True) # ADDED flush

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    # Create base output directory if it doesn't exist
    base_output_dir_for_file = os.path.join(OUTPUT_DIR, original_filename_base) # MODIFIED to use original_filename_base
    if not os.path.exists(base_output_dir_for_file):
        os.makedirs(base_output_dir_for_file)
        print(f"DEBUG_MAIN: Created base output directory: {base_output_dir_for_file}", flush=True) # ADDED flush

    try:
        mesh = trimesh.load_mesh(args.input_stl)
        print(f"DEBUG_MAIN: Mesh {args.input_stl} loaded successfully. Volume: {mesh.volume if hasattr(mesh, 'volume') else 'N/A'}", flush=True) # ADDED flush
    except Exception as e:
        print(f"ERROR_MAIN: Failed to load STL file {args.input_stl}: {e}", flush=True) # ADDED flush
        return

    if mesh.is_empty:
        print(f"ERROR_MAIN: Loaded mesh is empty. Cannot proceed with cutting.", flush=True)
        return

    # Align mesh to its Oriented Bounding Box (OBB) for consistent cutting
    obb = mesh.bounding_box_oriented
    obb_transform = np.eye(4)
    obb_transform[:3, :3] = obb.primitive.transform[:3, :3]
    mesh.apply_transform(obb_transform)
    print("DEBUG_MAIN: Mesh aligned to OBB.", flush=True) # ADDED flush

    # Translate mesh so its min corner is at the origin
    min_bounds = mesh.bounds[0]
    translation_to_origin = -min_bounds
    mesh.apply_translation(translation_to_origin)
    print(f"DEBUG_MAIN: Mesh translated by {translation_to_origin} so min corner is at origin. New bounds: {mesh.bounds}", flush=True) # ADDED flush


    three_wall_cuts_successful = False
    if args.run_three_wall_corner_cuts:
        print(f"DEBUG_MAIN: --run_three_wall_corner_cuts is TRUE. Attempting three-wall corner cuts for {original_filename_base}", flush=True) # ADDED flush
        
        corner_cuts_output_dir = os.path.join(base_output_dir_for_file, f"{original_filename_base}_corner_cuts")
        if not os.path.exists(corner_cuts_output_dir):
            os.makedirs(corner_cuts_output_dir)
            print(f"DEBUG_MAIN: Created corner cuts output directory: {corner_cuts_output_dir}", flush=True) # ADDED flush

        resulting_pieces = perform_three_wall_corner_cuts(mesh, corner_cuts_output_dir, original_filename_base)
        
        if resulting_pieces and len(resulting_pieces) == 3:
            print(f"DEBUG_MAIN: Three-wall corner cutting successful for {original_filename_base}. Found {len(resulting_pieces)} pieces.", flush=True) # ADDED flush
            three_wall_cuts_successful = True
        else:
            print(f"DEBUG_MAIN: Three-wall corner cutting did not produce 3 validated pieces for {original_filename_base}.", flush=True) # ADDED flush
    else:
        print("DEBUG_MAIN: --run_three_wall_corner_cuts is FALSE. Skipping three-wall corner cuts.", flush=True) # ADDED flush

    if not three_wall_cuts_successful:
        print(f"DEBUG_MAIN: Three-wall cuts not successful or not run. Proceeding to standard angled cutting for {original_filename_base} (if implemented).", flush=True) # ADDED flush
        # ... (standard angled cutting logic would go here) ...
        print(f"DEBUG_MAIN: Standard angled cutting logic placeholder for {original_filename_base}.", flush=True) # ADDED flush
    else:
        print(f"DEBUG_MAIN: Three-wall cuts were successful. Skipping standard angled cutting for {original_filename_base}.", flush=True) # ADDED flush
    
    print(f"DEBUG_MAIN: Processing finished for {args.input_stl}.", flush=True) # ADDED flush

if __name__ == '__main__':
    print("DEBUG_PS: Script execution started (__name__ == '__main__').", flush=True) # ADDED flush
    main()
    print("DEBUG_PS: Script execution finished.", flush=True) # ADDED flush
