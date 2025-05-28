import sys
import os
import argparse
import numpy as np
import trimesh
from typing import List, Tuple

# Constants for validation and cutting logic
MIN_FLAT_FACE_DIMENSION_MM = 8.0  # Minimum dimension for a face to be considered 'flat' and potentially part of a corner
FLAT_FACE_FILL_RATIO = 0.95  # Minimum ratio of face area to its bounding box area
FLAT_FACE_NORMAL_TOLERANCE = 0.05  # Radians, about 2.8 degrees, for normal alignment with principal axes
MIN_VOLUME_CUT_PIECE_MM3 = 1.0 # Minimum volume for a piece to be considered substantial after a cut
MIN_FACES_CUT_PIECE = 10 # Minimum number of faces for a piece to be considered substantial

# Constants for angled cuts (can be adjusted)
MIN_VOLUME_FOR_SUBSTANTIAL_PART_ANGLED = 1.0 # For angled cuts
MIN_FACES_FOR_SUBSTANTIAL_PART_ANGLED = 10 # For angled cuts

OUTPUT_DIR = "output_parts"

# Core helper functions for validation and cutting logic START
# (This block replaces stubs or missing definitions for these functions)

def detect_three_wall_corner_candidate(mesh, min_dim=MIN_FLAT_FACE_DIMENSION_MM, fill_ratio_threshold=FLAT_FACE_FILL_RATIO, normal_alignment_tolerance=FLAT_FACE_NORMAL_TOLERANCE):
    """
    Detects if the OBB-aligned mesh is a candidate for three-wall corner cutting.
    Checks for three large, flat, mutually orthogonal faces aligned with X, Y, Z axes.
    """
    if mesh is None or mesh.is_empty:
        print("DEBUG_DTWCC: Mesh is empty.", flush=True)
        return False
    
    try:
        _ = mesh.facets # Trigger facet computation
        if not hasattr(mesh, 'facets_normal') or mesh.facets_normal is None or len(mesh.facets_normal) == 0:
            print("DEBUG_DTWCC: Mesh has no facets or facet normals after attempting to compute facets.", flush=True)
            return False
    except Exception as e:
        print(f"DEBUG_DTWCC: Exception during facet computation: {e}", flush=True)
        return False

    print(f"DEBUG_DTWCC: Starting detection. Mesh has {len(mesh.facets_normal)} facets. Min_dim: {min_dim}, Fill_ratio: {fill_ratio_threshold}, Normal_tolerance: {normal_alignment_tolerance}", flush=True)

    found_axis_faces = [False, False, False]  # X, Y, Z
    principal_axis_vectors_abs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

    for facet_idx, facet_normal in enumerate(mesh.facets_normal):
        abs_facet_normal = np.abs(facet_normal)
        aligned_axis_index = -1

        for i in range(3): # 0 for X, 1 for Y, 2 for Z
            # Check if the absolute normal is close to a principal axis vector
            if np.allclose(abs_facet_normal, principal_axis_vectors_abs[i], atol=normal_alignment_tolerance):
                aligned_axis_index = i
                break
        
        if aligned_axis_index != -1 and not found_axis_faces[aligned_axis_index]:
            facet_face_indices = mesh.facets[facet_idx]
            if not isinstance(facet_face_indices, (list, np.ndarray)) or len(facet_face_indices) == 0:
                continue

            actual_facet_area = np.sum(mesh.area_faces[facet_face_indices])
            
            if actual_facet_area < (min_dim * min_dim * 0.5): # Heuristic pre-filter
                continue

            facet_vertex_indices = np.unique(mesh.faces[facet_face_indices].flatten())
            facet_vertices = mesh.vertices[facet_vertex_indices]

            if len(facet_vertices) < 3:
                continue
            
            # Project facet vertices onto the plane defined by its normal to get 2D dimensions
            plane_origin = facet_vertices[0]
            transform_to_xy_plane = trimesh.geometry.plane_transform(origin=plane_origin, normal=facet_normal)
            x_axis_direction_plane = transform_to_xy_plane[0, :3]
            y_axis_direction_plane = transform_to_xy_plane[1, :3]
            
            vectors_from_origin = facet_vertices - plane_origin
            projected_coords_x = np.dot(vectors_from_origin, x_axis_direction_plane)
            projected_coords_y = np.dot(vectors_from_origin, y_axis_direction_plane)
            
            projected_vertices_2d = np.column_stack((projected_coords_x, projected_coords_y))
            
            min_2d = np.min(projected_vertices_2d, axis=0)
            max_2d = np.max(projected_vertices_2d, axis=0)
            dims_2d = max_2d - min_2d
            
            bounding_box_area_2d = dims_2d[0] * dims_2d[1]
            if bounding_box_area_2d < 1e-9: # Avoid division by zero or tiny areas
                continue
            
            current_fill_ratio = actual_facet_area / bounding_box_area_2d

            if dims_2d[0] >= min_dim and dims_2d[1] >= min_dim and current_fill_ratio >= fill_ratio_threshold:
                axis_name = ['X', 'Y', 'Z'][aligned_axis_index]
                print(f"DEBUG_DTWCC: Found qualifying facet for axis {axis_name} (idx {aligned_axis_index}). Normal: {facet_normal}, Dims: {dims_2d}, Area: {actual_facet_area:.2f}, Fill: {current_fill_ratio:.2f}", flush=True)
                found_axis_faces[aligned_axis_index] = True
                if all(found_axis_faces):
                    print("DEBUG_DTWCC: Found qualifying orthogonal faces for all three principal axes (X, Y, Z). Candidate detected.", flush=True)
                    return True
            # else: # Optional: for debugging facets that are axis-aligned but don't meet size/fill
                # axis_name_debug = ['X', 'Y', 'Z'][aligned_axis_index]
                # print(f"DEBUG_DTWCC: Facet for axis {axis_name_debug} (idx {aligned_axis_index}) did not meet size/fill. Dims: {dims_2d}, Fill: {current_fill_ratio:.2f}, Actual Area: {actual_facet_area:.2f}, BBox Area: {bounding_box_area_2d:.2f}", flush=True)


    if any(found_axis_faces):
        print(f"DEBUG_DTWCC: Found some axis-aligned faces but not all three. X: {found_axis_faces[0]}, Y: {found_axis_faces[1]}, Z: {found_axis_faces[2]}", flush=True)
    else:
        print("DEBUG_DTWCC: Did not find any qualifying large axis-aligned faces.", flush=True)
    return False

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
        # Ensure facets are computed by accessing them.
        # This can raise if the mesh is not well-formed.
        try:
            _ = part_mesh.facets
            _ = part_mesh.facets_normal # Also ensure normals are computed
        except Exception as e:
            # print(f"DEBUG_HSRFF: Could not compute facets for part_mesh: {e}", flush=True)
            return False, f"Could not compute facets for mesh: {e}"
    # ...existing code...
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
                continue # Avoid division by zero
            
            fill_ratio = actual_facet_area / bbox_area_2d
            
            if fill_ratio >= fill_ratio_threshold:
                print(f"DEBUG_HRFF: Found valid flat face on facet {facet_idx} for mesh being validated. Area {actual_facet_area:.2f}, Dims {dims_2d[0]:.2f}x{dims_2d[1]:.2f}, Fill ratio {fill_ratio:.2f}", flush=True)
                return True, f"Found flat face: Area {actual_facet_area:.2f}, Dims {dims_2d[0]:.2f}x{dims_2d[1]:.2f}, Fill {fill_ratio:.2f}"

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
            # messages.append("Not watertight even after fill_holes()") # Original line
            messages.append(f"Part '{part_name}' is not watertight even after fill_holes()") # MODIFIED for clarity
            validation_passed = False # MODIFIED: Ensure this sets validation_passed to False
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
        final_status_message += " Reasons: " + "; ".join(messages)
    return validation_passed, messages # MODIFIED to return messages list for better detail

def is_piece_substantial(piece_mesh):
    """Checks if a mesh piece is substantial enough to be considered valid after a cut."""
    if piece_mesh is None or piece_mesh.is_empty:
        return False
    # Using volume as the primary criterion for substantiality
    if hasattr(piece_mesh, 'volume') and piece_mesh.volume >= MIN_VOLUME_CUT_PIECE_MM3: 
        return True
    # If volume is not available or doesn't meet the threshold, check face count
    if len(piece_mesh.faces) >= MIN_FACES_CUT_PIECE: # Ensure this check is robust
        return True
    # print(f"DEBUG_IPS: Piece substantiality check failed. Volume: {getattr(piece_mesh, 'volume', 'N/A')}, Faces: {len(piece_mesh.faces)}", flush=True)
    return False

def attempt_two_axis_cuts(mesh_to_cut, first_cut_axis, second_cut_axis, first_cut_origin): # MODIFIED signature
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
    # print(f"DEBUG_ATAC: Attempting two-axis cuts. First axis: {first_cut_axis}, Second axis: {second_cut_axis}, First cut origin: {first_cut_origin}") # Keep if needed
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

def perform_three_wall_corner_cuts(mesh, original_filename_base: str) -> List[trimesh.Trimesh]: # MODIFIED signature and return type
    print(f"DEBUG_P3WCC: Starting three-wall corner cuts for {original_filename_base}", flush=True)
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
                if attempt_count == 1: # Realignment attempt
                    # Realign: Use center_mass for the first cut origin if the offset-based one was problematic or first attempt failed certain ways
                    current_first_cut_origin_point = mesh_copy.center_mass
                    realigned_first_cut = True
                    filename_suffix_realignment_indicator = "r" # Update indicator for logging
                    print(f"DEBUG_P3WCC:   Realigning for C{corner_idx}, Axis {first_cut_axis}. New First Cut Origin: {current_first_cut_origin_point}", flush=True)
                
                # print(f"DEBUG_P3WCC:     Attempt {attempt_count + 1} (Realigned: {realigned_first_cut})", flush=True) # ADDED flush
                # status_overall = "INIT" # To track status from attempt_two_axis_cuts through validation # Not strictly needed if we return early
                # final_validated_pieces = [] # Not strictly needed if we return early

                for second_cut_axis in principal_axes:
                    if second_cut_axis == first_cut_axis:
                        continue # Skip cutting along the same axis twice consecutively

                    # print(f"DEBUG_P3WCC:       Second Cut Axis: {second_cut_axis}", flush=True)
                    # MODIFIED: Call to attempt_two_axis_cuts without output_dir and base_filename args
                    status, pieces = attempt_two_axis_cuts(mesh_copy, first_cut_axis, second_cut_axis, current_first_cut_origin_point)
                    # status_overall = status # Store status from cutting attempt # Not strictly needed

                    if status == "SUCCESS" and pieces and len(pieces) == 3:
                        # print(f"DEBUG_P3WCC: attempt_two_axis_cuts successful for C{corner_idx}{filename_suffix_realignment_indicator}, Axes {first_cut_axis},{second_cut_axis}. Validating pieces.", flush=True)
                        
                        current_run_validated_pieces = []
                        all_pieces_in_run_valid = True
                        for piece_idx, p_mesh in enumerate(pieces):
                            temp_part_name = f"{original_filename_base}_C{corner_idx}{filename_suffix_realignment_indicator}_Axis{first_cut_axis}_{second_cut_axis}_p{piece_idx+1}"
                            is_p_valid, p_msgs = validate_part(p_mesh, temp_part_name)
                            if is_p_valid:
                                current_run_validated_pieces.append(p_mesh)
                            else:
                                all_pieces_in_run_valid = False
                                print(f"DEBUG_P3WCC: Piece {temp_part_name} failed validation: {'; '.join(p_msgs)}", flush=True)
                                break 
                        
                        if all_pieces_in_run_valid and len(current_run_validated_pieces) == 3:
                            print(f"DEBUG_P3WCC: Successfully cut and validated 3 pieces for C{corner_idx}{filename_suffix_realignment_indicator}, Axes {first_cut_axis},{second_cut_axis}. Returning them.", flush=True)
                            return current_run_validated_pieces # Return list of mesh objects
                        else:                            
                            print(f"DEBUG_P3WCC: Validation failed for one or more pieces from C{corner_idx}{filename_suffix_realignment_indicator}, Axes {first_cut_axis},{second_cut_axis}.", flush=True)
                    elif status in ["FIRST_CUT_INEFFECTIVE_SUBSTANTIAL", "FIRST_CUT_INEFFECTIVE_VOLUME_RATIO"]:
                        # print(f"DEBUG_P3WCC: First cut ineffective (status: {status}) for C{corner_idx}{filename_suffix_realignment_indicator}, Axis {first_cut_axis}.")
                        if attempt_count == 0: # Only break from second_cut_axis loop to try realignment if it's the first attempt
                            print(f"DEBUG_P3WCC: First cut ineffective for C{corner_idx}, Axis {first_cut_axis} (Status: {status}). Will try realignment if applicable.", flush=True)
                            # status_overall = status # track this to decide if realignment should be tried
                            break # Break from second_cut_axis loop, to outer attempt_count loop for realignment
                    # else: # Other failures from attempt_two_axis_cuts
                        # print(f"DEBUG_P3WCC: attempt_two_axis_cuts failed for C{corner_idx}{filename_suffix_realignment_indicator}, Axes {first_cut_axis},{second_cut_axis} with status: {status}. Trying next second_cut_axis.")
                # End of second_cut_axis loop
                
                # if status_overall == "VALIDATION_SUCCESS": # Should have returned if successful
                #      pass # Should be unreachable due to return
                # elif attempt_count == 0 and status_overall in ["FIRST_CUT_INEFFECTIVE_SUBSTANTIAL", "FIRST_CUT_INEFFECTIVE_VOLUME_RATIO"]:
                #     print(f"DEBUG_P3WCC: Proceeding to realignment for C{corner_idx}, Axis {first_cut_axis} due to {status_overall}.")
                #     continue # To the next attempt_count (realignment)
                # elif status_overall == "FIRST_CUT_FAILED_SLICE":
                #     print(f"DEBUG_P3WCC: First cut slice failed for C{corner_idx}, Axis {first_cut_axis}. No realignment for this, moving to next first_cut_axis.")
                #     break # Break from attempt_count loop (skipping realignment for this first_cut_axis)
                # If a successful cut was found and returned, this part is not reached for that first_cut_axis.
                # If the inner loop (second_cut_axis) completed without returning, and it was due to a first_cut_issue that warrants realignment:
                # The `break` from `second_cut_axis` loop when `status in ["FIRST_CUT_INEFFECTIVE_SUBSTANTIAL", ...]` and `attempt_count == 0`
                # will lead to the next `attempt_count` iteration (realignment).
                # If `status == "FIRST_CUT_FAILED_SLICE"`, we break from `attempt_count` loop entirely for this `first_cut_axis`.
                if status == "FIRST_CUT_FAILED_SLICE": # Check status from the last attempt_two_axis_cuts in the inner loop
                    print(f"DEBUG_P3WCC: First cut slice failed for C{corner_idx}, Axis {first_cut_axis} on attempt {attempt_count+1}. Moving to next first_cut_axis.", flush=True)
                    break # from attempt_count loop, try next first_cut_axis

            # End of attempt_count loop (initial and realignment)
    # End of first_cut_axis loop
    # End of corner_idx loop

    print(f"DEBUG_P3WCC: No successful three-wall cutting strategy found for {original_filename_base} after checking all combinations.", flush=True)
    return [] # Return empty list if no success

def perform_standard_angled_cut(mesh_to_cut, original_filename_base: str) -> List[trimesh.Trimesh]: # MODIFIED signature and return type
    print(f"DEBUG_PSAC: Starting standard angled cut for {original_filename_base}", flush=True)
    mesh_copy = mesh_to_cut.copy()
    if mesh_copy.is_empty:
        print("DEBUG_PSAC: Input mesh is empty.", flush=True)
        return []

    potential_normals = [
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, -1.0]),
    ]

    plane_origin = mesh_copy.center_mass
    print(f"DEBUG_PSAC: Using plane origin (center_mass): {plane_origin}", flush=True)

    for i, normal_vec in enumerate(potential_normals):
        # Normalize the normal vector
        norm = np.linalg.norm(normal_vec)
        if norm == 0: # Should not happen with predefined normals
            continue
        plane_normal = normal_vec / norm
        
        print(f"DEBUG_PSAC: Attempting cut {i+1} with normal: {plane_normal}", flush=True)

        try:
            # Slice the mesh
            part1 = trimesh.intersections.slice_mesh_plane(mesh_copy, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)
            part2 = trimesh.intersections.slice_mesh_plane(mesh_copy, plane_normal=-plane_normal, plane_origin=plane_origin, cap=True)
        except Exception as e:
            print(f"ERROR_PSAC: slice_mesh_plane failed for normal {plane_normal}: {e}", flush=True)
            continue

        if not part1 or part1.is_empty or not part2 or part2.is_empty:
            print(f"DEBUG_PSAC: Cut {i+1} (normal: {plane_normal}) resulted in one or more empty pieces.", flush=True)
            continue
        
        # Validate pieces using the existing validate_part function
        # Default min_flat_face_dim from MIN_FLAT_FACE_DIMENSION_MM (8.0) will be used.
        # This might be strict for angled faces, but parts might retain other original flat faces.
        
        part1_name = f"{original_filename_base}_angled_cut{i+1}_p1"
        is_valid1, msgs1 = validate_part(part1, part1_name) 
        if not is_valid1:
            print(f"DEBUG_PSAC: Part 1 ({part1_name}) from cut {i+1} is NOT valid. Reasons: {'; '.join(msgs1)}", flush=True)
            continue

        part2_name = f"{original_filename_base}_angled_cut{i+1}_p2"
        is_valid2, msgs2 = validate_part(part2, part2_name)
        if not is_valid2:
            print(f"DEBUG_PSAC: Part 2 ({part2_name}) from cut {i+1} is NOT valid. Reasons: {'; '.join(msgs2)}", flush=True)
            continue

        # If both parts are valid
        print(f"DEBUG_PSAC: Cut {i+1} with normal {plane_normal} successful. Produced 2 valid pieces.", flush=True)
        # print(f"DEBUG_PSAC: Part 1 ({part1_name}) validation: {'; '.join(msgs1)}", flush=True) # Redundant if validate_part prints
        # print(f"DEBUG_PSAC: Part 2 ({part2_name}) validation: {'; '.join(msgs2)}", flush=True) # Redundant
        
        # REMOVED file export logic
        return [part1, part2] # Return the mesh objects
        # REMOVED exception handling for export and cleanup logic
            # continue 

    print(f"DEBUG_PSAC: No successful standard angled cut found for {original_filename_base} after trying all predefined normals.", flush=True)
    return []

# def _evaluate_candidate_cut(original_mesh_to_slice, plane_normal, plane_origin, cut_description, current_depth):
# ... existing code ...

# NEW FUNCTION: process_mesh_object
def process_mesh_object(
    input_mesh: trimesh.Trimesh, 
    original_filename_base: str, 
    run_three_wall_flag: bool
) -> Tuple[List[trimesh.Trimesh], str]:
    """
    Processes a mesh object by attempting cuts and returns the pieces in their original positions.
    Returns a list of mesh objects and a string indicating the type of cut performed ("three_wall", "angled", "none").
    """
    print(f"DEBUG_PMO: process_mesh_object started for {original_filename_base}", flush=True)
    
    if input_mesh is None or input_mesh.is_empty:
        print("DEBUG_PMO: Input mesh is None or empty. Returning original.", flush=True)
        return [input_mesh.copy() if input_mesh else None], "none"

    processed_mesh = input_mesh.copy()

    # 1. Align mesh to its OBB and store inverse rotation
    # Ensure the mesh has volume; OBB might be problematic for flat/degenerate meshes
    if processed_mesh.volume < 1e-6: # Check for very small or zero volume
        print(f"DEBUG_PMO: Mesh {original_filename_base} has negligible volume ({processed_mesh.volume}). Skipping OBB alignment and translation.", flush=True)
        # Fallback: use identity transforms if OBB is not reliable
        rotation_matrix_obb = np.eye(4)
        translation_matrix = np.eye(4)
    else:
        try:
            obb = processed_mesh.bounding_box_oriented
            rotation_matrix_obb = np.eye(4)
            # The obb.primitive.transform's rotation part aligns the OBB's axes with the world axes.
            # Applying this rotation to the mesh achieves the desired alignment.
            rotation_matrix_obb[:3, :3] = obb.primitive.transform[:3, :3]
            processed_mesh.apply_transform(rotation_matrix_obb)
            print(f"DEBUG_PMO: Mesh aligned to OBB. Applied rotation for {original_filename_base}.", flush=True)

            # 2. Translate mesh so its min corner (of the now OBB-aligned mesh) is at the origin
            min_bounds_aligned = processed_mesh.bounds[0]
            translation_to_origin_vec = -min_bounds_aligned
            translation_matrix = trimesh.transformations.translation_matrix(translation_to_origin_vec)
            processed_mesh.apply_transform(translation_matrix)
            print(f"DEBUG_PMO: Mesh translated by {translation_to_origin_vec} for {original_filename_base}. New min bounds: {processed_mesh.bounds[0]}", flush=True)
        except Exception as e:
            print(f"ERROR_PMO: Failed during OBB alignment/translation for {original_filename_base}: {e}. Using identity transforms.", flush=True)
            # Fallback to identity transforms if OBB processing fails
            processed_mesh = input_mesh.copy() # Start fresh with a copy
            rotation_matrix_obb = np.eye(4)
            translation_matrix = np.eye(4)


    # Combined forward transform M_forward = translation_matrix @ rotation_matrix_obb
    # Inverse transform M_inverse = np.linalg.inv(rotation_matrix_obb) @ np.linalg.inv(translation_matrix)
    # (Order: inv(R) then inv(T) because T was applied to an already R-transformed mesh)
    try:
        inv_rotation = np.linalg.inv(rotation_matrix_obb)
        inv_translation = np.linalg.inv(translation_matrix)
        M_inverse_transform = inv_rotation @ inv_translation
    except np.linalg.LinAlgError:
        print(f"ERROR_PMO: Could not compute inverse transform for {original_filename_base}. Using identity.", flush=True)
        M_inverse_transform = np.eye(4)


    # 3. Ensure facets are computed on the processed_mesh
    try:
        _ = processed_mesh.facets 
        if not hasattr(processed_mesh, 'facets_normal') or processed_mesh.facets_normal is None or len(processed_mesh.facets_normal) == 0:
             print(f"DEBUG_PMO: Facets or facet_normals not available on preprocessed mesh {original_filename_base} after attempting computation.", flush=True)
    except Exception as e:
        print(f"ERROR_PMO: Could not ensure/compute facets on preprocessed mesh {original_filename_base}: {e}", flush=True)
        # Depending on severity, might return original mesh here
        # For now, let cutting functions attempt and potentially fail

    # 4. Cutting logic
    resulting_pieces = []
    cut_type = "none"

    auto_detected_three_wall = False
    if not run_three_wall_flag:
        print(f"DEBUG_PMO: run_three_wall_flag is FALSE. Auto-detecting three-wall candidate for {original_filename_base}.", flush=True)
        if detect_three_wall_corner_candidate(processed_mesh): # Using default params
            print(f"DEBUG_PMO: Auto-detected {original_filename_base} as three-wall candidate.", flush=True)
            auto_detected_three_wall = True
        else:
            print(f"DEBUG_PMO: Not auto-detected {original_filename_base} as three-wall candidate.", flush=True)
    
    should_run_three_wall_cuts = run_three_wall_flag or auto_detected_three_wall

    if should_run_three_wall_cuts:
        print(f"DEBUG_PMO: Attempting three-wall corner cuts for {original_filename_base}", flush=True)
        pieces_from_cut = perform_three_wall_corner_cuts(processed_mesh, original_filename_base)
        if pieces_from_cut: # perform_three_wall_corner_cuts returns list of 3 or empty
            print(f"DEBUG_PMO: Three-wall cutting for {original_filename_base} successful. Found {len(pieces_from_cut)} pieces.", flush=True)
            resulting_pieces = pieces_from_cut
            cut_type = "three_wall"
        else:
            print(f"DEBUG_PMO: Three-wall cutting for {original_filename_base} did not produce validated pieces.", flush=True)
    
    if not resulting_pieces: 
        print(f"DEBUG_PMO: Three-wall cuts not successful or not run for {original_filename_base}. Proceeding to standard angled cutting.", flush=True)
        pieces_from_cut = perform_standard_angled_cut(processed_mesh, original_filename_base)
        if pieces_from_cut: # perform_standard_angled_cut returns list of 2 or empty
            print(f"DEBUG_PMO: Standard angled cutting for {original_filename_base} successful. Found {len(pieces_from_cut)} pieces.", flush=True)
            resulting_pieces = pieces_from_cut
            cut_type = "angled"
        else:
            print(f"DEBUG_PMO: Standard angled cutting for {original_filename_base} did not produce validated pieces.", flush=True)

    # 5. Postprocess: Transform pieces back and return
    final_output_meshes = []
    if resulting_pieces:
        for piece in resulting_pieces:
            if piece and not piece.is_empty: 
                piece_copy = piece.copy() 
                piece_copy.apply_transform(M_inverse_transform)
                final_output_meshes.append(piece_copy)
        
        if not final_output_meshes: # All pieces were empty or failed transform
             print(f"DEBUG_PMO: Cutting for {original_filename_base} (type: {cut_type}) resulted in no substantial pieces after transformation. Returning original.", flush=True)
             final_output_meshes = [input_mesh.copy()] 
             cut_type = "none" 
        else:
             print(f"DEBUG_PMO: Returning {len(final_output_meshes)} pieces for {original_filename_base}, transformed back. Cut type: {cut_type}", flush=True)
    else: 
        print(f"DEBUG_PMO: No cuts made or no pieces resulted for {original_filename_base}. Returning original mesh.", flush=True)
        final_output_meshes = [input_mesh.copy()] 
        cut_type = "none"
        
    return final_output_meshes, cut_type

def main():
    print("DEBUG_MAIN: main() function started.", flush=True)
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
    base_output_dir_for_file = os.path.join(OUTPUT_DIR, original_filename_base) 
    if not os.path.exists(base_output_dir_for_file):
        os.makedirs(base_output_dir_for_file)
        print(f"DEBUG_MAIN: Created base output directory: {base_output_dir_for_file}", flush=True)

    try:
        initial_mesh = trimesh.load_mesh(args.input_stl) # RENAMED to initial_mesh
        print(f"DEBUG_MAIN: Mesh {args.input_stl} loaded successfully. Volume: {initial_mesh.volume if hasattr(initial_mesh, 'volume') else 'N/A'}", flush=True)
    except Exception as e:
        print(f"ERROR_MAIN: Failed to load STL file {args.input_stl}: {e}", flush=True)
        return

    if initial_mesh.is_empty:
        print(f"ERROR_MAIN: Loaded mesh {args.input_stl} is empty. Cannot proceed.", flush=True)
        return

    # REMOVED: OBB alignment, translation, facet computation (moved to process_mesh_object)
    # REMOVED: auto_detected_three_wall_candidate logic (moved to process_mesh_object)
    # REMOVED: should_run_three_wall_cuts logic (moved to process_mesh_object)
    # REMOVED: Direct calls to perform_three_wall_corner_cuts and perform_standard_angled_cut

    # Call the new processing function
    list_of_meshes, cut_type_str = process_mesh_object(
        initial_mesh, 
        original_filename_base, 
        args.run_three_wall_corner_cuts
    )

    # Save the resulting meshes
    if list_of_meshes:
        output_subdir_name = ""
        piece_label_prefix = original_filename_base
        
        if cut_type_str == "three_wall":
            output_subdir_name = f"{original_filename_base}_corner_cuts"
            piece_label_prefix += "_corner_piece"
        elif cut_type_str == "angled":
            output_subdir_name = f"{original_filename_base}_angled_cuts"
            piece_label_prefix += "_angled_piece"
        elif cut_type_str == "none":
            # For "none", files go into base_output_dir_for_file
            piece_label_prefix += "_no_cuts_applied"


        if output_subdir_name: 
            specific_output_dir = os.path.join(base_output_dir_for_file, output_subdir_name)
        else: 
            specific_output_dir = base_output_dir_for_file

        if not os.path.exists(specific_output_dir):
            os.makedirs(specific_output_dir)
            print(f"DEBUG_MAIN: Created output directory: {specific_output_dir}", flush=True)

        if not list_of_meshes: # Should not happen if process_mesh_object guarantees a list
             print(f"DEBUG_MAIN: No meshes returned by process_mesh_object for {original_filename_base}, nothing to save.", flush=True)
             return # Added return


        for i, part_mesh in enumerate(list_of_meshes):
            if part_mesh is None or part_mesh.is_empty:
                print(f"DEBUG_MAIN: Skipping save for empty/None mesh part {i+1} for {original_filename_base}.", flush=True)
                continue

            # If only one piece and type is "none", it's the original (or copy)
            if cut_type_str == "none" and len(list_of_meshes) == 1:
                 filename = f"{piece_label_prefix}.stl" # e.g. original_filename_base_no_cuts_applied.stl
            else: # Multiple pieces or specific cut type with single piece (though unlikely for cuts)
                 filename = f"{piece_label_prefix}_{i+1}.stl"
            
            filepath = os.path.join(specific_output_dir, filename)
            try:
                part_mesh.export(filepath)
                print(f"DEBUG_MAIN: Saved mesh to {filepath}", flush=True)
            except Exception as e_export:
                print(f"ERROR_MAIN: Failed to export mesh {filepath}: {e_export}", flush=True)
    else:
        print(f"DEBUG_MAIN: No meshes returned by process_mesh_object for {original_filename_base}, nothing to save.", flush=True)
    
    print(f"DEBUG_MAIN: Processing finished for {args.input_stl}.", flush=True)

if __name__ == '__main__':
    print("DEBUG_PS: Script execution started (__name__ == '__main__').", flush=True)
    main()
    print("DEBUG_PS: Script execution finished.", flush=True)
