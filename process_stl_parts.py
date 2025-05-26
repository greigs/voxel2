import os
import sys
import argparse
import numpy as np
import trimesh

# Global parameters (adjust as needed)
MAX_DIM = 150.0  # Maximum dimension in mm for any side of the part
MIN_FLAT_FACE_DIMENSION_MM = 8.0 # Minimum length for an edge of the required flat face
MIN_FLAT_FACE_ASPECT_RATIO = 1.0 # Minimum aspect ratio for a face to be considered (e.g., 1.0 for square-like)
FLAT_FACE_ANGLE_TOLERANCE_DEGREES = 1.0 # Max angle deviation for normals to be considered coplanar
MAX_CUTTING_DEPTH = 10 # Maximum recursion depth for cutting
SPACING_MM = 5.0 # Default spacing for cuts if applicable elsewhere
SEPARATION_MM = 1.0 # Separation distance for cut pieces
FLAT_FACE_NORMAL_TOLERANCE = 1e-3 
FLAT_FACE_FILL_RATIO = 0.80 # Relaxed from 0.85
NORMAL_SIMILARITY_TOLERANCE = 1e-4 # For comparing normals to find unique ones
MAX_DIM_FOR_MULTIAXIS_OVERSIZE_MM = 8.0 # New rule: max dimension if more than two axes exceed this
OUTPUT_DIR = "" # Initialize globally, will be set in main

def _evaluate_candidate_cut(original_mesh_to_slice, plane_normal, plane_origin, cut_description, current_depth):
    """
    Helper to slice a mesh, validate parts, and score the cut.
    Returns: (score, part1, part2)
    Score: -1 if slice failed or parts invalid, otherwise sum of flat face areas.
    """
    p1_candidate, p2_candidate = None, None
    try:
        # Always use a fresh copy of the original mesh for slicing
        # original_mesh_to_slice is the mesh state before this specific candidate cut
        mesh_for_slicing = original_mesh_to_slice.copy()
        # process=True on load, but re-processing after copy or if it's a part from previous cut
        mesh_for_slicing.process() 
        if not mesh_for_slicing.is_watertight:
            mesh_for_slicing.fill_holes()
            # if not mesh_for_slicing.is_watertight:
            #     print(f"    DEBUG (depth {current_depth}, {cut_description}): Mesh for slicing still not watertight.")

        p1_candidate = trimesh.intersections.slice_mesh_plane(
            mesh=mesh_for_slicing, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)
        p2_candidate = trimesh.intersections.slice_mesh_plane(
            mesh=mesh_for_slicing, plane_normal=-plane_normal, plane_origin=plane_origin, cap=True)
    except Exception as e:
        # print(f"    DEBUG: Error during slice_mesh_plane ({cut_description}, depth {current_depth}): {e}")
        return -1, None, None

    valid_p1 = p1_candidate and not p1_candidate.is_empty and hasattr(p1_candidate, 'volume') and p1_candidate.volume > 1e-6 and len(p1_candidate.faces) >=4
    valid_p2 = p2_candidate and not p2_candidate.is_empty and hasattr(p2_candidate, 'volume') and p2_candidate.volume > 1e-6 and len(p2_candidate.faces) >=4

    if valid_p1 and valid_p2:
        vol_original_for_slice = mesh_for_slicing.volume
        vol_p1 = p1_candidate.volume
        vol_p2 = p2_candidate.volume
        
        is_p1_original = np.allclose(vol_p1, vol_original_for_slice, rtol=1e-3, atol=1e-3)
        is_p2_original = np.allclose(vol_p2, vol_original_for_slice, rtol=1e-3, atol=1e-3)

        if not is_p1_original and not is_p2_original and \
           np.allclose(vol_p1 + vol_p2, vol_original_for_slice, rtol=0.05, atol=1e-3): # Added atol for small volumes
            
            score_p1 = has_required_flat_face(p1_candidate)
            score_p2 = has_required_flat_face(p2_candidate)
            current_score = score_p1 + score_p2
            # print(f"      DEBUG Cut {cut_description}: Valid geom, Score={current_score:.2f} (P1 area: {score_p1:.2f}, P2 area: {score_p2:.2f})")
            return current_score, p1_candidate, p2_candidate
        # else:
            # print(f"      DEBUG Cut {cut_description}: Invalid geom (vol issue: orig={vol_original_for_slice:.3f}, p1={vol_p1:.3f}, p2={vol_p2:.3f} or same as original)")
            # return -1, None, None # Invalid geometric cut (didn't split properly)
    # else:
        # print(f"      DEBUG Cut {cut_description}: One or both parts empty/invalid after slice.")
        pass # Fall through to return -1, None, None
    return -1, None, None


def find_flat_area_trimesh_vertices(mesh, min_area_mm2, angle_tolerance_rad):
    print(f"DEBUG_PS: find_flat_area_trimesh_vertices called. min_area_mm2={min_area_mm2:.2f}, angle_tolerance_rad={angle_tolerance_rad:.4f}, mesh_faces={len(mesh.faces) if mesh else 0}")
    if not mesh or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print(f"DEBUG_PS: Mesh is empty or invalid.")
        return []

    # Ensure mesh is processed, helps with properties like face_normals and adjacency graph
    mesh.process()

    if not mesh.is_watertight:
        # print("Warning: Mesh is not watertight, attempting to fix.")
        pass # Proceeding even if not watertight

    all_flat_faces_vertices = []
    
    try:
        face_normals = mesh.face_normals
        adjacency = mesh.face_adjacency # List of (face_idx_A, face_idx_B) pairs

        graph_edges = []
        if len(face_normals) > 0 and len(adjacency) > 0: # Proceed only if there are faces and adjacencies
            for i, j in adjacency:
                # Adjacency might contain -1 for boundary edges if mesh is not watertight / has boundaries
                if i < 0 or j < 0 or i >= len(face_normals) or j >= len(face_normals):
                    continue
                try:
                    # Corrected: trimesh.geometry.vector_angle
                    angle = trimesh.geometry.vector_angle([face_normals[i], face_normals[j]])
                    # vector_angle returns an array of angles if multiple pairs are passed.
                    # Here we pass two vectors, expect a single angle value.
                    # It expects a list of vector pairs, or a (2, 3) array for two vectors.
                    # Let's pass them as a list of two vectors.
                    angle_between_normals = trimesh.geometry.vector_angle((face_normals[i], face_normals[j]))

                    if angle_between_normals <= angle_tolerance_rad:
                        graph_edges.append(tuple(sorted((i, j)))) 
                except IndexError: 
                    print(f"DEBUG_PS: IndexError accessing face_normals for adjacency pair ({i}, {j}). Max face index: {len(face_normals)-1}")
                    continue
                except Exception as e_angle:
                    print(f"DEBUG_PS: Error calculating angle for faces {i},{j}: {e_angle}")
                    continue
        
        # Remove duplicate edges if any (though sorted tuples should handle most from simple adjacency)
        graph_edges = list(set(graph_edges))

        print(f"DEBUG_PS: Built graph with {len(graph_edges)} edges connecting faces within angle tolerance.")

        # Get connected components based on these graph edges.
        # Nodes are all face indices. If graph_edges is empty, each face becomes its own component.
        if len(mesh.faces) > 0:
            clusters = trimesh.graph.connected_components(edges=graph_edges, nodes=np.arange(len(mesh.faces)))
        else:
            clusters = []
            
    except Exception as e:
        print(f"DEBUG_PS: Error during manual face clustering graph construction: {e}")
        return []
        
    print(f"DEBUG_PS: Found {len(clusters)} face clusters using manual graph method.")

    for i, face_indices_in_cluster in enumerate(clusters):
        # Corrected check for non-empty cluster (NumPy array)
        if not isinstance(face_indices_in_cluster, np.ndarray) or face_indices_in_cluster.size == 0:
            # print(f"DEBUG_PS: Cluster {i+1} is empty or not a numpy array, skipping.")
            continue
        
        # print(f"DEBUG_PS: Processing cluster {i+1}/{len(clusters)} with {len(face_indices_in_cluster)} faces.")
        
        try:
            # Create a submesh from these faces that are roughly coplanar and connected by our graph logic
            # Ensure face_indices are valid for the mesh
            valid_face_indices = [idx for idx in face_indices_in_cluster if 0 <= idx < len(mesh.faces)]
            if not valid_face_indices:
                # print(f"DEBUG_PS: Cluster {i+1} had no valid face indices after filtering.")
                continue

            cluster_submesh = mesh.submesh([valid_face_indices], append=True, repair=False) 
        except Exception as e:
            print(f"DEBUG_PS: Error creating submesh for cluster {i+1}: {e}. Face indices: {face_indices_in_cluster}")
            continue

        if not cluster_submesh or len(cluster_submesh.faces) == 0:
            # print(f"DEBUG_PS: Cluster {i+1} resulted in an empty submesh.")
            continue

        # The cluster_submesh might still contain disconnected components if the graph logic
        # connected things that are not immediately geometrically connected after submeshing.
        # Split() finds truly connected components within this submesh.
        try:
            components = cluster_submesh.split(only_watertight=False) 
        except Exception as e:
            print(f"DEBUG_PS: Error splitting cluster_submesh {i+1}: {e}")
            continue
        
        # print(f"DEBUG_PS: Cluster {i+1} (from {len(face_indices_in_cluster)} faces) split into {len(components)} connected components.")

        for j, component_mesh in enumerate(components):
            if not component_mesh or len(component_mesh.faces) == 0:
                continue

            component_mesh.process() # Ensure properties like area are computed

            # print(f"DEBUG_PS: Cluster {i+1}, Component {j+1}: Area: {component_mesh.area:.2f} mm^2 (required min_area_mm2: {min_area_mm2:.2f})")
            if component_mesh.area >= min_area_mm2:
                try:
                    outline_entity = component_mesh.outline() 
                    
                    if outline_entity is not None and hasattr(outline_entity, 'vertices') and len(outline_entity.vertices) > 2:
                        paths = outline_entity.discrete
                        if not paths:
                            print(f"DEBUG_PS: C{i+1}/Comp{j+1}: No discrete paths from outline. Area: {component_mesh.area:.2f}")
                            continue

                        longest_path_vertices_3d = max(paths, key=len)

                        if len(longest_path_vertices_3d) < 3: 
                            print(f"DEBUG_PS: C{i+1}/Comp{j+1}: Longest path has < 3 vertices. Area: {component_mesh.area:.2f}")
                            continue
                        
                        print(f"DEBUG_PS: C{i+1}/Comp{j+1}: Found potential flat area. Area: {component_mesh.area:.2f}. Boundary vertices: {len(longest_path_vertices_3d)}. Adding.")
                        all_flat_faces_vertices.append(longest_path_vertices_3d)
                    else:
                        print(f"DEBUG_PS: C{i+1}/Comp{j+1}: Outline not suitable or too few vertices. Area: {component_mesh.area:.2f}. Outline vertices: {len(outline_entity.vertices) if outline_entity and hasattr(outline_entity, 'vertices') else 'None/Invalid'}")
                except Exception as e:
                    print(f"DEBUG_PS: C{i+1}/Comp{j+1}: Error getting outline (Area: {component_mesh.area:.2f}): {e}")
                    pass
            # else:
                # print(f"DEBUG_PS: C{i+1}/Comp{j+1}: Component area {component_mesh.area:.2f} is less than min_area_mm2 {min_area_mm2:.2f}")

    print(f"DEBUG_PS: find_flat_area_trimesh_vertices returning {len(all_flat_faces_vertices)} flat face candidates.")
    return all_flat_faces_vertices

def check_rectangular_face(flat_face_vertices, min_dim_mm, aspect_ratio):
    # aspect_ratio here is MIN_FLAT_FACE_ASPECT_RATIO, used to ensure the other side is not too small
    print(f"DEBUG_PS: check_rectangular_face called with {len(flat_face_vertices)} vertices. min_dim_mm={min_dim_mm}, aspect_ratio_for_other_side={aspect_ratio}")
    if len(flat_face_vertices) < 3:
        print(f"DEBUG_PS: Not enough vertices ({len(flat_face_vertices)}) for a polygon.")
        return False

    # Using Shapely's Polygon for geometric checks
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity

    try:
        polygon = Polygon(flat_face_vertices)
        if not polygon.is_valid:
            print(f"DEBUG_PS: Polygon created from {len(flat_face_vertices)} vertices is not valid: {explain_validity(polygon)}")
            # Attempt to simplify or buffer slightly if invalid due to self-intersection or floating point issues
            polygon = polygon.buffer(0) # Buffer by 0 can sometimes fix minor validity issues
            if not polygon.is_valid:
                print(f"DEBUG_PS: Polygon still invalid after buffer(0).")
                return False
        
        # Get the oriented bounding box (OBB)
        # For Shapely < 2.0, this might be minimum_rotated_rectangle
        # For Shapely >= 2.0, it's oriented_bounding_box
        if hasattr(polygon, 'oriented_bounding_box'):
            obb_polygon = polygon.oriented_bounding_box 
        elif hasattr(polygon, 'minimum_rotated_rectangle'): # Compatibility for older Shapely
            obb_polygon = polygon.minimum_rotated_rectangle
        else:
            print("DEBUG_PS: Neither oriented_bounding_box nor minimum_rotated_rectangle found on polygon object.")
            # Fallback to envelope (Axis-Aligned Bounding Box) - less accurate for rotated faces
            minx, miny, maxx, maxy = polygon.bounds
            dims = sorted([maxx - minx, maxy - miny])
            print(f"DEBUG_PS: Using AABB dimensions: {dims[0]:.2f} x {dims[1]:.2f} mm (less accurate)")
            result = dims[0] >= min_dim_mm and dims[1] >= min_dim_mm # For 8x8, both sides must be >= 8
            print(f"DEBUG_PS: AABB Check result: {result} (dims[0]={dims[0]:.2f}, dims[1]={dims[1]:.2f} vs min_dim={min_dim_mm})")
            return result

        # OBB is a polygon, get its exterior coordinates and calculate extents
        obb_coords = np.array(obb_polygon.exterior.coords)
        
        # Calculate edge lengths of the OBB
        edge_lengths = [np.linalg.norm(obb_coords[i] - obb_coords[i+1]) for i in range(len(obb_coords)-1)]
        
        # OBB should have 4 distinct edge lengths in pairs (e.g., w, h, w, h)
        # We need the width and height.
        if len(edge_lengths) < 4: # Should be 5 points for 4 segments in a closed loop
            print(f"DEBUG_PS: OBB has unexpected number of edge lengths: {len(edge_lengths)}")
            return False

        # Get unique lengths, should be 2 for a rectangle
        unique_lengths = sorted(list(set(np.round(edge_lengths, decimals=3)))) # Round to handle float precision
        
        if len(unique_lengths) == 1: # OBB is a square (all sides of OBB are approx equal)
             dims = [unique_lengths[0], unique_lengths[0]]
        elif len(unique_lengths) >= 2 : # OBB is a rectangle. Take the two smallest unique lengths as dimensions.
                                       # For a true rectangular OBB, unique_lengths should ideally have 2 values.
             dims = unique_lengths[:2]
        else: # len(unique_lengths) == 0. This implies an issue (e.g. OBB has no area or edges).
            print(f"DEBUG_PS: OBB has problematic unique edge lengths: {unique_lengths}. OBB area: {obb_polygon.area:.4f}")
            # Fallback: use shapely's OBB extents if available (newer shapely might provide this directly)
            # This part of trimesh's OBB was `obb.extents`
            # For shapely's polygon.oriented_bounding_box, we need to calculate extents.
            # The `minimum_rotated_rectangle` in older shapely is a polygon.
            # A common way to get extents from an OBB polygon:
            # 1. Get the points of the OBB polygon.
            # 2. Calculate the length of two adjacent sides.
            # This is what `edge_lengths` and `unique_lengths` tries to do.
            # If `obb_polygon.area` is near zero, it's degenerate.
            if obb_polygon.area < 1e-3: # Effectively zero area
                print(f"DEBUG_PS: OBB area is near zero ({obb_polygon.area:.4f}), confirming as not a valid rectangle.")
            # else: # This path is if unique_lengths is empty but area is not zero - very unlikely with current edge_lengths calc.
            #    print(f"DEBUG_PS: Could not reliably determine OBB dimensions from unique_lengths: {unique_lengths}, but area is {obb_polygon.area:.4f}")
            return False


    except Exception as e:
        print(f"DEBUG_PS: Error in check_rectangular_face during polygon processing or OBB: {e}")
        return False
    
    print(f"DEBUG_PS: OBB dimensions from unique_lengths: {dims[0]:.2f} x {dims[1]:.2f} mm")
    
    # The problem states "at least one flat rectangular face of 8mm x 8mm".
    # This means MIN_FLAT_FACE_DIMENSION_MM (8mm) should apply to both sides of the rectangle found.
    # So, dims[0] >= 8 and dims[1] >= 8.
    result = dims[0] >= min_dim_mm and dims[1] >= min_dim_mm 
    
    print(f"DEBUG_PS: Check result: {result} (dims[0]={dims[0]:.2f} >= {min_dim_mm} AND dims[1]={dims[1]:.2f} >= {min_dim_mm})")
    return result

def has_required_flat_face(part_mesh, part_id, output_dir):
    """
    Checks if a mesh has at least one flat, sufficiently large, filled rectangular face.
    Returns the area of the largest qualifying facet, or 0.0 if none.
    """
    print(f"DEBUG_PS: has_required_flat_face called for part_id: {part_id}, mesh faces: {len(part_mesh.faces) if part_mesh else 0}")
    if not part_mesh or len(part_mesh.faces) == 0:
        print(f"DEBUG_PS: Part {part_id}: Mesh is empty. Cannot have flat face.")
        return False

    min_area_for_flat_face = MIN_FLAT_FACE_DIMENSION_MM**2 # Area of an 8x8 square is 64. Aspect ratio considered in check_rectangular_face.
                                                          # The original was MIN_FLAT_FACE_DIMENSION_MM**2 / MIN_FLAT_FACE_ASPECT_RATIO
                                                          # If aspect ratio is 1, this is the same. Let's keep it consistent.
    min_area_for_flat_face = (MIN_FLAT_FACE_DIMENSION_MM ** 2) / MIN_FLAT_FACE_ASPECT_RATIO


    flat_faces_vertices = find_flat_area_trimesh_vertices(
        part_mesh,
        min_area_mm2=min_area_for_flat_face, 
        angle_tolerance_rad=np.deg2rad(FLAT_FACE_ANGLE_TOLERANCE_DEGREES)
    )
    print(f"DEBUG_PS: Part {part_id}: Found {len(flat_faces_vertices)} potential flat face candidates from find_flat_area_trimesh_vertices.")

    if not flat_faces_vertices:
        # print(f"Part {part_id}: No sufficiently large flat areas found by find_flat_area_trimesh_vertices.")
        # This message is effectively covered by the main error message if it returns False
        pass # No need for extra print here, the count above is enough

    found_valid_face = False
    for i, vertices_3d in enumerate(flat_faces_vertices):
        print(f"DEBUG_PS: Part {part_id}: Checking flat face candidate #{i+1} with {len(vertices_3d)} 3D vertices.")
        if len(vertices_3d) < 3:
            print(f"DEBUG_PS: Part {part_id}: Candidate #{i+1} has < 3 vertices, skipping.")
            continue

        try:
            # Use trimesh.points.plane_fit to get centroid (origin) and normal
            plane_origin, plane_normal = trimesh.points.plane_fit(vertices_3d)
            
            norm_mag = np.linalg.norm(plane_normal)
            if norm_mag < 1e-6: 
                print(f"DEBUG_PS: Part {part_id}: Candidate #{i+1} - degenerate normal from plane_fit. Skipping.")
                continue
            # plane_normal should already be unit from plane_fit

            # Transform to 2D
            # polygon_vertices_2d, T = trimesh.geometry.to_2D(vertices_3d, plane_normal=plane_normal, plane_origin=plane_origin)
            # Attempt to use project_to_plane instead of to_2D
            # polygon_vertices_2d = trimesh.geometry.project_to_plane(
            #     points=vertices_3d,
            #     plane_normal=plane_normal,
            #     plane_origin=plane_origin,
            #     return_transform=False # Ensure we only get points
            # )
            # project_to_plane returns (n,2) points, no separate transform T by default

            # Try trimesh.points.project_to_plane
            polygon_vertices_2d = trimesh.points.project_to_plane(
                points=vertices_3d,
                plane_normal=plane_normal,
                plane_origin=plane_origin,
                return_transform=False,
                return_planar=True # Ensure we get (n,2) points directly
            )

        except AttributeError as ae:
             if 'plane_fit' in str(ae):
                print(f"DEBUG_PS: Part {part_id}: Candidate #{i+1} - 'plane_fit' not found in trimesh.points. Error: {ae}")
             elif 'best_fit_plane' in str(ae):
                print(f"DEBUG_PS: Part {part_id}: Candidate #{i+1} - 'best_fit_plane' previously failed. Error: {ae}")
             else:
                print(f"DEBUG_PS: Part {part_id}: Candidate #{i+1} - AttributeError during 3D to 2D projection: {ae}")
             continue
        except Exception as e:
            print(f"DEBUG_PS: Part {part_id}: Candidate #{i+1} - Error during 3D to 2D projection (plane_fit): {e}")
            continue
            
        print(f"DEBUG_PS: Part {part_id}: Candidate #{i+1} projected to {len(polygon_vertices_2d)} 2D vertices.")

        if check_rectangular_face(polygon_vertices_2d, MIN_FLAT_FACE_DIMENSION_MM, MIN_FLAT_FACE_ASPECT_RATIO):
            print(f"DEBUG_PS: Part {part_id}: Found valid rectangular face for candidate #{i+1}.")
            # Save the successful part mesh for inspection (optional)
            # flat_face_path = os.path.join(output_dir, f"{part_id}_flat_face_candidate_{i+1}.stl")
            # temp_mesh_from_vertices = trimesh.Trimesh(vertices=vertices_3d, faces=[list(range(len(vertices_3d)))]) # This is not a valid mesh face
            # print(f"DEBUG_PS: Note: Saving of flat face candidate mesh is illustrative, not a proper mesh.")
            found_valid_face = True
            break # Found one, no need to check others for this part
        # else: # This else is too verbose if many candidates are checked
            # print(f"DEBUG_PS: Part {part_id}: Candidate #{i+1} did not meet rectangular criteria after 2D projection.")

    print(f"DEBUG_PS: Part {part_id}: has_required_flat_face final result: {found_valid_face}")
    return found_valid_face

def validate_part(mesh, part_name, max_dim=MAX_DIM, min_flat_face_dim=MIN_FLAT_FACE_DIMENSION_MM):
    """
    Validates a single part based on dimensions and flat face criteria.
    """
    messages = []
    is_valid_overall = True
    
    # Ensure mesh is not None and has faces before proceeding
    if mesh is None or len(mesh.faces) == 0:
        messages.append(f"Part {part_name}: Mesh is empty or invalid.")
        return False, messages

    # Use a fresh copy for validation if modifications are made (e.g. processing)
    # For this validation, we mostly read properties, but processing can alter.
    # If has_required_flat_face or other checks modify, this is important.
    # mesh.process() is called within has_required_flat_face, so it's handled there.
    mesh_to_process = mesh # Use the passed mesh directly for now

    # Check dimensions
    bounds = mesh_to_process.bounds
    dims = bounds[1] - bounds[0]
    if any(d > max_dim for d in dims):
        messages.append(f"Part {part_name}: Exceeds max dimension ({max_dim}mm). Dimensions: {dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f}mm")
        is_valid_overall = False

    # Check for at least one flat face (e.g., >= 8x8mm)
    min_single_face_area_for_proxy = min_flat_face_dim * min_flat_face_dim 

    if hasattr(mesh_to_process, 'facets_area'):
        # Using min_single_face_area_for_proxy for individual facet check as a loose proxy
        dominant_facets_indices = [i for i, area in enumerate(mesh_to_process.facets_area) if area >= min_single_face_area_for_proxy] # Corrected variable
        if not dominant_facets_indices:
            pass 
    else:
        pass

    # The primary validation for the flat face:
    if not has_required_flat_face(mesh_to_process, part_name, "dummy_output_dir_for_validation_debug"):
        messages.append(f"Part {part_name}: Does not have a required >= {min_flat_face_dim:.1f}x{min_flat_face_dim:.1f}mm flat rectangular face.")
        is_valid_overall = False
        
    if not is_valid_overall:
        print(f"DEBUG_PS: Validation FAILED for {part_name}: {'; '.join(messages)}")
    else:
        print(f"DEBUG_PS: Validation PASSED for {part_name}.")

    return is_valid_overall, messages

def apply_separation_to_mesh_pieces(pieces, cut_axis, separation_distance):
    """Applies separation to two mesh pieces along a cut axis."""
    if len(pieces) != 2 or not pieces[0] or not pieces[1]:
        return pieces # Expecting two pieces

    # Apply translation
    translation_vector1 = np.zeros(3)
    translation_vector1[cut_axis] = -separation_distance / 2.0
    pieces[0].apply_translation(translation_vector1)

    translation_vector2 = np.zeros(3)
    translation_vector2[cut_axis] = separation_distance / 2.0
    pieces[1].apply_translation(translation_vector2)
    return pieces

def recursively_cut_mesh(mesh, max_dim, current_depth=0):
    """
    Recursively cuts a mesh if it exceeds max_dim OR if more than two dimensions exceed MAX_DIM_FOR_MULTIAXIS_OVERSIZE_MM.
    """
    print(f"DEBUG_PS: recursively_cut_mesh called. Depth: {current_depth}, Mesh faces: {len(mesh.faces) if mesh else 'None'}")
    if not mesh or not mesh.is_volume:
        print(f"DEBUG_PS: Mesh is None or not a volume at depth {current_depth}. Returning empty list.")
        return []
    
    if current_depth > MAX_CUTTING_DEPTH:
        print(f"DEBUG_PS: Max cutting depth {MAX_CUTTING_DEPTH} reached. Returning current mesh as a single part.")
        return [mesh]

    bounds = mesh.bounds
    dims = bounds[1] - bounds[0]
    
    # Rule 1: Check against the overall max dimension (e.g., 150mm, passed as max_dim)
    dims_exceeding_overall_max_indices = [i for i, dim_val in enumerate(dims) if dim_val > max_dim]
    
    # Rule 2: Check how many dimensions exceed MAX_DIM_FOR_MULTIAXIS_OVERSIZE_MM (e.g., 8mm)
    num_dims_over_multiaxis_limit = sum(1 for d in dims if d > MAX_DIM_FOR_MULTIAXIS_OVERSIZE_MM)

    needs_cutting = False
    cut_axis = -1 # Initialize cut_axis
    cut_reason = ""

    print(f"DEBUG_PS: Info (depth {current_depth}): Mesh dimensions {dims}. Overall max_dim: {max_dim}mm. Multi-axis limit: {MAX_DIM_FOR_MULTIAXIS_OVERSIZE_MM}mm.")
    print(f"DEBUG_PS: Indices of dims exceeding overall max_dim ({max_dim}mm): {dims_exceeding_overall_max_indices}. Num dims > {MAX_DIM_FOR_MULTIAXIS_OVERSIZE_MM}mm: {num_dims_over_multiaxis_limit}.")

    if dims_exceeding_overall_max_indices:
        needs_cutting = True
        # Cut along the largest dimension that exceeds max_dim
        cut_axis = max(dims_exceeding_overall_max_indices, key=lambda i: dims[i])
        cut_reason = f"dimension {dims[cut_axis]:.2f}mm on axis {cut_axis} exceeds overall max_dim {max_dim}mm"
    elif num_dims_over_multiaxis_limit > 2:
        needs_cutting = True
        # All dimensions are <= max_dim, but 3 dimensions are > MAX_DIM_FOR_MULTIAXIS_OVERSIZE_MM.
        # Cut along the largest dimension overall.
        cut_axis = np.argmax(dims)
        cut_reason = f"{num_dims_over_multiaxis_limit} dimensions exceed {MAX_DIM_FOR_MULTIAXIS_OVERSIZE_MM}mm"
    
    if not needs_cutting:
        print(f"DEBUG_PS: No cut needed for mesh at depth {current_depth} (dims: {dims}). Conditions not met. Returning as single part.")
        return [mesh]

    # Proceed with cutting using the determined cut_axis
    print(f"DEBUG_PS: Depth {current_depth}: Cutting required due to: {cut_reason}.")
    print(f"DEBUG_PS: Selected cut_axis: {cut_axis} (dimension: {dims[cut_axis]:.2f}mm).")
    
    cut_origin = mesh.center_mass 
    cut_normal = np.zeros(3)
    cut_normal[cut_axis] = 1.0

    print(f"DEBUG_PS: Depth {current_depth}: Cutting along axis {cut_axis} (normal: {cut_normal}) at origin {cut_origin}. Dim: {dims[cut_axis]:.2f}") # Removed incorrect comparison to max_dim here

    try:
        p1_mesh = trimesh.intersections.slice_mesh_plane(
            mesh=mesh, plane_normal=cut_normal, plane_origin=cut_origin, cap=True)
        p2_mesh = trimesh.intersections.slice_mesh_plane(
            mesh=mesh, plane_normal=-cut_normal, plane_origin=cut_origin, cap=True) # Use negative normal for the other side

        # Validate the pieces before proceeding
        # A valid slice should result in two non-empty pieces whose volumes roughly sum to the original.
        # (This volume check can be tricky due to capping and potential small slivers)
        valid_p1 = p1_mesh and not p1_mesh.is_empty and hasattr(p1_mesh, 'volume') and p1_mesh.volume > 1e-6 and len(p1_mesh.faces) >=4
        valid_p2 = p2_mesh and not p2_mesh.is_empty and hasattr(p2_mesh, 'volume') and p2_mesh.volume > 1e-6 and len(p2_mesh.faces) >=4

        if not (valid_p1 and valid_p2):
            print(f"DEBUG_PS: Depth {current_depth}: Slice resulted in invalid or empty pieces. p1 valid: {valid_p1}, p2 valid: {valid_p2}. Returning original mesh.")
            # Attempt to save the failed pieces for debugging if they exist
            if p1_mesh and not p1_mesh.is_empty:
                p1_mesh.export(os.path.join(OUTPUT_DIR, f"failed_slice_p1_depth{current_depth}_axis{cut_axis}.stl"))
            if p2_mesh and not p2_mesh.is_empty:
                p2_mesh.export(os.path.join(OUTPUT_DIR, f"failed_slice_p2_depth{current_depth}_axis{cut_axis}.stl"))
            return [mesh] # Return original mesh if slice failed to produce two valid parts
        
        # Optional: Check if the slice actually split the mesh or just capped an open one
        # This can happen if the cut plane is outside or tangent in a weird way.
        if np.allclose(p1_mesh.volume, mesh.volume, rtol=1e-3) or np.allclose(p2_mesh.volume, mesh.volume, rtol=1e-3):
            print(f"DEBUG_PS: Depth {current_depth}: Slice did not significantly change mesh volume. p1_vol={p1_mesh.volume:.3f}, p2_vol={p2_mesh.volume:.3f}, orig_vol={mesh.volume:.3f}. Returning original.")
            return [mesh]
        if not np.allclose(p1_mesh.volume + p2_mesh.volume, mesh.volume, rtol=0.1, atol=1e-2): # Relaxed tolerance for volume sum
            print(f"DEBUG_PS: Depth {current_depth}: Sum of piece volumes ({p1_mesh.volume + p2_mesh.volume:.3f}) doesn't match original volume ({mesh.volume:.3f}) sufficiently. Returning original.")
            # p1_mesh.export(os.path.join(OUTPUT_DIR, f"vol_mismatch_p1_depth{current_depth}_axis{cut_axis}.stl"))
            # p2_mesh.export(os.path.join(OUTPUT_DIR, f"vol_mismatch_p2_depth{current_depth}_axis{cut_axis}.stl"))
            return [mesh]

        print(f"DEBUG_PS: Depth {current_depth}: Slice successful. p1 faces: {len(p1_mesh.faces) if p1_mesh else 0}, p2 faces: {len(p2_mesh.faces) if p2_mesh else 0}.")

        current_parts = [] # Initialize list for parts from this level of recursion
        current_parts.extend(recursively_cut_mesh(p1_mesh, max_dim, current_depth + 1))
        current_parts.extend(recursively_cut_mesh(p2_mesh, max_dim, current_depth + 1))
        return current_parts # Return the collected parts
        
    except Exception as e:
        print(f"DEBUG_PS: Error during slicing or separation at depth {current_depth}: {e}. Returning original mesh.")
        # Log the mesh that caused an error during slicing for inspection
        if mesh and OUTPUT_DIR: # Ensure OUTPUT_DIR is set
            error_mesh_path = os.path.join(OUTPUT_DIR, f"error_slice_input_depth{current_depth}_axis{cut_axis}.stl")
            try:
                mesh.export(error_mesh_path)
                print(f"DEBUG_PS: Saved mesh that caused slicing error to: {error_mesh_path}")
            except Exception as export_e:
                print(f"DEBUG_PS: Could not save error mesh: {export_e}")
        return [mesh] # Important to return the mesh in a list, as expected by the caller

    # This part should not be reached if needs_cutting is true and slicing is attempted.
    # If needs_cutting is false, it returns [mesh] earlier.
    # If slicing fails, it returns [mesh] in the except block.
    # If slicing is successful, it returns the result of recursive calls.
    return [] # Should ideally not be reached if logic is correct


def main():
    global OUTPUT_DIR # Declare OUTPUT_DIR as global to modify it

    parser = argparse.ArgumentParser(description="Process STL files to find parts with specific flat faces.")
    parser.add_argument("input_stl", help="Input STL file to process.")
    parser.add_argument("--output_dir", help="Directory to save output parts.", default="output_parts")
    parser.add_argument("--max_dim", type=float, help="Maximum dimension for cutting (default: 150.0 mm)", default=150.0)
    parser.add_argument("--min_flat_face_dim", type=float, help="Minimum flat face dimension (default: 8.0 mm)", default=8.0)
    parser.add_argument("--skip_manual_check", action="store_true", help="Skip manual inspection check for flat faces")
    
    args = parser.parse_args()

    input_stl_path = args.input_stl
    OUTPUT_DIR = args.output_dir # Set global output directory
    MAX_DIM = args.max_dim
    MIN_FLAT_FACE_DIMENSION_MM = args.min_flat_face_dim

    if not os.path.exists(input_stl_path):
        print(f"Error: Input STL file not found: {input_stl_path}")
        sys.exit(1)

    print(f"Loading STL file: {input_stl_path}")
    try:
        mesh = trimesh.load_mesh(input_stl_path, process=True) # Ensure processing on load
    except Exception as e:
        print(f"Error loading STL file: {e}")
        sys.exit(1)
    
    if not mesh.is_watertight:
        print("Warning: Initial mesh is not watertight. Attempting to fill holes.")
        mesh.fill_holes()
        if not mesh.is_watertight:
            print("Error: Could not make initial mesh watertight. Exiting.")
            sys.exit(1)

    print(f"DEBUG_PS: Original mesh center of mass: {mesh.center_mass}")
    print(f"DEBUG_PS: Original mesh extents: {mesh.extents}")
    
    # Get the transform that aligns the OBB with the axes
    # and apply its inverse to the mesh to achieve the alignment.
    # This should make the mesh's OBB axis-aligned.
    obb_transform = mesh.bounding_box_oriented.transform
    mesh.apply_transform(np.linalg.inv(obb_transform))
    # After this, the mesh.bounds should be equivalent to its OBB extents
    # and its axes should be aligned with the world axes.
    # It's also good to move the mesh to origin based on its new bounds for consistency.
    mesh.apply_translation(-mesh.bounds[0])

    print("DEBUG_PS: Mesh reoriented using OBB alignment and translated to origin.")
    print(f"DEBUG_PS: New mesh center of mass after OBB reorient and translate: {mesh.center_mass}") 
    print(f"DEBUG_PS: New mesh extents after OBB reorient and translate: {mesh.extents}")
    
    # Apply a small uniform scaling if dimensions are extremely small, to avoid precision issues
    # This is a heuristic and might need adjustment or removal depending on typical model scales
    if np.any(mesh.extents < 0.1): # If any dimension is less than 0.1mm
        print("Warning: Mesh has very small extents. Applying a small uniform scaling (x10) to potentially improve robustness.")
        mesh.apply_scale(10)


    print("Initial mesh loaded and preprocessed.")
    print(f"Initial bounding box: Min {np.round(mesh.bounds[0],3)}, Max {np.round(mesh.bounds[1],3)}")
    print(f"Initial dimensions: {np.round(mesh.extents,3)}")

    base_name = os.path.splitext(os.path.basename(input_stl_path))[0]
    output_dir = f"output_parts_{base_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output will be saved to: {output_dir}")

    print(f"Starting recursive cutting process with MAX_DIMENSION = {MAX_DIM}mm...")
    processed_parts = recursively_cut_mesh(mesh, MAX_DIM, 0)

    print(f"\\nProcessing complete. Generated {len(processed_parts)} potential parts.")
    
    final_saved_parts_count = 0
    failed_parts_count = 0

    for i, part in enumerate(processed_parts):
        part_name = f"part_{i+1}"
        print(f"--- Validating and Saving {part_name} ---")
        
        # Ensure part is processed before validation, especially if it's an original uncut mesh
        # or a part that didn't go through the full cutting path that processes sub-parts.
        if not hasattr(part, 'facets') or part.facets is None: # A simple check
            part.process()
        if not part.is_watertight: # Attempt to fix if it became non-watertight
            # print(f"  Warning: Part {part_name} is not watertight before final validation. Attempting to fill.")
            part.fill_holes()


        is_valid, validation_messages = validate_part(part, part_name, max_dim=MAX_DIM, min_flat_face_dim=MIN_FLAT_FACE_DIMENSION_MM)
        
        for msg in validation_messages:
            print(msg)

        if is_valid:
            try:
                output_stl_path = os.path.join(output_dir, f"{part_name}.stl")
                part.export(output_stl_path)
                print(f"Successfully saved: {output_stl_path}")
                
                # Post-save validation (optional, but good for sanity check)
                # loaded_check_part = trimesh.load_mesh(output_stl_path, process=True)
                # is_still_valid, check_msgs = validate_part(loaded_check_part, f"{part_name}_reloaded", MAX_DIM, MIN_FLAT_FACE_EDGE)
                # if not is_still_valid:
                #     print(f"--- {part_name} FAILED post-save validation. Issues: {check_msgs} ---")
                # else:
                #     print(f"--- {part_name} PASSED post-save validation. ---")
                final_saved_parts_count += 1
            except Exception as e:
                print(f"Error saving part {part_name}: {e}")
                failed_parts_count +=1
        else:
            failed_parts_count += 1
            try:
                failed_output_stl_path = os.path.join(output_dir, f"{part_name}_FAILED.stl")
                part.export(failed_output_stl_path)
                print(f"Saved FAILED part for inspection: {failed_output_stl_path}")
            except Exception as e:
                print(f"Error saving FAILED part {part_name}: {e}")
            print(f"--- {part_name} FAILED validation. ---")

    print(f"\\nSummary: {final_saved_parts_count} parts saved successfully to '{output_dir}'.")
    if failed_parts_count > 0:
        print(f"{failed_parts_count} parts FAILED validation or saving.")

    if final_saved_parts_count == 0 and len(processed_parts) > 0 :
        print("\\nNo parts successfully met all validation criteria.")
        print("The script attempts one pass of cutting. If validation fails for some parts,")
        print("the cutting strategy within 'recursively_cut_mesh' or the validation logic in 'has_required_flat_face'")
        print("might need further refinement for your specific STL model.")

if __name__ == "__main__":
    main()
