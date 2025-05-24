import argparse
import trimesh
import numpy as np
import pyvista as pv
import imageio
import os

def create_animation(input_stl_path, output_animation_path, duration=10, fps=20, max_explosion_scale=1.5, spin_revolutions=2):
    """
    Creates an animation of objects in an STL file exploding outwards and spinning.
    """
    # 1. Load STL and split into components
    try:
        mesh_data = trimesh.load_mesh(input_stl_path)
    except Exception as e:
        print(f"Error loading STL file '{input_stl_path}': {e}")
        return

    if isinstance(mesh_data, trimesh.Scene):
        # If the STL is already a scene with multiple bodies
        components = [geom for geom_name, geom in mesh_data.geometry.items() if isinstance(geom, trimesh.Trimesh)]
    elif isinstance(mesh_data, trimesh.Trimesh):
        # If it's a single mesh, try to split into disconnected components
        components = mesh_data.split(only_watertight=False)
    else:
        print(f"Loaded data from '{input_stl_path}' is not a recognized Trimesh or Scene object. Cannot process.")
        return
    
    if not components:
        print(f"No separable components found in '{input_stl_path}'.")
        return
    print(f"Found {len(components)} components in '{input_stl_path}'.")

    # Convert trimesh components to PyVista PolyData
    pv_components = [pv.wrap(c) for c in components if c.vertices.shape[0] > 0 and c.faces.shape[0] > 0]

    if not pv_components:
        print("No valid PyVista components could be created from the STL.")
        return

    # 2. Calculate centroids
    component_centroids = np.array([comp.center_of_mass() for comp in pv_components])
    overall_centroid = np.mean(component_centroids, axis=0)

    # 3. Animation parameters
    total_frames = int(duration * fps)
    
    # Create a plotter object for off-screen rendering
    plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])

    # Store initial displacement vectors of components from the overall centroid
    initial_vectors_from_center = component_centroids - overall_centroid

    # Estimate model size for camera distance
    if isinstance(mesh_data, (trimesh.Trimesh, trimesh.Scene)):
        overall_bounds = mesh_data.bounds
    else: # Fallback if original mesh_data type was unexpected
        all_points_list = [comp.points for comp in pv_components if comp.n_points > 0]
        if not all_points_list:
            print("Components have no points for bounds calculation.")
            overall_bounds = np.array([[-1,-1,-1],[1,1,1]]) # Default bounds
        else:
            all_points = np.vstack(all_points_list)
            min_coords = all_points.min(axis=0)
            max_coords = all_points.max(axis=0)
            overall_bounds = np.array([min_coords, max_coords])

    model_span = np.linalg.norm(overall_bounds[1] - overall_bounds[0])
    if model_span < 1e-6: model_span = 10.0 # Default span for tiny/point models
    cam_distance = model_span * 2.0 # Camera distance heuristic
    if cam_distance < 1e-6: cam_distance = 10.0


    # Open a writer for the animation file
    try:
        writer = imageio.get_writer(output_animation_path, fps=fps)
    except Exception as e:
        print(f"Error creating animation writer for '{output_animation_path}': {e}")
        print("Please ensure imageio and necessary backends (like ffmpeg for MP4) are installed.")
        return

    print(f"Starting animation rendering ({total_frames} frames)...")
    for frame_idx in range(total_frames):
        plotter.clear_actors() 
        
        t_norm = frame_idx / (total_frames -1) if total_frames > 1 else 0 # Normalized time 0 to 1

        # Explosion factor: 0 -> 1 (at mid_time) -> 0 using a sine curve
        explosion_t = np.sin(t_norm * np.pi) 

        # Current scale for explosion distance (1.0 = original position, >1.0 = exploded)
        current_explosion_distance_scale = 1.0 + (max_explosion_scale - 1.0) * explosion_t

        # Spin angle for the scene
        current_spin_angle_deg = (t_norm * spin_revolutions * 360.0)

        for i, pv_comp_original in enumerate(pv_components):
            pv_comp_transformed = pv_comp_original.copy(deep=True) # Work on a copy

            # Calculate current target centroid for this component
            exploded_relative_pos = initial_vectors_from_center[i] * current_explosion_distance_scale
            target_centroid_pos = overall_centroid + exploded_relative_pos
            
            # Translate the component from its original centroid to the new target centroid
            translation_vector = target_centroid_pos - component_centroids[i]
            pv_comp_transformed.translate(translation_vector, inplace=True)
            
            plotter.add_mesh(pv_comp_transformed)

        # Configure camera
        plotter.camera.focal_point = overall_centroid.tolist()
        plotter.camera.viewup = (0, 0, 1) # Assuming Z is the up direction for the model
        plotter.camera.azimuth = current_spin_angle_deg
        plotter.camera.elevation = -45 # 45 degrees looking down
        plotter.camera.distance = cam_distance
        
        # Ensure the entire scene is visible
        plotter.reset_camera_clipping_range()

        # Capture frame
        try:
            frame_image = plotter.screenshot(None, return_img=True)
            writer.append_data(frame_image)
        except Exception as e:
            print(f"Error capturing frame {frame_idx + 1}: {e}")
            # Continue to next frame if possible, or break
            break 
        
        if (frame_idx + 1) % fps == 0: # Print progress every second of animation
             print(f"Rendered frame {frame_idx + 1}/{total_frames}")

    writer.close()
    plotter.close()
    print(f"Animation successfully saved to '{output_animation_path}'")

def main():
    parser = argparse.ArgumentParser(description="Create an exploding/imploding and spinning animation from an STL file containing multiple objects.")
    parser.add_argument("input_stl", help="Path to the input STL file (e.g., a _gapped_diff.stl).")
    parser.add_argument("output_animation", help="Path to save the output animation (e.g., animation.gif or animation.mp4).")
    parser.add_argument("--duration", type=float, default=10, help="Duration of the animation in seconds (default: 10).")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for the animation (default: 20).")
    parser.add_argument("--max_explosion_scale", type=float, default=1.5, 
                        help="Scale factor for maximum explosion distance relative to initial component positions from center (e.g., 1.5 means components move out to 1.5x their original distance from the center, default: 1.5). Must be >= 1.0.")
    parser.add_argument("--spin_revolutions", type=float, default=2, help="Total number of horizontal revolutions for the scene (default: 2).")
    
    args = parser.parse_args()

    if args.max_explosion_scale < 1.0:
        print("Warning: --max_explosion_scale must be >= 1.0. Adjusting to 1.0 (no outward explosion beyond original positions).")
        args.max_explosion_scale = 1.0

    create_animation(args.input_stl, args.output_animation, 
                     args.duration, args.fps, args.max_explosion_scale, args.spin_revolutions)

if __name__ == "__main__":
    main()
