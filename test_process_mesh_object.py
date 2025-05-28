\
import os
import trimesh
import shutil

# Assuming process_stl_parts.py is in the same directory or accessible in PYTHONPATH
from process_stl_parts import process_mesh_object, OUTPUT_DIR as DEFAULT_PROCESS_STL_OUTPUT_DIR

# Configuration
TEST_STL_DIR = "test_stls"  # Relative to the script's location
TEST_OUTPUT_BASE_DIR = "test_output_parts_from_script" # To avoid conflict with process_stl_parts.py's default

# Ensure the base output directory exists
if not os.path.exists(TEST_OUTPUT_BASE_DIR):
    os.makedirs(TEST_OUTPUT_BASE_DIR)
else:
    # Clean up previous test runs if the directory already exists
    print(f"Cleaning up existing test output directory: {TEST_OUTPUT_BASE_DIR}")
    shutil.rmtree(TEST_OUTPUT_BASE_DIR)
    os.makedirs(TEST_OUTPUT_BASE_DIR)


def run_tests():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_stl_abs_dir = os.path.join(script_dir, TEST_STL_DIR)
    test_output_abs_base_dir = os.path.join(script_dir, TEST_OUTPUT_BASE_DIR)

    print(f"Looking for test STL files in: {test_stl_abs_dir}")
    
    if not os.path.isdir(test_stl_abs_dir):
        print(f"ERROR: Test STL directory not found: {test_stl_abs_dir}")
        return

    for filename in os.listdir(test_stl_abs_dir):
        if filename.lower().endswith(".stl"):
            input_stl_path = os.path.join(test_stl_abs_dir, filename)
            original_filename_base = os.path.splitext(filename)[0]
            print(f"\\nProcessing: {filename}...")

            # Create a specific output directory for this test file's results
            specific_test_output_dir = os.path.join(test_output_abs_base_dir, original_filename_base)
            if not os.path.exists(specific_test_output_dir):
                os.makedirs(specific_test_output_dir)

            try:
                input_mesh = trimesh.load_mesh(input_stl_path)
                if input_mesh.is_empty:
                    print(f"  WARNING: Loaded mesh {filename} is empty. Skipping processing.")
                    continue
                print(f"  Mesh {filename} loaded successfully. Volume: {input_mesh.volume if hasattr(input_mesh, 'volume') else 'N/A'}")
            except Exception as e:
                print(f"  ERROR: Failed to load STL file {input_stl_path}: {e}")
                continue

            # Call process_mesh_object
            # Setting run_three_wall_flag=False to test auto-detection logic
            # The original process_stl_parts.py script sets a global OUTPUT_DIR.
            # process_mesh_object itself doesn't use it for saving, but functions it calls might if not fully refactored.
            # For this test, we are handling saving explicitly.
            try:
                list_of_processed_meshes, cut_type = process_mesh_object(
                    input_mesh=input_mesh,
                    original_filename_base=original_filename_base,
                    run_three_wall_flag=False 
                )
                print(f"  process_mesh_object returned {len(list_of_processed_meshes)} mesh(es). Cut type: {cut_type}")

                # Save the resulting meshes
                if list_of_processed_meshes and any(m is not None and not m.is_empty for m in list_of_processed_meshes):
                    # Filter out None or empty meshes before attempting concatenation or saving
                    valid_meshes = [m for m in list_of_processed_meshes if m is not None and not m.is_empty]
                    
                    if not valid_meshes:
                        print(f"  All returned meshes for {original_filename_base} are empty or None. Nothing to save.")
                        continue

                    combined_mesh = None
                    if len(valid_meshes) > 1:
                        try:
                            combined_mesh = trimesh.util.concatenate(valid_meshes)
                            print(f"  Successfully combined {len(valid_meshes)} processed parts into a single mesh.")
                        except Exception as e_concat:
                            print(f"  ERROR: Failed to concatenate meshes for {original_filename_base}: {e_concat}. Saving parts individually if possible.")
                            # Fallback: save individually if concatenation fails, though the request is for one file.
                            # For now, let's stick to the request and not save if concatenation fails.
                            # Or, alternatively, save the first valid piece only with a warning.
                            print(f"  Skipping save for {original_filename_base} due to concatenation error.")
                            continue # Skip to next file
                    elif len(valid_meshes) == 1:
                        combined_mesh = valid_meshes[0]
                    
                    if combined_mesh is not None and not combined_mesh.is_empty:
                        output_filename = f"{original_filename_base}_processed_as_{cut_type}_combined.stl"
                        output_stl_path = os.path.join(specific_test_output_dir, output_filename)
                        
                        try:
                            combined_mesh.export(output_stl_path)
                            print(f"  Saved combined processed mesh to: {output_stl_path}")
                        except Exception as e_export:
                            print(f"  ERROR: Failed to export combined mesh {output_stl_path}: {e_export}")
                    else:
                        print(f"  Combined mesh for {original_filename_base} is empty or None. Nothing to save.")
                elif list_of_processed_meshes: # All were None or empty
                    print(f"  All returned meshes for {original_filename_base} are empty or None. Nothing to save.")
                else: # list_of_processed_meshes was empty
                    print(f"  No meshes returned by process_mesh_object for {original_filename_base}, nothing to save.")

            except Exception as e_process:
                print(f"  ERROR: Failed during process_mesh_object for {filename}: {e_process}")
                import traceback
                traceback.print_exc()

    print("\\nTest processing complete.")
    print(f"Output parts are in: {test_output_abs_base_dir}")

if __name__ == "__main__":
    # This is to ensure that if process_stl_parts.py has its own main guard,
    # it doesn't run when we import from it.
    # Also, explicitly set the OUTPUT_DIR for any functions in process_stl_parts
    # that might still rely on it, to our test-specific one, though ideally they shouldn't.
    # However, process_mesh_object and its direct callees are refactored not to write files.
    
    # The global OUTPUT_DIR in process_stl_parts.py is used by its main() for output.
    # Since we are calling process_mesh_object directly and handling our own output,
    # we don't strictly need to override it unless some deeper, unrefactored utility
    # function within process_stl_parts.py unexpectedly writes files based on that global.
    # The refactoring goal was to make process_mesh_object pure in terms of file I/O for outputs.
    
    run_tests()
