# Voxel Processor

This script, `process_vox.py`, is a command-line tool for processing `.vox` files. It allows scaling of voxel data, erosion of the scaled data, and saving the output in both `.vox` and `.stl` formats.

## Features

*   Load MagicaVoxel `.vox` files.
*   Scale voxel data by a specified factor.
*   Erode voxel data by a specified number of voxel layers.
*   Save processed voxel data back to a `.vox` file (with dimension clipping if necessary).
*   Save processed voxel data as an `.stl` mesh file using the Marching Cubes algorithm for robust geometry.

## Prerequisites

*   Python 3.7+ (developed with 3.7)
*   A virtual environment is recommended.

## Setup

1.  **Clone the repository or download the script.**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows (pwsh.exe or PowerShell)
    .\.venv\Scripts\Activate.ps1
    # On macOS/Linux (bash/zsh)
    # source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install:
    *   `numpy`
    *   `scipy`
    *   `py-vox-io` (for `.vox` file handling)
    *   `numpy-stl` (for `.stl` file saving, though `skimage` is now primary for mesh generation)
    *   `scikit-image` (for the Marching Cubes algorithm)

## Usage

The script is run from the command line:

```bash
python process_vox.py <input_vox_file> [options]
```

**Arguments:**

*   `input_vox_file`: Path to the input `.vox` file (e.g., `scene.vox`). This is a required argument.

**Options:**

*   `--scale_factor <float>`: Factor to scale voxels by. Default: `10.0`.
*   `--erosion_voxels <int>`: Number of voxel layers to erode after scaling. Default: `1` (as per last script update, previously was 2).
*   `-h`, `--help`: Show the help message and exit.

**Example:**

```bash
python process_vox.py scene.vox --scale_factor 10 --erosion_voxels 2
```

This command will:
1.  Load `scene.vox`.
2.  Scale its voxel data by a factor of 10.
3.  Erode the scaled data by 2 voxel layers.
4.  Save the result as `scene_processed.vox` in the same directory as `scene.vox`.
5.  Save the result as `scene_processed.stl` in the same directory as `scene.vox`.

## Output Files

The script will generate two output files in the same directory as the input file:

*   `<input_base_name>_processed.vox`: The processed voxel data in `.vox` format.
*   `<input_base_name>_processed.stl`: The processed voxel data as an STL mesh.

## Notes

*   **`.vox` File Clipping:** The MagicaVoxel `.vox` format has a maximum size for a single model (typically 256x256x256). If the scaling operation results in dimensions larger than this, the script will clip the data to fit, and a warning will be printed.
*   **Empty Models:** If the input model is empty or becomes empty after processing, appropriate warnings will be displayed, and empty files (or files representing empty models) will be generated.
*   **STL Generation:** The STL generation uses the Marching Cubes algorithm, which generally produces better and more manifold meshes from voxel data compared to simply creating a cube for each voxel.

## `.gitignore`

A `.gitignore` file is included to exclude common Python, virtual environment, IDE, OS-specific, and project output files from version control.

## License

This project is unlicensed. (Or specify your license if you have one).
