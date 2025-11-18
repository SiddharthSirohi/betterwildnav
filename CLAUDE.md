# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an IoT project repository containing **WildNav**, a vision-based GNSS-free localization system for UAVs (Unmanned Aerial Vehicles) in non-urban environments. The project matches drone camera images with georeferenced satellite imagery to determine geographical coordinates without relying on GPS.

## MIGRATION DIRECTIVE

**ACTIVE GOAL**: Migrate the feature matching pipeline from SuperGlue/SuperPoint to LightGlue/DINOv3.

**Migration Steps**:
1. **Phase 1**: Replace SuperGlue matcher with LightGlue (keeping SuperPoint for now)
2. **Phase 2**: Replace SuperPoint feature extractor with DINOv3

**Rationale**:
- LightGlue is the successor to SuperGlue with better performance and speed
- DINOv3 provides superior dense features for matching drone/satellite imagery
- Documentation for both is available in `documentation/` folder

## Repository Structure

```
wildnav/
├── src/                     # Main source code
│   ├── wildnav.py          # Core localization algorithm (main entry point)
│   ├── superglue_utils.py  # SuperGlue feature matching wrapper
│   ├── build_map.py        # Satellite map builder using Google Maps API
│   ├── extract_image_meta_exif.py  # EXIF metadata extraction from drone images
│   └── superglue_lib/      # SuperGlue submodule (git submodule)
├── assets/
│   ├── query/              # Drone images and photo_metadata.csv
│   └── map/                # Satellite images and map.csv
├── results/                # Output directory for localization results
└── env/                    # Python virtual environment

documentation/
├── README (dinoV3).md     # DINOv3 feature extractor documentation
└── README (LightGlue).md  # LightGlue matcher documentation
```

## Development Setup

### Environment Setup

The project uses Python 3.10+ with a virtual environment:

```bash
# Activate virtual environment
cd wildnav
source env/bin/activate

# Install dependencies (if not already installed)
pip3 install -r requirements.txt

# Initialize SuperGlue submodule (required on first setup)
git submodule update --init --recursive
```

### Running the Localization Algorithm

```bash
cd wildnav/src
python3 wildnav.py
```

This runs the main localization pipeline which:
1. Reads satellite map images from `../assets/map/map.csv`
2. Reads drone images from `../assets/query/photo_metadata.csv`
3. Matches each drone image against satellite map patches using SuperGlue
4. Calculates geographical coordinates based on feature matches
5. Outputs results to `../results/calculated_coordinates.csv` and individual `*_located.png` images

### Preprocessing Drone Images

Extract EXIF metadata from drone photos (requires `exiftool` to be installed):

```bash
cd wildnav/src
python3 extract_image_meta_exif.py
```

This creates `../assets/query/photo_metadata.csv` containing GPS coordinates and camera orientation parameters.

### Building Satellite Maps

Generate satellite map patches using Google Maps Static API:

```bash
cd wildnav/src
python3 build_map.py
```

**Note**: You need your own Google Maps API key (hardcoded API key in the file is IP-restricted and won't work).

## Core Architecture

### Feature Matching Pipeline (wildnav.py)

The main algorithm follows this flow:
1. Load satellite map images as `GeoPhoto` objects with geo-tagged corners (lat/lon for top-left and bottom-right)
2. Load drone images as `GeoPhotoDrone` objects with GNSS metadata and gimbal/flight orientation
3. For each drone image:
   - Try different rotation angles (if GNSS rotation metadata is unreliable)
   - Write query image to `../assets/map/1_query_image.png`
   - Call `superglue_utils.match_image()` to find best matching satellite patch
   - Use feature matches to compute homography transformation
   - Calculate geographical coordinates from matched position
   - Save visualization and results

### SuperGlue Integration (superglue_utils.py)

The `match_image()` function:
- Loads all satellite map patches from `../assets/map/`
- Uses SuperPoint for keypoint detection and SuperGlue for matching
- Iterates through all satellite patches to find best match
- Computes homography using `cv2.findHomography()` with RANSAC
- Returns: satellite map index, center coordinates, visualization, features, and match count

**Key parameters** (in `superglue_utils.py:match_image()`):
- `resize = [800]`: Image resize dimension for processing
- `superglue = 'outdoor'`: SuperGlue model variant
- `keypoint_threshold = 0.01`: Keypoint detection confidence
- `match_threshold = 0.5`: Feature matching confidence
- `force_cpu = False`: Set to True to run on CPU (significantly slower)

### Data Models

**GeoPhotoDrone**: Stores drone image with GNSS coordinates and camera/flight orientation parameters (roll, pitch, yaw for both gimbal and flight controller)

**GeoPhoto**: Stores satellite image with geo-tagged bounding box (top-left and bottom-right lat/lon coordinates)

## Common Issues

### PyTorch CUDA Compatibility

If you encounter CUDA capability errors:
```bash
pip3 uninstall torch
# Install version matching your GPU from https://pytorch.org/get-started/locally/
```

### Running on CPU

Edit `wildnav/src/superglue_utils.py:37`:
```python
force_cpu = True
```

### EXIF Extraction

The `extract_image_meta_exif.py` script requires `exiftool` to be installed system-wide. The script is tailored for DJI drone metadata format - adjust field names if using different drone hardware.

## Testing and Validation

Results are validated by comparing calculated coordinates against GNSS ground truth from drone EXIF data. The algorithm outputs:
- **Mean Absolute Error (MAE)** in meters using haversine distance
- **Localization success rate** (percentage of images successfully matched)
- Individual error visualizations in `results/` directory

## Migration Reference Documentation

The `documentation/` folder contains READMEs for the target implementations:
- **DINOv3** (`README (dinoV3).md`): Self-supervised vision transformer for dense feature extraction - target replacement for SuperPoint
- **LightGlue** (`README (LightGlue).md`): Lightweight, adaptive feature matching (successor to SuperGlue) - target replacement for SuperGlue

These documents should be consulted during the migration process for API details and best practices.

## Important Notes

- The project uses a **git submodule** for SuperGlue (`wildnav/src/superglue_lib/`). Always run `git submodule update --init --recursive` after cloning.
- Satellite map CSV format: `Filename, Top_left_lat, Top_left_lon, Bottom_right_lat, Bottom_right_lon`
- Drone metadata CSV format: `Filename, Latitude, Longitude, Altitude, Gimball_Roll, Gimball_Yaw, Gimball_Pitch, Flight_Roll, Flight_Yaw, Flight_Pitch`
- The algorithm requires at least 4 feature matches to compute homography (OpenCV requirement)
- Image paths are relative to the `src/` directory (all scripts should be run from `wildnav/src/`)
