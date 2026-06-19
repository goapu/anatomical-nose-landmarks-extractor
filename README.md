# Anatomical Nose Landmarks Extractor

## Overview

`noseLandmarksExtraction.py` is a Python research script for extracting anatomical nose landmark proposals and dense nasal surface templates from 3D human face meshes.

The script estimates a reproducible nasal region of interest (ROI), proposes named anatomical landmarks, generates dense nasal guide structures, and exports review files for visual inspection and measurement pre-processing.

It is intended for research, visualization, and experimental 3D facial analysis workflows.

## Methodology

This research script follows a geometry-first approach. Landmark proposals are derived from local 3D mesh geometry rather than 2D image-based face detectors.

The extraction workflow includes:

* loading and cleaning the input 3D mesh
* estimating a nasal region of interest
* extracting an anterior connected nose surface patch
* filtering posterior and non-nasal surface points
* detecting anatomical landmarks using local geometry
* generating dense nasal surface templates
* exporting review files, point clouds, overlays, and measurements

The script uses geometric cues such as:

* anterior surface depth
* local curvature
* alar width
* nostril basin structure
* surface connectivity
* anterior-envelope filtering

## Supported Mesh Formats

The script supports standard 3D mesh formats readable by Open3D:

* `.obj`
* `.ply`
* `.stl`

## Extracted Anatomical Landmarks

The script proposes the following named anatomical landmarks:

* nasion/radix
* pronasale
* subnasale
* columella point
* left and right alare
* left and right subalare
* left and right alar crease
* left and right nostril medial points
* left and right nostril lateral points
* left and right nostril superior points
* left and right nostril inferior points

Each anatomical landmark is exported with:

* 3D coordinates
* extraction method
* confidence score
* manual-review flag
* projected review coordinates

## Dense Surface Templates

In addition to named landmarks, the script exports dense and auditable nasal surface guide templates, including:

* nasal midline curve
* left and right sidewall curves
* left and right alar crease curves
* left and right nostril rim loops
* dense anterior nose surface grid
* optional transverse lower-nose guide lines from the alar rim toward the infratip

Vertical columella coverage lines are intentionally not generated in the current version.

## Optional Pose Normalization

The script includes optional symmetry-based pose normalization using Iterative Closest Point (ICP):

```bash
--enable_pose_normalize
```

This option is disabled by default because symmetry normalization can reduce or alter true facial asymmetry. For asymmetry-sensitive research, pose normalization should be used only after careful review.

## Installation

Install the required Python packages:

```bash
python -m pip install open3d numpy pillow
```

Recommended Python version:

```text
Python 3.10+
```

## Basic Usage

Run the script on a single mesh:

```bash
python noseLandmarksExtraction.py \
  --obj "/path/to/input_mesh.obj" \
  --outdir "/path/to/output_directory" \
  --front=+z \
  --up=+y
```

The axis arguments define mesh orientation:

| Argument  | Description                                | Example      |
| --------- | ------------------------------------------ | ------------ |
| `--front` | anterior-facing direction of the face mesh | `--front=+z` |
| `--up`    | superior/upward direction of the face mesh | `--up=+y`    |

Use the equals syntax for signed axis values:

```bash
--front=+z --up=+y
```

## Batch Processing

To process all `.obj` files recursively inside a folder:

```bash
find "/Users/gmac/Desktop/Freelancing /aesthetic_project/Data" -type f -name "*.obj" | while IFS= read -r obj; do
  name=$(basename "$obj" .obj)
  python noseLandmarksExtraction.py \
    --obj "$obj" \
    --outdir "/Users/gmac/Desktop/Freelancing /aesthetic_project/Data/result/$name" \
    --front=+z \
    --up=+y \
    --patch_width_factor 0.72 \
    --patch_left_factor 1.55 \
    --patch_right_factor 1.35 \
    --patch_y_margin 0.01 \
    --patch_bottom_margin 0.00 \
    --patch_bottom_raise 0.18 \
    --patch_front_percentile 76 \
    --patch_front_slack_factor 0.25 \
    --dense_grid_width_factor 0.74 \
    --dense_left_factor 1.45 \
    --dense_right_factor 1.25 \
    --dense_front_percentile 82 \
    --dense_front_slack_factor 0.20 \
    --columella_transverse_lines 4 \
    --columella_transverse_samples 15 \
    --columella_transverse_top_drop 0.020 \
    --columella_transverse_max_half 0.20 \
    --columella_tip_blend 0.30 \
    --columella_lip_guard_factor 0.040 \
    --columella_left_expand 2.30 \
    --columella_right_expand 1.10 \
    --columella_x_band 0.080 \
    --columella_y_band 0.090 \
    --columella_front_percentile 72 \
    --columella_front_slack 0.20 \
    --columella_projection_blend 0.85 \
    --columella_duplicate_spacing_factor 0.006 \
    --bottom_ratio 0.22 \
    --bottom_extension 0.00
done
```

Each mesh is exported to:

```text
Data/result/<mesh_name>/
```

## Output Files

The script generates the following files in the output directory:

```text
anatomical_nose_landmarks_3d.csv
anatomical_nose_landmarks_review.json
dense_nose_template_points.csv
anatomical_nose_landmarks_cloud.ply
dense_nose_template_cloud.ply
nose_surface_patch_cloud.ply
nose_landmarks_medical_review_overlay.png
measurements_review_required.json
pose_transform_matrix.csv
```

## Output Descriptions

### `anatomical_nose_landmarks_3d.csv`

CSV file containing extracted anatomical landmark proposals with 3D coordinates, extraction methods, confidence scores, and manual-review flags.

Typical columns include:

```text
id
name
method
confidence
needs_manual_review
x
y
z
screen_x
screen_y
depth_front
```

### `anatomical_nose_landmarks_review.json`

Structured JSON file containing the named anatomical landmark proposals and audit information.

### `dense_nose_template_points.csv`

CSV file containing dense nasal surface template points, such as:

```text
midline_curve
left_sidewall_curve
right_sidewall_curve
left_alar_crease_curve
right_alar_crease_curve
left_nostril_rim_loop
right_nostril_rim_loop
surface_grid
columella_tip_to_alar_lines
```

The `columella_tip_to_alar_lines` group contains optional transverse lower-nose guide points.

### `anatomical_nose_landmarks_cloud.ply`

Point cloud containing the named anatomical landmark proposals.

### `dense_nose_template_cloud.ply`

Point cloud containing the dense nasal surface template points and guide structures.

### `nose_surface_patch_cloud.ply`

Point cloud containing the extracted anterior nose surface patch used during processing.

### `nose_landmarks_medical_review_overlay.png`

2D projected review image for visual auditing. It displays the mesh depth shading, extracted nose patch, dense guide points, and numbered anatomical landmarks.

### `measurements_review_required.json`

JSON file containing derived measurements and manual-review status.

Computed measurements include:

```text
Nasal_Height_nasion_to_subnasale
Alar_Base_Width
Tip_to_Subnasale_Distance
Nasal_Index_Width_over_Height
manual_review_required
```

### `pose_transform_matrix.csv`

Stores the 4×4 transformation matrix if optional pose normalization is enabled.

## Manual Review and Confidence

Each anatomical landmark receives a confidence score between `0.0` and `1.0`.

A landmark is marked for review when its confidence falls below the configured threshold:

```bash
--review_threshold 0.45
```

If any core landmark requires review, the measurement file includes:

```json
"manual_review_required": true
```

These flags should be reviewed before using the measurements in downstream analysis.

## Research Use Statement

This research script is intended for experimental 3D facial analysis, visualization, and measurement pre-processing.

It is not a medical device and does not provide diagnosis, treatment planning, or automated clinical assessment. All landmark proposals and measurements should be reviewed before use in scientific or publication workflows.

## Limitations

* Landmark quality depends on mesh resolution, orientation, completeness, and noise level.
* Sparse or partially occluded meshes may reduce landmark confidence.
* Nostril and lower-nose regions can be difficult when under-sampled.
* The script proposes landmarks; it does not guarantee final anatomical correctness.
* Human review is required before using the outputs for research reporting or measurement-based interpretation.

## Recommended Workflow

1. Prepare clean 3D face meshes.
2. Confirm mesh orientation.
3. Run the extraction script.
4. Inspect `nose_landmarks_medical_review_overlay.png`.
5. Review `anatomical_nose_landmarks_review.json`.
6. Check `measurements_review_required.json`.
7. Use measurements only after checking low-confidence landmarks.

## Suggested Repository Structure

```text
anatomical-nose-landmarks-extractor/
├── noseLandmarksExtraction.py
├── README.md
├── requirements.txt
├── data/
│   ├── input/
│   └── result/
└── examples/
```

## Minimal `requirements.txt`

```text
open3d
numpy
pillow
```

## Author

**Dilip Goswami**<br>
MSc in Geodesy and Geoinformation Science<br>
Technical University of Berlin

## Disclaimer

This repository contains a research script for automated 3D nasal landmark proposal and surface-template generation. It is intended for research and visualization workflows only and should not be used as a substitute for expert review.
