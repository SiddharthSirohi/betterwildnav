# OmniGlue Integration with WildNav

This document describes the integration of OmniGlue feature matching into WildNav for improved UAV localization.

## Overview

**WildNav** is a vision-based GNSS-free localization system for UAVs. Originally, it used **SuperGlue** for feature matching. This integration replaces SuperGlue with **OmniGlue**, a CVPR 2024 state-of-the-art matcher that leverages foundation model guidance (DINOv2) for better generalization.

### Expected Improvements

- **Better generalization** to unseen environments (+18.8% on out-of-domain data)
- **More robust matching** across different seasons, lighting, weather conditions
- **Foundation model guidance** provides semantic understanding
- **Maintained compatibility** with existing WildNav pipeline

---

## Quick Start (Google Colab)

```python
# 1. Clone repository
!git clone <your-repo-url>
%cd wildnav

# 2. Setup models (downloads to Google Drive for persistence)
!python setup_models_colab.py

# 3. Install dependencies
!pip install -q -r requirements_omniglue.txt

# 4. Test integration
%cd src
!python test_integration.py

# 5. Run localization
!python wildnav.py
```

---

## Installation (Local)

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, CPU fallback supported)
- ~2GB disk space for models

### Step-by-Step

```bash
# 1. Navigate to wildnav directory
cd wildnav

# 2. Create virtual environment (recommended)
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# 3. Install dependencies
pip install -r requirements_omniglue.txt

# 4. Download models
bash setup_models.sh
# Or: python setup_models_colab.py

# 5. Verify installation
cd src
python test_integration.py

# 6. Run WildNav
python wildnav.py
```

---

## Architecture

### Integration Points

```
wildnav.py (Main Pipeline)
    ↓
    ├─ [OLD] superglue_utils.py  ← Original matcher
    │      Uses: PyTorch SuperGlue
    │
    └─ [NEW] omniglue_utils.py   ← New matcher (drop-in replacement)
           Uses: TensorFlow OmniGlue + PyTorch DINOv2
           Components:
             ├─ SuperPoint (keypoint detection)
             ├─ DINOv2 (semantic features)
             └─ OmniGlue Matcher (feature matching)
```

### File Structure

```
wildnav/
├── config_omniglue.yaml       # Configuration parameters
├── setup_models_colab.py      # Model downloader (Colab-optimized)
├── setup_models.sh            # Model downloader (bash)
├── requirements_omniglue.txt  # Unified dependencies
├── models/                    # Downloaded model weights (~375MB)
│   ├── sp_v6/                # SuperPoint
│   ├── dinov2_vitb14_pretrain.pth  # DINOv2
│   └── og_export/            # OmniGlue
└── src/
    ├── wildnav.py            # Main pipeline (modified)
    ├── omniglue_utils.py     # New wrapper module
    ├── test_integration.py   # Test suite
    └── omniglue_lib/         # OmniGlue source code
        ├── omniglue_extract.py
        ├── superpoint_extract.py
        ├── dino_extract.py
        ├── utils.py
        └── third_party/dinov2/
```

---

## Configuration

Edit `config_omniglue.yaml` to tune parameters without modifying code:

### Key Parameters

**Confidence Threshold** (`omniglue.confidence_threshold`)
- Default: `0.02`
- Range: `0.01` (permissive) to `0.05` (strict)
- Higher = fewer but more confident matches

**Image Resize** (`omniglue.resize_max`)
- Default: `800` (matches original WildNav)
- Alternative: `630` (optimal for DINOv2)
- Affects speed vs accuracy tradeoff

**RANSAC Threshold** (`ransac.threshold`)
- Default: `5.0` pixels
- Geometric validation tolerance
- Lower = stricter validation

**Selection Strategy** (`selection.primary_criterion`)
- `"num_matches"`: Most inlier matches (default, original behavior)
- `"confidence_sum"`: Highest total confidence (experimental)

### Switching Between Matchers

In `wildnav.py`, change line 8:

```python
USE_OMNIGLUE = True   # Use OmniGlue
USE_OMNIGLUE = False  # Use original SuperGlue
```

---

## Testing & Validation

### Test Suite

```bash
cd src
python test_integration.py
```

**Tests performed:**
1. ✓ Import dependencies (TensorFlow, PyTorch, OpenCV, etc.)
2. ✓ OmniGlue library accessibility
3. ✓ Model files exist and are readable
4. ✓ OmniGlue initialization (loads all models)
5. ✓ Simple image matching (synthetic test)
6. ✓ WildNav integration (module imports)
7. ✓ GPU availability (PyTorch + TensorFlow)

### Expected Output

```
============================
TEST SUMMARY
============================
  imports              ✓ PASS
  omniglue_lib         ✓ PASS
  models               ✓ PASS
  initialization       ✓ PASS
  simple_match         ✓ PASS
  integration          ✓ PASS
  gpu                  ✓ PASS

Total: 7/7 tests passed

✓ All tests passed! Ready to run WildNav with OmniGlue.
```

### Validation Metrics

After running `wildnav.py`, compare results:

```bash
python plot_data.py
```

**Metrics to compare:**
- **Localization success rate** (% matched) - expect improvement
- **Mean Absolute Error (MAE)** - expect reduction
- **RMSE** - expect reduction

**Baseline (SuperGlue on Dataset 1):**
- Success: 62%
- MAE: 15.82m

**Target (OmniGlue):**
- Success: >65% (goal)
- MAE: <15m (goal)

---

## Usage

### Basic Usage

```bash
cd wildnav/src
python wildnav.py
```

**What it does:**
1. Loads satellite map tiles from `../assets/map/map.csv`
2. Loads drone images from `../assets/query/photo_metadata.csv`
3. For each drone image:
   - Matches against all satellite patches using OmniGlue
   - Finds best match with RANSAC validation
   - Calculates geographical coordinates
   - Saves visualization to `../results/`
4. Writes results to `../results/calculated_coordinates.csv`

### Adding Your Own Data

**1. Add drone images:**
```bash
# Place drone photos in:
wildnav/assets/query/

# Extract metadata:
cd src
python extract_image_meta_exif.py
# Creates: ../assets/query/photo_metadata.csv
```

**2. Add satellite maps:**
```bash
# Option A: Use build_map.py (requires Google Maps API key)
python build_map.py

# Option B: Manual
# 1. Place satellite images in: ../assets/map/
# 2. Create ../assets/map/map.csv with format:
#    Filename,Top_left_lat,Top_left_lon,Bottom_right_lat,Bottom_right_lon
#    sat_map_0.png,60.506787,22.311631,60.501037,22.324467
```

**3. Run localization:**
```bash
python wildnav.py
```

---

## Troubleshooting

### Issue 1: Models Not Found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: '../models/og_export'
```

**Solution:**
```bash
# Run model setup script
python setup_models_colab.py
# Or: bash setup_models.sh
```

### Issue 2: TensorFlow/PyTorch Conflicts

**Symptom:**
```
ImportError: cannot import name 'XXX' from 'tensorflow'
```

**Solution:**
```bash
# Reinstall with compatible versions
pip uninstall tensorflow torch
pip install tensorflow>=2.12.0 torch>=2.0.0
```

### Issue 3: CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce image size in `config_omniglue.yaml`:
   ```yaml
   resize_max: 630  # or 512
   ```

2. Use CPU mode (slower):
   ```yaml
   use_gpu: false
   ```

3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Issue 4: No Matches Found

**Symptom:**
```
✗ No valid match found
```

**Possible causes:**
1. **Confidence threshold too high** → Lower in `config_omniglue.yaml`
2. **Poor image quality** → Check drone/satellite image quality
3. **No overlap** → Verify satellite map covers flight zone
4. **Wrong map patch size** → Satellite images should cover sufficient area

**Solutions:**
```yaml
# Try more permissive settings
confidence_threshold: 0.01
ransac.threshold: 7.0
```

### Issue 5: Slow Performance

**Expected times (GPU):**
- Model loading: 3-5 seconds (once)
- Per image pair: 0.2-0.5 seconds
- Full dataset (100 images, 15 patches): 3-8 minutes

**Optimization tips:**
1. Verify GPU is being used:
   ```python
   import torch; print(torch.cuda.is_available())
   import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
   ```

2. Reduce resolution:
   ```yaml
   resize_max: 630
   ```

3. Enable early stopping:
   ```yaml
   early_stop_confidence: 0.9
   ```

---

## Implementation Details

### omniglue_utils.match_image() Flow

```
1. Initialize OmniGlue model (once)
   ├─ Load SuperPoint (TF)
   ├─ Load DINOv2 (PyTorch)
   └─ Load OmniGlue matcher (TF)

2. Load query image (1_query_image.png)

3. FOR EACH satellite patch:
   │
   ├─ FindMatches(query, satellite)
   │  ├─ SuperPoint: extract keypoints
   │  ├─ DINOv2: extract semantic features
   │  └─ OmniGlue: match features
   │
   ├─ Filter by confidence threshold
   │
   ├─ RANSAC homography validation
   │  └─ Get inlier matches
   │
   ├─ Calculate perspective transform
   │  └─ Find center of matched region
   │
   ├─ Validate center within bounds
   │
   └─ Track best match (max inliers)

4. Return best match details
```

### Return Value Specification

```python
(
    satellite_map_index,  # int: Index of best patch (0-based)
    center,              # tuple(float, float): Normalized (x, y) in [0, 1]
    located_image,       # np.ndarray: Visualization BGR image
    features_mean,       # np.ndarray: Mean keypoint position, shape (2,)
    query_image,         # np.ndarray: Query image BGR
    max_matches          # int: Number of inlier matches
)
```

### Coordinate System

```
Satellite Image Coordinates:
    (0, 0) ────────────► X (width)
      │
      │    Query matched here
      │         ◉ center
      │    (cX, cY) pixels
      │
      ▼ Y (height)

Normalized:
    center = (cX / width, cY / height)
    Range: [0.0, 1.0]

Geographic:
    lat = top_left_lat + center[1] * (bottom_right_lat - top_left_lat)
    lon = top_left_lon + center[0] * (bottom_right_lon - top_left_lon)
```

---

## Performance Comparison

### Computational Costs

| Component | Time (GPU) | Time (CPU) |
|-----------|------------|------------|
| Model Load | 3-5s | 10-15s |
| SuperPoint | 20-50ms | 200-500ms |
| DINOv2 | 50-100ms | 500-1000ms |
| OmniGlue Matcher | 50-100ms | 500-1000ms |
| Homography | 1-5ms | 1-5ms |
| **Total per pair** | **0.2-0.5s** | **1-2.5s** |

### Memory Usage

| Component | VRAM (GPU) | RAM (CPU) |
|-----------|------------|-----------|
| Models | ~2GB | ~4GB |
| Images | ~100MB | ~100MB |
| **Total** | **~2-3GB** | **~4-5GB** |

---

## Advanced Topics

### Multi-Scale Matching

Enable in `config_omniglue.yaml`:
```yaml
experimental:
  enable_multiscale: true
  scales: [0.5, 1.0, 1.5]
```

Tests images at multiple scales for robustness.

### Spatial Prior

Use previous localization to narrow search:
```yaml
experimental:
  use_spatial_prior: true
```

Only searches patches near previous position (faster, less robust).

### Custom Visualization

Modify `create_match_visualization()` in `omniglue_utils.py`:
- Change colors, line thickness
- Add confidence heatmaps
- Overlay geographical info

---

## Future Improvements

1. **Batch Processing:** Process multiple satellite patches in parallel
2. **Adaptive Thresholding:** Auto-tune confidence based on image quality
3. **Temporal Smoothing:** Use Kalman filter for trajectory consistency
4. **Fine-tuning:** Train OmniGlue on aerial-to-satellite specific data
5. **Real-time Mode:** Optimize for onboard UAV processing
6. **Web Interface:** Deploy as web service for remote localization

---

## References

- **WildNav Paper:** "Vision-Based GNSS-Free Localization for UAVs in the Wild" (IEEE ICMERR 2022)
- **OmniGlue Paper:** "Generalizable Feature Matching with Foundation Model Guidance" (CVPR 2024)
  - arXiv: https://arxiv.org/abs/2405.12979
  - Code: https://github.com/google-research/omniglue
- **SuperGlue:** https://github.com/magicleap/SuperGluePretrainedNetwork
- **DINOv2:** https://github.com/facebookresearch/dinov2

---

## Support

For issues specific to:
- **Integration:** Open issue in this repository
- **WildNav:** https://github.com/TIERS/wildnav/issues
- **OmniGlue:** https://github.com/google-research/omniglue/issues

---

## License

- **WildNav:** Original license applies
- **OmniGlue:** Apache 2.0 License
- **This Integration:** Inherits from parent projects

---

**Last Updated:** 2025-11-20
**Integration Version:** 1.0
**Tested On:** Google Colab, Ubuntu 20.04
