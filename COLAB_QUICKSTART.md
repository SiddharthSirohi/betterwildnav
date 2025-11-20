# WildNav + OmniGlue: Google Colab Quick Start

Run WildNav with OmniGlue feature matching in Google Colab. This guide gets you from zero to localization in ~10 minutes.

---

## ⚡ Quick Start (Copy-Paste)

Open a new Colab notebook and run these cells:

### Cell 1: Clone Repository
```python
!git clone <YOUR_GITHUB_REPO_URL>
%cd wildnav
!ls -la
```

### Cell 2: Setup Models (One-Time, ~5 mins)
```python
# This downloads models to Google Drive for persistence
!python setup_models_colab.py
```

**Output:** Models saved to `/content/drive/MyDrive/wildnav_models/` (~375MB)

### Cell 3: Install Dependencies (~2 mins)
```python
!pip install -q -r requirements_omniglue.txt
```

### Cell 4: Test Integration (~30 sec)
```python
%cd src
!python test_integration.py
```

**Expected:** `7/7 tests passed`

### Cell 5: Run Localization
```python
!python wildnav.py
```

**Output:**
- Matches each drone image against satellite maps
- Saves visualizations to `../results/`
- Creates `../results/calculated_coordinates.csv`

### Cell 6: View Results
```python
import cv2
from google.colab.patches import cv2_imshow
import glob

# Show first result
results = sorted(glob.glob('../results/*_located.png'))
if results:
    img = cv2.imread(results[0])
    cv2_imshow(img)
    print(f"Found {len(results)} localized images")
else:
    print("No results yet. Run wildnav.py first.")
```

### Cell 7: Analyze Performance
```python
!python plot_data.py
```

---

## 📊 Understanding the Output

### Visualization Images

Each matched drone image produces a `*_located.png` file showing:

```
┌─────────────────────────────────────────────────────┐
│ [Query Image]      |      [Matched Satellite]       │
│                    |                                 │
│    Drone View      |      Satellite View             │
│                    |                                 │
│  ● Feature points  |  ● Feature points               │
│                    |  ◉ Match center                 │
│                    |  □ Matched region               │
│─────────────────────────────────────────────────────│
│ Calculated: 60.403091°, 22.461824°                  │
│ Ground truth: 60.403095°, 22.461820°                │
└─────────────────────────────────────────────────────┘
```

**Green lines** = High confidence matches
**Red lines** = Lower confidence matches

### CSV Output

`results/calculated_coordinates.csv`:
```csv
Filename,Latitude,Longitude,Calculated_Latitude,Calculated_Longitude,Meters_Error
drone_img1.jpg,60.403091,22.461824,60.403095,22.461820,4.2
drone_img2.jpg,60.404123,22.462345,60.404130,22.462350,5.8
```

---

## 🔧 Configuration

Edit `config_omniglue.yaml` before running:

```yaml
# More permissive matching (more matches, may include false positives)
omniglue:
  confidence_threshold: 0.01  # Default: 0.02

# Stricter matching (fewer but more confident matches)
omniglue:
  confidence_threshold: 0.05

# Faster processing (lower quality)
omniglue:
  resize_max: 512  # Default: 800

# Better quality (slower)
omniglue:
  resize_max: 800
```

---

## 📁 Adding Your Own Data

### Option 1: Upload to Colab

```python
from google.colab import files
import shutil

# Upload drone images
uploaded = files.upload()
for filename in uploaded.keys():
    shutil.move(filename, f'assets/query/{filename}')

# Extract metadata
!python extract_image_meta_exif.py
```

### Option 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive
!cp /content/drive/MyDrive/my_drone_images/*.jpg assets/query/
!python extract_image_meta_exif.py
```

### Option 3: Download from URL

```python
!wget -P assets/query/ https://your-server.com/drone_images.zip
!unzip assets/query/drone_images.zip -d assets/query/
!python extract_image_meta_exif.py
```

---

## 🚀 Performance Tips

### Use GPU Runtime

1. Runtime → Change runtime type
2. Hardware accelerator: **T4 GPU** (free tier) or **A100** (Colab Pro)
3. Verify: Run `!nvidia-smi`

**Expected speedup:** 5-10x faster than CPU

### Enable High-RAM (if needed)

If you get "Out of Memory" errors:
1. Runtime → Change runtime type
2. Runtime shape: **High-RAM**

### Persistent Model Storage

Models are automatically saved to Google Drive at:
```
/content/drive/MyDrive/wildnav_models/
```

On subsequent sessions, models load instantly from Drive (no re-download needed).

---

## 🐛 Common Issues

### Issue: Drive Not Mounted

**Error:** `google.colab.errors.DriveTimeoutError`

**Fix:**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Issue: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Fix 1:** Restart runtime and reduce image size
```python
# In config_omniglue.yaml
omniglue:
  resize_max: 512
```

**Fix 2:** Use CPU mode (slower)
```python
# In config_omniglue.yaml
performance:
  use_gpu: false
```

### Issue: Models Download Fails

**Error:** `wget: unable to resolve host address`

**Fix:** Manually download to Drive:
```python
# Download each model separately with retries
!wget --tries=5 https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth -P models/
!wget --tries=5 https://storage.googleapis.com/omniglue/og_export.zip -P models/
!unzip models/og_export.zip -d models/ && rm models/og_export.zip
```

### Issue: No Matches Found

**Causes:**
1. Satellite map doesn't cover flight area
2. Image quality too poor
3. Threshold too strict

**Fix:**
```python
# Lower confidence threshold
# In config_omniglue.yaml:
omniglue:
  confidence_threshold: 0.01
```

---

## 📈 Benchmarking

Compare OmniGlue vs SuperGlue:

```python
# Run with OmniGlue (default)
!python wildnav.py
!cp ../results/calculated_coordinates.csv ../results/omniglue_results.csv

# Switch to SuperGlue
# Edit wildnav.py: USE_OMNIGLUE = False
!python wildnav.py
!cp ../results/calculated_coordinates.csv ../results/superglue_results.csv

# Compare
import pandas as pd
og = pd.read_csv('../results/omniglue_results.csv')
sg = pd.read_csv('../results/superglue_results.csv')

print(f"OmniGlue  - Success: {len(og)}  MAE: {og['Meters_Error'].mean():.2f}m")
print(f"SuperGlue - Success: {len(sg)}  MAE: {sg['Meters_Error'].mean():.2f}m")
```

---

## 💾 Saving Results

### Download Results to Local

```python
from google.colab import files

# Download CSV
files.download('../results/calculated_coordinates.csv')

# Download all visualizations (zip first)
!zip -r results.zip ../results/*.png
files.download('results.zip')
```

### Save to Google Drive

```python
import shutil

# Copy results to Drive
!mkdir -p /content/drive/MyDrive/wildnav_results
!cp -r ../results/* /content/drive/MyDrive/wildnav_results/
print("Results saved to Drive: /MyDrive/wildnav_results/")
```

---

## 🔬 Advanced: Colab-Specific Optimizations

### Pre-warm Models (Faster Multiple Runs)

```python
# Keep model in memory between runs
import sys
sys.path.insert(0, 'src/omniglue_lib')
from omniglue_lib.omniglue_extract import OmniGlue

# Load once
global og_model
og_model = OmniGlue(
    og_export='models/og_export',
    sp_export='models/sp_v6',
    dino_export='models/dinov2_vitb14_pretrain.pth'
)

# Modify omniglue_utils.py to accept pre-loaded model
```

### Batch Process Multiple Datasets

```python
datasets = [
    '/content/drive/MyDrive/dataset1',
    '/content/drive/MyDrive/dataset2',
]

for dataset_path in datasets:
    print(f"\nProcessing {dataset_path}...")
    !cp {dataset_path}/*.jpg assets/query/
    !python extract_image_meta_exif.py
    !python wildnav.py

    # Save results
    result_name = dataset_path.split('/')[-1]
    !cp ../results/calculated_coordinates.csv /content/drive/MyDrive/{result_name}_results.csv
```

### Monitor GPU Usage

```python
# Cell 1: Start monitoring
!nvidia-smi dmon -s u &

# Cell 2: Run processing
!python wildnav.py

# View GPU utilization over time
```

---

## 📞 Need Help?

1. **First:** Run `!python test_integration.py` to diagnose issues
2. **Check:** INTEGRATION.md for detailed troubleshooting
3. **Ask:** Open an issue with test output

---

## ⏱️ Expected Timing (T4 GPU)

| Task | Time |
|------|------|
| Model download (first time) | 3-5 min |
| Model loading | 3-5 sec |
| Single image matching | 0.3-0.5 sec |
| Full dataset (100 images, 15 patches) | 5-10 min |

---

## ✅ Validation Checklist

Before running full dataset:

- [ ] Test passes: `python test_integration.py`
- [ ] GPU is active: `!nvidia-smi` shows T4/A100
- [ ] Models exist: `!ls -la models/`
- [ ] Query images present: `!ls -la assets/query/*.jpg`
- [ ] Satellite maps present: `!ls -la assets/map/*.png`
- [ ] Map CSV exists: `!ls -la assets/map/map.csv`

---

**Happy Localizing! 🛸📍**
