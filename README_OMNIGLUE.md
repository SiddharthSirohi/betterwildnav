# WildNav with OmniGlue Integration 🛸📍

## Overview

This repository contains **WildNav** (GNSS-free UAV localization) enhanced with **OmniGlue** (CVPR 2024 feature matching). The integration improves localization accuracy and generalization to unseen environments.

### What's New?

✅ **OmniGlue Matcher** - State-of-the-art feature matching with foundation model guidance
✅ **Better Generalization** - 18.8% improvement on out-of-domain data
✅ **Seamless Integration** - Drop-in replacement, keeps original pipeline
✅ **Easy Switching** - Toggle between SuperGlue and OmniGlue
✅ **Colab-Ready** - Optimized for Google Colab with persistent model storage
✅ **Comprehensive Testing** - Full test suite with validation

---

## 🚀 Quick Start

### For Google Colab Users (Recommended)

See **[COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)** - Get running in 10 minutes!

```python
# In Colab notebook:
!git clone <your-repo-url>
%cd wildnav
!python setup_models_colab.py
!pip install -q -r requirements_omniglue.txt
%cd src
!python test_integration.py  # Verify installation
!python wildnav.py           # Run localization
```

### For Local Users

```bash
cd wildnav
pip install -r requirements_omniglue.txt
bash setup_models.sh
cd src
python test_integration.py
python wildnav.py
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)** | 10-minute Colab tutorial |
| **[INTEGRATION.md](INTEGRATION.md)** | Complete integration guide |
| **[CLAUDE.md](../CLAUDE.md)** | Developer reference & migration plan |
| **[config_omniglue.yaml](config_omniglue.yaml)** | Tunable parameters |

---

## 📁 Key Files

### New Integration Files
```
wildnav/
├── requirements_omniglue.txt       ⭐ Unified dependencies
├── config_omniglue.yaml            ⭐ Configuration
├── setup_models_colab.py           ⭐ Model downloader (Colab)
├── setup_models.sh                 ⭐ Model downloader (bash)
├── INTEGRATION.md                  ⭐ Integration docs
├── COLAB_QUICKSTART.md             ⭐ Quick start guide
├── models/                         ⭐ Model weights (~375MB)
│   ├── sp_v6/
│   ├── dinov2_vitb14_pretrain.pth
│   └── og_export/
└── src/
    ├── wildnav.py                  🔧 Modified (matcher selection)
    ├── omniglue_utils.py           ⭐ New wrapper module
    ├── test_integration.py         ⭐ Test suite
    └── omniglue_lib/               ⭐ OmniGlue source
```

### Original WildNav Files
```
wildnav/
├── README.md                       # Original documentation
├── requirements.txt                # Original dependencies
└── src/
    ├── superglue_utils.py          # Original matcher
    ├── build_map.py                # Satellite downloader
    ├── extract_image_meta_exif.py  # EXIF extractor
    └── plot_data.py                # Results visualizer
```

---

## 🎯 Usage

### Basic Localization

```bash
cd wildnav/src
python wildnav.py
```

**Input:**
- Drone images: `assets/query/*.jpg`
- Satellite maps: `assets/map/*.png`
- Metadata: `assets/query/photo_metadata.csv`, `assets/map/map.csv`

**Output:**
- Visualizations: `results/*_located.png`
- Coordinates: `results/calculated_coordinates.csv`

### Switch Between Matchers

Edit `src/wildnav.py` line 8:
```python
USE_OMNIGLUE = True   # OmniGlue (new)
USE_OMNIGLUE = False  # SuperGlue (original)
```

### Tune Parameters

Edit `config_omniglue.yaml`:
```yaml
omniglue:
  confidence_threshold: 0.02  # Adjust 0.01-0.05
  resize_max: 800             # Adjust 512-800
```

### Test Installation

```bash
cd src
python test_integration.py
```

Expected output: `7/7 tests passed`

---

## 🔬 How It Works

### Pipeline Comparison

**Original (SuperGlue):**
```
Query Image → SuperPoint → SuperGlue → RANSAC → Location
```

**New (OmniGlue):**
```
Query Image → SuperPoint ──┐
               ↓           ├→ OmniGlue → RANSAC → Location
              DINOv2 ──────┘
```

### Key Advantages

1. **Foundation Model Guidance** - DINOv2 provides semantic-rich features
2. **Better Generalization** - Works across seasons, weather, lighting
3. **Position-Guided Attention** - Disentangles spatial/appearance info
4. **Maintained Compatibility** - Same output format as original

---

## 📊 Expected Performance

### Baseline (SuperGlue)
- **Success Rate:** 56-62%
- **MAE:** 15.82-26.58m

### Target (OmniGlue)
- **Success Rate:** 65%+ (goal)
- **MAE:** <15m (goal)
- **Generalization:** +18.8% on out-of-domain data

### Computational Cost (GPU)
- **Model Load:** 3-5 seconds (once)
- **Per Image Pair:** 0.2-0.5 seconds
- **Full Dataset:** 5-10 minutes (100 images, 15 patches)

---

## 🛠️ Configuration Options

### Matching Parameters

```yaml
# config_omniglue.yaml

# Confidence threshold (0.0 - 1.0)
confidence_threshold: 0.02

# Image resize (pixels)
resize_max: 800

# RANSAC threshold (pixels)
ransac.threshold: 5.0

# Selection strategy
selection.primary_criterion: "num_matches"  # or "confidence_sum"
```

### Performance Tuning

```yaml
# Use GPU
performance.use_gpu: true

# Early stopping (0.0 - 1.0)
performance.early_stop_confidence: 0.95

# Batch processing
performance.batch_size: 1
```

---

## 🐛 Troubleshooting

### Quick Diagnostics

```bash
cd src
python test_integration.py
```

### Common Issues

| Issue | Solution |
|-------|----------|
| **Models not found** | Run `python setup_models_colab.py` |
| **TF/PyTorch conflict** | `pip install tensorflow>=2.12 torch>=2.0` |
| **CUDA OOM** | Reduce `resize_max` to 512 or 630 |
| **No matches** | Lower `confidence_threshold` to 0.01 |
| **Slow** | Verify GPU: `nvidia-smi` |

Full troubleshooting: See [INTEGRATION.md](INTEGRATION.md#troubleshooting)

---

## 📈 Validation Workflow

### 1. Test Integration
```bash
python test_integration.py
```

### 2. Run Localization
```bash
python wildnav.py
```

### 3. Analyze Results
```bash
python plot_data.py
```

### 4. Compare Matchers
```python
# A/B testing
USE_OMNIGLUE = True  → Run → Save results as omniglue_results.csv
USE_OMNIGLUE = False → Run → Save results as superglue_results.csv
# Compare MAE, success rate
```

---

## 🔮 Future Enhancements

- [ ] Batch processing for parallel satellite patch matching
- [ ] Adaptive confidence thresholding based on image quality
- [ ] Temporal smoothing with Kalman filter
- [ ] Fine-tuning OmniGlue on aerial-specific datasets
- [ ] Real-time optimization for onboard UAV processing
- [ ] Web interface for remote localization

---

## 📖 References

### Papers
- **OmniGlue:** "Generalizable Feature Matching with Foundation Model Guidance" (CVPR 2024)
  [arXiv](https://arxiv.org/abs/2405.12979) | [GitHub](https://github.com/google-research/omniglue)

- **WildNav:** "Vision-Based GNSS-Free Localization for UAVs in the Wild" (IEEE ICMERR 2022)
  [DOI](https://doi.org/10.1109/ICMERR56497.2022.10097798) | [GitHub](https://github.com/TIERS/wildnav)

- **SuperGlue:** "Learning Feature Matching with Graph Neural Networks" (CVPR 2020)
  [GitHub](https://github.com/magicleap/SuperGluePretrainedNetwork)

- **DINOv2:** "Learning Robust Visual Features without Supervision" (2023)
  [GitHub](https://github.com/facebookresearch/dinov2)

### Resources
- [Original WildNav README](README.md)
- [Integration Details](INTEGRATION.md)
- [Colab Tutorial](COLAB_QUICKSTART.md)
- [Developer Guide](../CLAUDE.md)

---

## 🤝 Contributing

### Testing New Configurations

1. Modify `config_omniglue.yaml`
2. Run `python wildnav.py`
3. Compare with baseline results
4. Share findings!

### Reporting Issues

Include in your report:
- Output of `python test_integration.py`
- Configuration used (`config_omniglue.yaml`)
- Sample images (if possible)
- Error messages/logs

---

## 📜 License

- **WildNav:** Original license applies
- **OmniGlue:** Apache 2.0 License
- **Integration Code:** Inherits from parent projects

---

## 🎓 Citation

If you use this integration in your research:

```bibtex
@inproceedings{jiang2024omniglue,
  title={OmniGlue: Generalizable Feature Matching with Foundation Model Guidance},
  author={Jiang, Hanwen and Karpur, Arjun and Cao, Bingyi and Huang, Qixing and Araujo, Andre},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@inproceedings{wildnav2022,
  title={Vision-Based GNSS-Free Localization for UAVs in the Wild},
  booktitle={IEEE International Conference on Mechatronics, Electronics and Robotics Research},
  year={2022}
}
```

---

## ✨ Acknowledgments

- **TIERS Lab** - Original WildNav implementation
- **Google Research** - OmniGlue implementation
- **Meta AI** - DINOv2 foundation model
- **Magic Leap** - SuperGlue baseline

---

## 📞 Support

- **Integration Issues:** Open issue in this repository
- **WildNav Issues:** https://github.com/TIERS/wildnav/issues
- **OmniGlue Issues:** https://github.com/google-research/omniglue/issues

---

**Version:** 1.0
**Last Updated:** 2025-11-20
**Status:** ✅ Ready for Testing

---

Happy Localizing! 🛸📍
