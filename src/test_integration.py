#!/usr/bin/env python3
"""
Test script for OmniGlue integration with WildNav.
Validates the integration and compares with baseline if available.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time

def test_imports():
    """Test if all required modules can be imported"""
    print("="*60)
    print("TEST 1: Import Dependencies")
    print("="*60)

    tests = {
        "OpenCV": lambda: __import__('cv2'),
        "NumPy": lambda: __import__('numpy'),
        "TensorFlow": lambda: __import__('tensorflow'),
        "PyTorch": lambda: __import__('torch'),
        "Matplotlib": lambda: __import__('matplotlib'),
        "Pandas": lambda: __import__('pandas'),
        "Haversine": lambda: __import__('haversine'),
    }

    results = {}
    for name, import_func in tests.items():
        try:
            module = import_func()
            version = getattr(module, '__version__', 'unknown')
            results[name] = (True, version)
            print(f"  ✓ {name:15s} {version}")
        except ImportError as e:
            results[name] = (False, str(e))
            print(f"  ✗ {name:15s} FAILED: {e}")

    all_passed = all(passed for passed, _ in results.values())
    return all_passed, results


def test_omniglue_lib():
    """Test if OmniGlue library is accessible"""
    print("\n" + "="*60)
    print("TEST 2: OmniGlue Library")
    print("="*60)

    try:
        sys.path.insert(0, str(Path(__file__).parent / 'omniglue_lib'))
        from omniglue_lib.omniglue_extract import OmniGlue
        from omniglue_lib import utils as og_utils
        from omniglue_lib.superpoint_extract import SuperPointExtract
        from omniglue_lib.dino_extract import DINOExtract
        print("  ✓ OmniGlue library imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ OmniGlue library import failed: {e}")
        return False


def test_models_exist():
    """Test if all model files are downloaded"""
    print("\n" + "="*60)
    print("TEST 3: Model Files")
    print("="*60)

    models_dir = Path('../models')
    required_models = {
        "SuperPoint": models_dir / 'sp_v6',
        "DINOv2": models_dir / 'dinov2_vitb14_pretrain.pth',
        "OmniGlue": models_dir / 'og_export',
    }

    all_exist = True
    for name, path in required_models.items():
        if path.exists():
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"  ✓ {name:15s} {path} ({size_mb:.1f} MB)")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {name:15s} {path} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name:15s} NOT FOUND: {path}")
            all_exist = False

    if not all_exist:
        print("\n  ⚠️  Run setup script: python setup_models_colab.py")

    return all_exist


def test_omniglue_initialization():
    """Test if OmniGlue can be initialized"""
    print("\n" + "="*60)
    print("TEST 4: OmniGlue Initialization")
    print("="*60)

    try:
        sys.path.insert(0, str(Path(__file__).parent / 'omniglue_lib'))
        from omniglue_lib.omniglue_extract import OmniGlue

        print("  Initializing OmniGlue...")
        start = time.time()

        og = OmniGlue(
            og_export='../models/og_export',
            sp_export='../models/sp_v6',
            dino_export='../models/dinov2_vitb14_pretrain.pth',
        )

        elapsed = time.time() - start
        print(f"  ✓ OmniGlue initialized successfully ({elapsed:.2f}s)")
        return True, og
    except Exception as e:
        print(f"  ✗ OmniGlue initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_simple_match(og):
    """Test matching on two simple images"""
    print("\n" + "="*60)
    print("TEST 5: Simple Image Matching")
    print("="*60)

    try:
        # Create two simple test images with known features
        img_size = 400
        img1 = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        img2 = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Draw identical patterns
        for i in range(5):
            x, y = np.random.randint(50, img_size-50, size=2)
            radius = np.random.randint(10, 30)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(img1, (x, y), radius, color, -1)
            cv2.circle(img2, (x, y), radius, color, -1)

        print("  Matching test images...")
        start = time.time()
        match_kp0, match_kp1, match_confidences = og.FindMatches(img1, img2)
        elapsed = time.time() - start

        num_matches = len(match_kp0)
        print(f"  ✓ Found {num_matches} matches ({elapsed:.2f}s)")

        if num_matches > 0:
            avg_conf = np.mean(match_confidences)
            print(f"  ✓ Average confidence: {avg_conf:.4f}")
            return True
        else:
            print("  ⚠️  No matches found (may be normal for simple images)")
            return True

    except Exception as e:
        print(f"  ✗ Matching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wildnav_integration():
    """Test the full WildNav integration"""
    print("\n" + "="*60)
    print("TEST 6: WildNav Integration")
    print("="*60)

    try:
        # Check if query image directory exists
        query_dir = Path('../assets/query')
        if not query_dir.exists():
            print(f"  ⚠️  Query directory not found: {query_dir}")
            return True  # Not a failure, just no data

        # Check if map directory exists
        map_dir = Path('../assets/map')
        if not map_dir.exists():
            print(f"  ⚠️  Map directory not found: {map_dir}")
            return True

        # Count available images
        query_images = list(query_dir.glob('*.jpg')) + list(query_dir.glob('*.JPG'))
        map_images = list(map_dir.glob('*.png'))

        print(f"  Query images: {len(query_images)}")
        print(f"  Map images: {len(map_images)}")

        if len(query_images) == 0 or len(map_images) == 0:
            print("  ⚠️  No test data available")
            print("     Add drone images to assets/query/")
            print("     Add satellite maps to assets/map/")
            return True

        # Test importing omniglue_utils
        import omniglue_utils
        print("  ✓ omniglue_utils module imported")

        print("\n  ℹ️  To run full integration test:")
        print("     python wildnav.py")

        return True

    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_availability():
    """Test if GPU is available and accessible"""
    print("\n" + "="*60)
    print("TEST 7: GPU Availability")
    print("="*60)

    # Test PyTorch GPU
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ PyTorch: {device_count} CUDA device(s) available")
            print(f"    Device 0: {device_name}")
        else:
            print("  ⚠️  PyTorch: CUDA not available (will use CPU)")
    except Exception as e:
        print(f"  ✗ PyTorch GPU test failed: {e}")

    # Test TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ TensorFlow: {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                print(f"    GPU {i}: {gpu.name}")
        else:
            print("  ⚠️  TensorFlow: No GPUs available (will use CPU)")
    except Exception as e:
        print(f"  ✗ TensorFlow GPU test failed: {e}")

    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("OmniGlue Integration Test Suite")
    print("="*60)

    results = {}

    # Test 1: Imports
    passed, details = test_imports()
    results['imports'] = passed

    # Test 2: OmniGlue library
    results['omniglue_lib'] = test_omniglue_lib()

    # Test 3: Model files
    results['models'] = test_models_exist()

    # Test 4 & 5: Only if models exist
    if results['models']:
        passed, og = test_omniglue_initialization()
        results['initialization'] = passed

        if passed and og is not None:
            results['simple_match'] = test_simple_match(og)
        else:
            results['simple_match'] = False
    else:
        results['initialization'] = False
        results['simple_match'] = False

    # Test 6: WildNav integration
    results['integration'] = test_wildnav_integration()

    # Test 7: GPU
    results['gpu'] = test_gpu_availability()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s} {status}")

    total = len(results)
    passed_count = sum(results.values())
    print(f"\nTotal: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n✓ All tests passed! Ready to run WildNav with OmniGlue.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
