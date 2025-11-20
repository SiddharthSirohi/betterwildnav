"""
OmniGlue wrapper for WildNav integration.
Drop-in replacement for superglue_utils.py
"""

from pathlib import Path
import cv2
import numpy as np
import sys
import time

# Import OmniGlue components
sys.path.insert(0, str(Path(__file__).parent / 'omniglue_lib'))
from omniglue_lib.omniglue_extract import OmniGlue
from omniglue_lib import utils as og_utils


class Timer:
    """Simple timer for profiling"""
    def __init__(self):
        self.times = {}
        self.start_time = None

    def update(self, name):
        if self.start_time is None:
            self.start_time = time.time()
        else:
            if name not in self.times:
                self.times[name] = []
            self.times[name].append(time.time() - self.start_time)
            self.start_time = time.time()

    def print(self):
        for name, times_list in self.times.items():
            avg_time = np.mean(times_list)
            print(f'{name}: {avg_time:.3f}s')


def match_image():
    """
    Wrapper function for matching two images using OmniGlue.
    Provides an interface compatible with original superglue_utils.match_image()

    Returns:
        satellite_map_index: Index of best matching satellite patch (int)
        center: Normalized (x, y) center coordinates in matched patch (tuple of floats, 0-1 range)
        located_image: Visualization image with matches drawn (numpy array)
        features_mean: Mean (x, y) of matched keypoints in query (numpy array, shape (2,))
        last_frame: Query image (numpy array)
        max_matches: Number of matches in best patch (int)
    """

    # Configuration parameters
    input_dir = '../assets/map/'
    output_dir = "../results"
    image_glob = ['*.png', '*.jpg', '*.jpeg', '*.JPG']

    # OmniGlue parameters
    confidence_threshold = 0.02  # Filter matches below this confidence
    resize_max = 800  # Max dimension for image resizing (to match original SuperGlue)

    # RANSAC parameters (matching original)
    ransac_threshold = 5.0  # pixels
    min_matches_for_homography = 4

    # Visualization
    show_keypoints = True
    no_display = True  # Disable GUI in Colab

    print('='*50)
    print('OmniGlue Feature Matching')
    print('='*50)

    # Initialize OmniGlue model
    print('Loading OmniGlue (SuperPoint + DINOv2 + Matcher)...')
    timer = Timer()
    timer.update('init')

    try:
        og = OmniGlue(
            og_export='../models/og_export',
            sp_export='../models/sp_v6',
            dino_export='../models/dinov2_vitb14_pretrain.pth',
        )
        print(f'✓ OmniGlue loaded successfully')
    except Exception as e:
        print(f'✗ Error loading OmniGlue: {e}')
        print('Make sure models are downloaded. Run: python setup_models_colab.py')
        raise

    timer.update('model_load')

    # Load query image (drone image)
    query_path = Path(input_dir) / '1_query_image.png'
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    query_image = cv2.imread(str(query_path))
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    query_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    print(f'Query image loaded: {query_image.shape}')
    timer.update('load_query')

    # Get all satellite map images
    sat_images = []
    for pattern in image_glob:
        sat_images.extend(sorted(Path(input_dir).glob(pattern)))

    # Remove query image from list
    sat_images = [img for img in sat_images if img.name != '1_query_image.png']
    print(f'Found {len(sat_images)} satellite images to match against')

    if not sat_images:
        raise FileNotFoundError(f"No satellite images found in {input_dir}")

    # Tracking variables for best match
    satellite_map_index = None
    max_matches = -1
    best_center = None
    best_located_image = None
    best_features_mean = np.array([0, 0])
    best_confidence_sum = -1
    second_best_confidence_sum = -1
    best_homography_det = None

    # Iterate through all satellite patches
    for idx, sat_path in enumerate(sat_images):
        print(f'\n[{idx+1}/{len(sat_images)}] Matching against: {sat_path.name}')

        # Load satellite image
        sat_image = cv2.imread(str(sat_path))
        if sat_image is None:
            print(f'  ✗ Failed to load {sat_path.name}')
            continue

        sat_rgb = cv2.cvtColor(sat_image, cv2.COLOR_BGR2RGB)
        timer.update('load_sat')

        # Perform OmniGlue matching
        try:
            match_kp0, match_kp1, match_confidences = og.FindMatches(query_rgb, sat_rgb)
            timer.update('matching')
        except Exception as e:
            print(f'  ✗ Matching failed: {e}')
            continue

        print(f'  Raw matches: {len(match_kp0)}')

        # Filter by confidence threshold
        keep_idx = match_confidences > confidence_threshold
        match_kp0_filtered = match_kp0[keep_idx]
        match_kp1_filtered = match_kp1[keep_idx]
        match_confidences_filtered = match_confidences[keep_idx]

        num_filtered = len(match_kp0_filtered)
        print(f'  Filtered matches (conf > {confidence_threshold}): {num_filtered}')

        if num_filtered < min_matches_for_homography:
            print(f'  ✗ Not enough matches for homography (need {min_matches_for_homography})')
            continue

        # Apply RANSAC homography for geometric verification
        try:
            M, mask = cv2.findHomography(
                match_kp0_filtered,
                match_kp1_filtered,
                cv2.RANSAC,
                ransac_threshold
            )

            if M is None:
                print(f'  ✗ Homography estimation failed')
                continue

            # Calculate homography determinant for quality check
            homography_det = np.linalg.det(M[:2, :2])  # Only use top-left 2x2 for rotation/scale part

            # Filter matches by RANSAC inliers
            inlier_mask = mask.ravel().astype(bool)
            mkpts0 = match_kp0_filtered[inlier_mask]
            mkpts1 = match_kp1_filtered[inlier_mask]
            confidences = match_confidences_filtered[inlier_mask]

            num_inliers = len(mkpts0)
            print(f'  RANSAC inliers: {num_inliers}')

            if num_inliers < min_matches_for_homography:
                print(f'  ✗ Not enough inliers after RANSAC')
                continue

        except Exception as e:
            print(f'  ✗ Homography error: {e}')
            continue

        timer.update('homography')

        # Calculate perspective transform center
        try:
            h, w = query_gray.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Calculate centroid of transformed region
            moments = cv2.moments(dst)
            if moments["m00"] == 0:
                print(f'  ✗ Invalid moments')
                continue

            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            center_px = (cX, cY)

            # Normalize center coordinates (0-1 range)
            center_normalized = (cX / sat_image.shape[1], cY / sat_image.shape[0])

            # Validate center is within image bounds
            if center_normalized[0] >= 1.0 or center_normalized[1] >= 1.0:
                print(f'  ✗ Center outside bounds: {center_normalized}')
                continue

            print(f'  ✓ Match center: {center_normalized}')

        except Exception as e:
            print(f'  ✗ Perspective transform error: {e}')
            continue

        timer.update('transform')

        # Calculate mean feature position in query image
        features_mean = np.mean(mkpts0, axis=0)
        confidence_sum = np.sum(confidences)

        # Determine if this is the best match
        # Primary criterion: number of matches
        # Secondary criterion: sum of confidences
        is_better = False
        if num_inliers > max_matches:
            is_better = True
        elif num_inliers == max_matches and confidence_sum > best_confidence_sum:
            is_better = True

        if is_better:
            print(f'  ★ NEW BEST MATCH! (matches: {num_inliers}, conf_sum: {confidence_sum:.2f})')

            # Create visualization
            # Draw transformed region boundary
            sat_viz = sat_image.copy()
            sat_viz = cv2.polylines(sat_viz, [np.int32(dst)], True, (255, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(sat_viz, center_px, radius=10, color=(255, 0, 255), thickness=5)

            # Draw feature mean on query
            query_viz = query_image.copy()
            cv2.circle(query_viz, (int(features_mean[0]), int(features_mean[1])),
                      radius=10, color=(255, 0, 0), thickness=2)

            # Create side-by-side visualization with matches
            located_image = create_match_visualization(
                query_viz, sat_viz, mkpts0, mkpts1, confidences, show_keypoints
            )

            # Track second-best confidence for ratio calculation
            if best_confidence_sum > 0:
                second_best_confidence_sum = max(second_best_confidence_sum, best_confidence_sum)

            # Update best match
            satellite_map_index = idx
            max_matches = num_inliers
            best_center = center_normalized
            best_located_image = located_image
            best_features_mean = features_mean
            best_confidence_sum = confidence_sum
            best_homography_det = homography_det
        elif confidence_sum > second_best_confidence_sum:
            # Track second-best even if not the best
            second_best_confidence_sum = confidence_sum

        timer.update('viz')

    # Final results
    print('\n' + '='*50)
    if satellite_map_index is not None:
        print(f'✓ Best match found:')
        print(f'  Satellite patch index: {satellite_map_index}')
        print(f'  Satellite patch: {sat_images[satellite_map_index].name}')
        print(f'  Number of matches: {max_matches}')
        print(f'  Center (normalized): {best_center}')
    else:
        print('✗ No valid match found')
    print('='*50)

    timer.print()

    # Calculate confidence ratio
    confidence_ratio = best_confidence_sum / second_best_confidence_sum if second_best_confidence_sum > 0 else float('inf')

    return (
        satellite_map_index,
        best_center,
        best_located_image,
        best_features_mean,
        query_image,
        max_matches,
        confidence_ratio,
        best_homography_det,
        timer  # Return timer for detailed timing info
    )


def create_match_visualization(img0, img1, kpts0, kpts1, confidences, show_keypoints=True):
    """
    Create side-by-side visualization of matched images.

    Args:
        img0: Query image (BGR)
        img1: Satellite image (BGR) with boundary/center already drawn
        kpts0: Matched keypoints in img0 (N, 2)
        kpts1: Matched keypoints in img1 (N, 2)
        confidences: Match confidences (N,)
        show_keypoints: Whether to draw keypoints

    Returns:
        Combined visualization image
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Create side-by-side canvas
    h_max = max(h0, h1)
    w_total = w0 + w1
    canvas = np.zeros((h_max, w_total, 3), dtype=np.uint8)

    # Place images
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0+w1] = img1

    # Draw matches
    num_matches = min(len(kpts0), 100)  # Limit for visibility
    if num_matches > 0:
        # Sort by confidence and take top matches
        if len(confidences) > num_matches:
            top_idx = np.argsort(confidences)[-num_matches:]
        else:
            top_idx = np.arange(len(kpts0))

        for idx in top_idx:
            pt0 = tuple(kpts0[idx].astype(int))
            pt1 = tuple((kpts1[idx] + np.array([w0, 0])).astype(int))

            # Color by confidence (green = high, red = low)
            conf = confidences[idx]
            color = (int(255 * (1 - conf)), int(255 * conf), 0)

            # Draw line
            cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)

            # Draw keypoints
            if show_keypoints:
                cv2.circle(canvas, pt0, 3, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)

    return canvas
