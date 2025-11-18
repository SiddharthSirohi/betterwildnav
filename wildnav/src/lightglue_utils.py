"""
LightGlue feature matching wrapper for WildNav
Replaces SuperGlue with LightGlue for improved performance and speed
Supports both SuperPoint and DINOv3 feature extractors
"""
from pathlib import Path
import cv2
import torch
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from dinov2_extractor import DINOv2Extractor

torch.set_grad_enabled(False)


def match_image(use_dinov2=True):
    """
    Wrapper function for matching drone image against satellite map using LightGlue

    Args:
        use_dinov2: If True, use DINOv2 feature extractor. If False, use SuperPoint.

    Returns: satellite_map_index, center, located_image, features_mean, last_frame, max_matches
    """
    input_path = '../assets/map/'
    output_dir = "../results"
    image_glob = ['*.png', '*.jpg', '*.jpeg', '*.JPG']

    # LightGlue configuration parameters
    max_keypoints = 2048  # Number of keypoints to extract per image
    resize_dim = 800  # Resize images to this dimension
    force_cpu = False  # Set to True to run on CPU (slower)
    show_keypoints = True  # Visualize keypoints in output
    no_display = False  # Set to True to disable CV2 window

    # LightGlue matcher parameters
    filter_threshold = 0.1  # Match confidence threshold
    depth_confidence = 0.95  # Early stopping control (lower = faster, -1 to disable)
    width_confidence = 0.99  # Point pruning control (lower = fewer points, -1 to disable)

    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'

    # Select feature extractor
    if use_dinov2:
        print('Running LightGlue + DINOv2 inference on device \"{}\"'.format(device))
        # Use ViT-S/14 model for efficiency (21M params)
        # Options: dinov2_vits14 (21M), dinov2_vitb14 (86M), dinov2_vitl14 (300M), dinov2_vitg14 (1.1B)
        # All DINOv2 descriptors are projected to 256-dim for compatibility with LightGlue
        extractor = DINOv2Extractor(
            model_name='dinov2_vits14',
            max_keypoints=max_keypoints,
            device=device
        ).eval()
    else:
        print('Running LightGlue + SuperPoint inference on device \"{}\"'.format(device))
        extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)

    # Initialize LightGlue matcher
    # Use 'disk' features type (default 128-dim, matching our DINOv2 projection)
    matcher = LightGlue(
        features='disk',
        filter_threshold=filter_threshold,
        depth_confidence=depth_confidence,
        width_confidence=width_confidence
    ).eval().to(device)

    # Create output directory
    if output_dir is not None:
        print('==> Will write outputs to {}'.format(output_dir))
        Path(output_dir).mkdir(exist_ok=True)

    # Load query image (drone image)
    query_image_path = Path(input_path) / '1_query_image.png'
    if not query_image_path.exists():
        raise FileNotFoundError(f"Query image not found at {query_image_path}")

    # Load and preprocess query image
    image0_tensor = load_image(str(query_image_path)).to(device)
    image0_cv = cv2.imread(str(query_image_path), cv2.IMREAD_GRAYSCALE)

    # Resize query image for consistent processing
    h0, w0 = image0_cv.shape
    if resize_dim > 0:
        scale = resize_dim / max(h0, w0)
        new_h, new_w = int(h0 * scale), int(w0 * scale)
        image0_cv = cv2.resize(image0_cv, (new_w, new_h))

    print(f'Will resize max dimension to {resize_dim}')

    # Extract features from query image
    print("Extracting features from query image...")
    feats0 = extractor.extract(image0_tensor)

    # Get all satellite map images
    map_path = Path(input_path)
    satellite_images = []
    for pattern in image_glob:
        satellite_images.extend(sorted(map_path.glob(pattern)))

    # Filter out the query image itself
    satellite_images = [img for img in satellite_images if img.name != '1_query_image.png']

    if not satellite_images:
        raise FileNotFoundError(f"No satellite images found in {input_path}")

    print(f"Found {len(satellite_images)} satellite images to match against")

    # Create display window if needed
    if not no_display:
        cv2.namedWindow('LightGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('LightGlue matches', 640*2, 480*2)

    # Tracking variables for best match
    satellite_map_index = None
    max_matches = -1
    best_center = None
    best_located_image = None
    best_features_mean = [0, 0]
    best_mkpts0 = None
    best_mkpts1 = None

    # Iterate through all satellite images
    for index, sat_image_path in enumerate(satellite_images):
        print(f"\nProcessing satellite image {index + 1}/{len(satellite_images)}: {sat_image_path.name}")

        # Load satellite image
        image1_tensor = load_image(str(sat_image_path)).to(device)
        image1_cv = cv2.imread(str(sat_image_path), cv2.IMREAD_GRAYSCALE)

        # Resize satellite image
        h1, w1 = image1_cv.shape
        if resize_dim > 0:
            scale = resize_dim / max(h1, w1)
            new_h, new_w = int(h1 * scale), int(w1 * scale)
            image1_cv = cv2.resize(image1_cv, (new_w, new_h))

        # Extract features from satellite image
        feats1 = extractor.extract(image1_tensor)

        # Match features using LightGlue
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0_rbd, feats1_rbd, matches01_rbd = [rbd(x) for x in [feats0, feats1, matches01]]

        # Get matched keypoints
        matches = matches01_rbd['matches']  # Shape: (K, 2)
        kpts0 = feats0_rbd['keypoints'].cpu().numpy()  # All keypoints in image0
        kpts1 = feats1_rbd['keypoints'].cpu().numpy()  # All keypoints in image1

        # Get coordinates of matched keypoints
        if len(matches) > 0:
            mkpts0 = kpts0[matches[:, 0]]  # Matched keypoints in image0
            mkpts1 = kpts1[matches[:, 1]]  # Matched keypoints in image1
            num_matches = len(matches)
        else:
            mkpts0 = np.array([])
            mkpts1 = np.array([])
            num_matches = 0

        print(f"Matches found: {num_matches}")

        # Attempt to compute homography if we have enough matches
        if num_matches >= 4:
            try:
                # Compute homography using RANSAC
                M, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)

                if M is not None:
                    # Transform query image corners to satellite image space
                    h, w = image0_cv.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

                    try:
                        dst = cv2.perspectiveTransform(pts, M)

                        # Calculate center of transformed region
                        moments = cv2.moments(dst)
                        if moments["m00"] != 0:
                            cX = int(moments["m10"] / moments["m00"])
                            cY = int(moments["m01"] / moments["m00"])
                            center = (cX, cY)

                            # Normalize center to ratio (0-1 range)
                            center_ratio = (cX / image1_cv.shape[1], cY / image1_cv.shape[0])

                            # Check if this is the best match so far
                            if num_matches > max_matches and center_ratio[0] < 1 and center_ratio[1] < 1:
                                satellite_map_index = index
                                max_matches = num_matches
                                best_center = center_ratio
                                best_features_mean = np.mean(mkpts0, axis=0)
                                best_mkpts0 = mkpts0.copy()
                                best_mkpts1 = mkpts1.copy()

                                # Create visualization
                                image1_cv_color = cv2.cvtColor(image1_cv, cv2.COLOR_GRAY2BGR)
                                image1_cv_color = cv2.polylines(image1_cv_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                                cv2.circle(image1_cv_color, center, radius=10, color=(255, 0, 255), thickness=5)

                                image0_cv_color = cv2.cvtColor(image0_cv, cv2.COLOR_GRAY2BGR)
                                cv2.circle(image0_cv_color, (int(best_features_mean[0]), int(best_features_mean[1])),
                                          radius=10, color=(255, 0, 0), thickness=2)

                                # Create side-by-side visualization
                                best_located_image = create_match_visualization(
                                    image0_cv_color, image1_cv_color, best_mkpts0, best_mkpts1, show_keypoints
                                )

                                print(f"✓ New best match! Matches: {max_matches}, Center: {center_ratio}")

                    except cv2.error as e:
                        print(f"Perspective transform error: {e}")

            except cv2.error as e:
                print(f"Homography computation error: {e}")
        else:
            print("Not enough matches for homography (need at least 4)")

        # Display current match if window is enabled
        if not no_display and num_matches > 0:
            display_img = create_match_visualization(
                cv2.cvtColor(image0_cv, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(image1_cv, cv2.COLOR_GRAY2BGR),
                mkpts0, mkpts1, show_keypoints
            )
            cv2.imshow('LightGlue matches', display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('Exiting (via q)')
                break

    # Cleanup
    if not no_display:
        cv2.destroyAllWindows()

    # Return results
    if satellite_map_index is not None:
        print(f"\n{'='*50}")
        print(f"Best match found: Satellite image #{satellite_map_index}")
        print(f"Total matches: {max_matches}")
        print(f"Center position: {best_center}")
        print(f"{'='*50}")
        return satellite_map_index, best_center, best_located_image, best_features_mean, image0_cv, max_matches
    else:
        print("\nNo valid matches found!")
        return None, None, None, [0, 0], image0_cv, 0


def create_match_visualization(image0, image1, mkpts0, mkpts1, show_keypoints=True):
    """
    Create a side-by-side visualization of matched keypoints
    """
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]

    # Create output image
    h = max(h0, h1)
    w = w0 + w1
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # Place images side by side
    out[:h0, :w0] = image0
    out[:h1, w0:w0+w1] = image1

    # Draw matches
    if len(mkpts0) > 0 and len(mkpts1) > 0:
        # Ensure arrays are properly shaped
        mkpts0 = np.atleast_2d(mkpts0)
        mkpts1 = np.atleast_2d(mkpts1)

        for i in range(len(mkpts0)):
            # Ensure we have valid 2D coordinates
            if mkpts0[i].size >= 2 and mkpts1[i].size >= 2:
                pt0 = (int(mkpts0[i][0]), int(mkpts0[i][1]))
                pt1 = (int(mkpts1[i][0] + w0), int(mkpts1[i][1]))

                # Draw keypoints
                if show_keypoints:
                    cv2.circle(out, pt0, 3, (0, 255, 0), -1)
                    cv2.circle(out, pt1, 3, (0, 255, 0), -1)

                # Draw matching line
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.line(out, pt0, pt1, color, 1, cv2.LINE_AA)

    # Add text with match count
    text = f"Matches: {len(mkpts0)}"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return out
