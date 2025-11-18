"""
RoMa feature matching wrapper for WildNav
Uses RoMa (CVPR 2024) with DINOv2 backbone for robust dense matching
"""
from pathlib import Path
import cv2
import torch
import numpy as np

torch.set_grad_enabled(False)


def match_image():
    """
    Wrapper function for matching drone image against satellite map using RoMa

    Returns: satellite_map_index, center, located_image, features_mean, last_frame, num_matches
    """
    input_path = '../assets/map/'
    output_dir = "../results"
    image_glob = ['*.png', '*.jpg', '*.jpeg', '*.JPG']

    # RoMa configuration
    certainty_threshold = 0.5  # Minimum certainty for valid matches
    min_matches_threshold = 4  # Minimum matches needed for homography
    force_cpu = False

    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print('Running RoMa + DINOv2 inference on device \"{}\"'.format(device))

    # Initialize RoMa model
    from romatch import roma_outdoor
    matcher = roma_outdoor(device=device)
    print("RoMa model loaded (using DINOv2 backbone)")

    # Create output directory
    if output_dir is not None:
        print('==> Will write outputs to {}'.format(output_dir))
        Path(output_dir).mkdir(exist_ok=True)

    # Load query image (drone image)
    query_image_path = Path(input_path) / '1_query_image.png'
    if not query_image_path.exists():
        raise FileNotFoundError(f"Query image not found at {query_image_path}")

    # Load query image with OpenCV for later visualization
    image0_cv = cv2.imread(str(query_image_path), cv2.IMREAD_GRAYSCALE)

    # Get all satellite map images
    map_path = Path(input_path)
    satellite_images = []
    for pattern in image_glob:
        satellite_images.extend(sorted(map_path.glob(pattern)))

    # Remove query image from satellite list
    satellite_images = [img for img in satellite_images if img.name != '1_query_image.png']

    print(f'Found {len(satellite_images)} satellite images to match against\n')

    # Variables to track best match
    satellite_map_index = None
    best_center = None
    best_located_image = None
    best_features_mean = [0, 0]
    max_matches = 0

    # Iterate through all satellite images
    for idx, sat_path in enumerate(satellite_images):
        print(f"Processing satellite image {idx + 1}/{len(satellite_images)}: {sat_path.name}")

        try:
            # Match using RoMa
            warp, certainty = matcher.match(str(query_image_path), str(sat_path), device=device)

            # Sample sparse matches from dense warp
            matches, certainty_sampled = matcher.sample(warp, certainty)

            # Get image dimensions
            from PIL import Image
            query_img = Image.open(query_image_path)
            sat_img = Image.open(sat_path)
            H_query, W_query = query_img.size[1], query_img.size[0]
            H_sat, W_sat = sat_img.size[1], sat_img.size[0]

            # Convert to pixel coordinates
            kpts_query, kpts_sat = matcher.to_pixel_coordinates(
                matches, H_query, W_query, H_sat, W_sat
            )

            # Filter by certainty threshold
            valid_mask = certainty_sampled > certainty_threshold
            num_matches = valid_mask.sum().item()

            print(f"Matches found: {num_matches} (certainty > {certainty_threshold})")

            if num_matches >= min_matches_threshold:
                # Get valid matches
                kpts0_valid = kpts_query[valid_mask].cpu().numpy()
                kpts1_valid = kpts_sat[valid_mask].cpu().numpy()

                # Compute homography to get perspective transform
                try:
                    H, mask = cv2.findHomography(kpts0_valid, kpts1_valid, cv2.RANSAC, 5.0)

                    if H is not None:
                        # Get corners of query image
                        h0, w0 = image0_cv.shape
                        corners_query = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2)

                        # Transform corners to satellite image space
                        corners_sat = cv2.perspectiveTransform(corners_query, H)

                        # Calculate center of transformed region
                        center_x = np.mean(corners_sat[:, 0, 0])
                        center_y = np.mean(corners_sat[:, 0, 1])

                        # Normalize to [0, 1] range
                        center_ratio = [center_x / W_sat, center_y / H_sat]

                        # Calculate mean of matched features
                        features_mean = [np.mean(kpts1_valid[:, 0]), np.mean(kpts1_valid[:, 1])]

                        # Check if this is the best match so far
                        if num_matches > max_matches and center_ratio[0] < 1 and center_ratio[1] < 1:
                            satellite_map_index = idx
                            best_center = center_ratio
                            best_features_mean = features_mean
                            max_matches = num_matches

                            # Create visualization
                            image0_color = cv2.cvtColor(image0_cv, cv2.COLOR_GRAY2BGR)
                            image1_cv = cv2.imread(str(sat_path), cv2.IMREAD_GRAYSCALE)
                            image1_color = cv2.cvtColor(image1_cv, cv2.COLOR_GRAY2BGR)

                            # Draw matches
                            best_located_image = create_match_visualization(
                                image0_color, image1_color,
                                kpts0_valid, kpts1_valid,
                                show_keypoints=True
                            )

                            print(f"✓ New best match! Matches: {max_matches}, Center: {center_ratio}")

                except cv2.error as e:
                    print(f"Homography computation error: {e}")
            else:
                print("Not enough matches for homography")

        except Exception as e:
            print(f"Error processing {sat_path.name}: {e}")
            continue

    # Return results (same interface as lightglue_utils)
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


def create_match_visualization(image0, image1, kpts0, kpts1, show_keypoints=True, max_viz_matches=50):
    """
    Create side-by-side visualization of matches

    Args:
        image0: Query image (BGR)
        image1: Reference image (BGR)
        kpts0: Keypoints in image0 (Nx2 array)
        kpts1: Keypoints in image1 (Nx2 array)
        show_keypoints: Whether to draw keypoint circles
        max_viz_matches: Maximum number of matches to visualize

    Returns:
        Combined visualization image
    """
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]

    # Create side-by-side image
    h_max = max(h0, h1)
    viz = np.zeros((h_max, w0 + w1, 3), dtype=np.uint8)
    viz[:h0, :w0] = image0
    viz[:h1, w0:w0+w1] = image1

    # Limit number of matches to visualize
    num_viz = min(len(kpts0), max_viz_matches)

    # Draw matches
    for i in range(num_viz):
        pt0 = (int(kpts0[i, 0]), int(kpts0[i, 1]))
        pt1 = (int(kpts1[i, 0] + w0), int(kpts1[i, 1]))

        # Random color for this match
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Draw line
        cv2.line(viz, pt0, pt1, color, 1, cv2.LINE_AA)

        # Draw keypoint circles if requested
        if show_keypoints:
            cv2.circle(viz, pt0, 3, color, -1, cv2.LINE_AA)
            cv2.circle(viz, pt1, 3, color, -1, cv2.LINE_AA)

    # Add match count text
    cv2.putText(viz, f"Matches: {len(kpts0)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return viz
