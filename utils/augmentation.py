from random import randint, uniform
import cv2
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

from scipy.ndimage import map_coordinates

def bilinear_sample_displacement(disp_map, x, y):
    """Sample displacement at subpixel coordinates using bilinear interpolation."""
    return map_coordinates(disp_map, [[y], [x]], order=1, mode='reflect')[0]


# More realistic session-to-session transform candidates for fingerprints.
# Keep legacy names supported; new pipeline prefers a single affine jitter.
transforms = [
    "affine",               # small rotation/translation/scale/shear combined
    "elastic_transform",    # subtle skin deformation
    "gaussian_blur",        # slight defocus
    "motion_blur",          # finger slip
    "noise",                # sensor noise / dust specks
    "brightness_contrast_gamma",  # scanner exposure variability
    "clahe",                # local contrast changes due to dryness/wetness
    "jpeg_compress",        # compression artifacts
]


def augment_image(image, annotation, min_points=5):
    """Apply multiple random transformations to an image, each type at most once.

    Notes:
    - Designed for fingerprint images across sessions: mild pose (affine),
      subtle elastic deformation, photometric jitter, slight blur/noise.
    - Preserves keypoints under geometric transforms; photometric transforms
      do not alter keypoint coordinates.
    """
    # Store original image and annotations in case we need to fall back
    original_image = image.copy()
    original_annotations = annotation.copy()
    
    # Create a random permutation of transformations
    transform_permutation = np.random.permutation(transforms)

    # Decide how many transformations to apply (1 to ~half)
    num_transforms = randint(1, max(2, math.ceil(len(transforms) / 2)))
    
    # Select the first num_transforms from the permutation
    selected_transforms = transform_permutation[:num_transforms]
    
    # Start with original image and annotations
    transformed_image = image.copy()
    transformed_annotations = annotation.copy()
    
    # Maximum number of attempts to get valid transformations
    max_attempts = 3
    attempts = 0
    
    while attempts < max_attempts:
        # Apply selected transformations sequentially
        temp_image = transformed_image.copy()
        temp_annotations = transformed_annotations.copy()
        
        for transform_type in selected_transforms:
            temp_image, temp_annotations = apply_single_transform(
                temp_image, temp_annotations, transform_type
            )
        
        # Check if we have enough valid keypoints
        if len(temp_annotations) >= min_points:
            return temp_image, temp_annotations
        
        # If not enough points, try a different subset of transformations
        attempts += 1
        # Use fewer transformations each time to increase chance of keeping points
        selected_transforms = transform_permutation[:max(1, num_transforms - attempts)]
    
    # If we couldn't get enough points with random transformations,
    # fall back to minimal transformation (just resize and crop)
    h, w = original_image.shape[:2]
    
    # Resize with no other transformations
    resized = cv2.resize(original_image, (320, 320), interpolation=cv2.INTER_LINEAR)
    scale_x, scale_y = 320 / w, 320 / h
    resized_annotations = [[id_, x * scale_x, y * scale_y] for id_, x, y in original_annotations]
    
    # Center crop
    crop_h, crop_w = 240, 320
    start_x = (320 - crop_w) // 2  # This will be 0
    start_y = (320 - crop_h) // 2  # This will be 40
    cropped = resized[start_y:start_y + crop_h, start_x:start_x + crop_w]
    
    # Only keep annotations that are within the crop
    cropped_annotations = [
        [id_, x - start_x, y - start_y] 
        for id_, x, y in resized_annotations 
        if start_x <= x < start_x + crop_w and start_y <= y < start_y + crop_h
    ]
    
    # If we still don't have enough points, create synthetic ones to meet minimum
    if len(cropped_annotations) < min_points:
        print(f"Warning: Returning Original to meet minimum requirement of {min_points} points")
        return image, annotation  # Return original image and annotations if we can't meet the requirement
    
    return cropped, cropped_annotations

def apply_single_transform(image, annotation, transformation_type):
    """Apply a single transformation to an image.

    Supported types:
    - "affine": small rotate/translate/scale/shear
    - "rotate", "translate": legacy cases mapped to affine
    - "elastic_transform": smooth non-rigid displacement
    - "gaussian_blur": slight defocus
    - "motion_blur": linear motion blur
    - "noise": gaussian or salt-and-pepper
    - "brightness_contrast_gamma": photometric jitter
    - "clahe": local contrast equalization
    - "jpeg_compress": add JPEG artifacts
    - "vertical_flip", "horizontal_flip": legacy (kept but not preferred)
    """
    h, w = image.shape[:2]
    
    if transformation_type in ("affine", "translate", "rotate"):
        # Affine jitter: small rotation/translation/scale/shear combined
        # Session-to-session on scanners tends to be mild
        angle = uniform(-15, 15)
        dx, dy = randint(-20, 20), randint(-20, 20)
        scale = uniform(0.9, 1.1)
        shear_deg = uniform(-5, 5)
        shear = math.tan(math.radians(shear_deg))

        cx, cy = w / 2.0, h / 2.0

        # Build affine matrix in homogeneous coords
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        RS = np.array([[scale * cos_a, -scale * sin_a, 0],
                       [scale * sin_a,  scale * cos_a, 0],
                       [0, 0, 1]], dtype=np.float32)
        SH = np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        T2 = np.array([[1, 0, cx + dx], [0, 1, cy + dy], [0, 0, 1]], dtype=np.float32)
        M = T2 @ SH @ RS @ T1
        matrix = M[:2, :]

        transformed_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        transformed_annotations = []
        for id_, x, y in annotation:
            coords = (M @ np.array([x, y, 1.0], dtype=np.float32))
            nx, ny = float(coords[0]), float(coords[1])
            if 0 <= nx < w and 0 <= ny < h:
                transformed_annotations.append([id_, nx, ny])
    elif transformation_type == "vertical_flip":
        transformed_image = cv2.flip(image, 0)
        transformed_annotations = [[id_, x, h - y - 1] for id_, x, y in annotation if 0 <= x < w and 0 <= h - y < h]
    elif transformation_type == "horizontal_flip":
        transformed_image = cv2.flip(image, 1)
        transformed_annotations = [[id_, w - x - 1, y] for id_, x, y in annotation if 0 <= w - x < w and 0 <= y < h]
        
    elif transformation_type == "elastic_transform":
        # Subtle skin deformation
        sigma = np.random.uniform(8, 20)
        alpha = np.random.uniform(0, 120)

        # Random displacement fields
        dx = np.random.rand(h, w) * 2 - 1
        dy = np.random.rand(h, w) * 2 - 1
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        # Apply to image
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x_coords + dx).astype(np.float32)
        map_y = (y_coords + dy).astype(np.float32)

        transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Apply to annotations using bilinear sampling
        transformed_annotations = []
        for id_, x, y in annotation:
            if 0 <= x < w and 0 <= y < h:
                new_x = x + bilinear_sample_displacement(dx, x, y)
                new_y = y + bilinear_sample_displacement(dy, x, y)
                if 0 <= new_x < w and 0 <= new_y < h:
                    transformed_annotations.append([id_, new_x, new_y])

    elif transformation_type == "gaussian_blur":
        # Slight defocus
        k = np.random.choice([3, 5])
        transformed_image = cv2.GaussianBlur(image, (k, k), sigmaX=0)
        transformed_annotations = [[id_, x, y] for id_, x, y in annotation]

    elif transformation_type == "motion_blur":
        degree = np.random.choice([7, 9, 11, 13])
        angle = randint(0, 180)
        
        # Create the motion blur kernel
        M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
        motion_blur_kernel = np.zeros((degree, degree))
        motion_blur_kernel[int((degree-1)/2), :] = 1
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / np.sum(motion_blur_kernel)
        
        # Apply the kernel to create motion blur effect
        transformed_image = cv2.filter2D(image, -1, motion_blur_kernel)
        
        # Motion blur doesn't change keypoint positions
        transformed_annotations = [[id_, x, y] for id_, x, y in annotation]
    elif transformation_type == "noise":
        # Randomly choose between Gaussian and Salt & Pepper noise
        noise_type = np.random.choice(["gaussian", "salt_and_pepper"])
        
        if noise_type == "gaussian":
            # Apply Gaussian noise - common in digital sensor noise
            mean = 0
            # Vary the noise intensity randomly (keep mild)
            sigma = np.random.uniform(0.5, 2.0)
            noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
            transformed_image = image.astype(np.float32) + noise
            # Clip values to valid range
            transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
            
        else:  # salt_and_pepper
            # Apply salt and pepper noise - simulates dust/dirt on scanner
            transformed_image = image.copy()
            # Noise density between 0.3% and 1%
            amount = np.random.uniform(0.003, 0.01)
            # Salt (white) vs pepper (black) ratio
            s_vs_p = np.random.uniform(0.3, 0.7)
            
            # Generate salt noise (white pixels)
            num_salt = int(np.ceil(amount * image.size * s_vs_p))
            salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
            transformed_image[salt_coords[0], salt_coords[1]] = 255
            
            # Generate pepper noise (black pixels)
            num_pepper = int(np.ceil(amount * image.size * (1 - s_vs_p)))
            pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
            transformed_image[pepper_coords[0], pepper_coords[1]] = 0
        
        # Noise doesn't change keypoint positions
        transformed_annotations = [[id_, x, y] for id_, x, y in annotation]

    elif transformation_type == "brightness_contrast_gamma":
        # Photometric jitter: adjust brightness, contrast, gamma
        img = image.astype(np.float32)
        # Contrast in [0.9, 1.1], brightness in [-20, 20]
        alpha_c = uniform(0.9, 1.1)
        beta_b = uniform(-20, 20)
        img = img * alpha_c + beta_b
        img = np.clip(img, 0, 255)
        # Gamma in [0.8, 1.2]
        gamma = uniform(0.8, 1.2)
        inv_gamma = 1.0 / max(gamma, 1e-6)
        img = np.power(img / 255.0, inv_gamma) * 255.0
        transformed_image = np.clip(img, 0, 255).astype(np.uint8)
        transformed_annotations = [[id_, x, y] for id_, x, y in annotation]

    elif transformation_type == "clahe":
        # Local contrast changes (dry/wet fingers)
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        clahe = cv2.createCLAHE(clipLimit=uniform(2.0, 3.0), tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        if len(image.shape) == 3 and image.shape[2] == 3:
            transformed_image = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
        else:
            transformed_image = cl
        transformed_annotations = [[id_, x, y] for id_, x, y in annotation]

    elif transformation_type == "jpeg_compress":
        # Simulate compression artifacts
        quality = randint(50, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        if result:
            transformed_image = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
        else:
            transformed_image = image
        transformed_annotations = [[id_, x, y] for id_, x, y in annotation]
            
        
    # Downsample to 320x320
    transformed_image = cv2.resize(transformed_image, (320, 320), interpolation=cv2.INTER_LINEAR)
    scale_x, scale_y = 320 / w, 320 / h
    transformed_annotations = [[id_, x * scale_x, y * scale_y] for id_, x, y in transformed_annotations]

    # Center crop to 240x320
    crop_h, crop_w = 240, 320
    start_x = (320 - crop_w) // 2
    start_y = (320 - crop_h) // 2
    cropped_image = transformed_image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    cropped_annotations = [[id_, x - start_x, y - start_y] for id_, x, y in transformed_annotations if start_x <= x < start_x + crop_w and start_y <= y < start_y + crop_h]

    return cropped_image, cropped_annotations


def augment_image_pair(image, annotation, min_points=5, min_common=4, max_attempts=5, n_jobs=2):
    """Generate two augmented views of the same fingerprint more efficiently.

    - Runs augmentations in parallel threads (OpenCV releases the GIL).
    - Ensures a minimum number of shared keypoint labels across the two views
      (after cropping), falling back to a minimal standardization when needed.

    Returns: (img1, annos1_filtered), (img2, annos2_filtered)
    """
    assert n_jobs >= 2

    def _standardize(img, ann):
        h, w = img.shape[:2]
        resized = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
        sx, sy = 320 / w, 320 / h
        ann = [[i, x * sx, y * sy] for i, x, y in ann]
        crop_h, crop_w = 240, 320
        start_x = (320 - crop_w) // 2
        start_y = (320 - crop_h) // 2
        cropped = resized[start_y:start_y + crop_h, start_x:start_x + crop_w]
        ann = [[i, x - start_x, y - start_y] for i, x, y in ann if start_x <= x < start_x + crop_w and start_y <= y < start_y + crop_h]
        return cropped, ann

    for _ in range(max_attempts):
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(augment_image, image, annotation, min_points) for _ in range(2)]
            (img1, annos1), (img2, annos2) = futs[0].result(), futs[1].result()
        labels1 = {a[0] for a in annos1}
        labels2 = {a[0] for a in annos2}
        common = labels1.intersection(labels2)
        if len(common) >= min_common:
            annos1_f = [a for a in annos1 if a[0] in common]
            annos2_f = [a for a in annos2 if a[0] in common]
            return (img1, annos1_f), (img2, annos2_f)

    # Fallback: standardized pair with identical geometry
    img1, ann1 = _standardize(image, annotation)
    img2, ann2 = _standardize(image, annotation)
    labels = {a[0] for a in ann1}
    ann2 = [a for a in ann2 if a[0] in labels]
    return (img1, ann1), (img2, ann2)


def augment_two_images(image1, annotation1, image2, annotation2, min_points=5, n_jobs=2):
    """Augment two different images in parallel and return their results.

    Returns: (img1_aug, ann1_aug), (img2_aug, ann2_aug)
    """
    with ThreadPoolExecutor(max_workers=max(2, n_jobs)) as ex:
        fut1 = ex.submit(augment_image, image1, annotation1, min_points)
        fut2 = ex.submit(augment_image, image2, annotation2, min_points)
        return fut1.result(), fut2.result()
