from random import randint, uniform
import cv2
import numpy as np

from scipy.ndimage import map_coordinates

def bilinear_sample_displacement(disp_map, x, y):
    """Sample displacement at subpixel coordinates using bilinear interpolation."""
    return map_coordinates(disp_map, [[y], [x]], order=1, mode='reflect')[0]


transforms = ["translate", "rotate", "elastic_transform", "noise", "motion_blur"]
# transforms = ["translate", "rotate", "vertical_flip", "horizontal_flip", "elastic_transform", "noise", "motion_blur"]


def augment_image(image, annotation, min_points=5):
    """Apply multiple random transformations to an image, each type at most once."""
    # Store original image and annotations in case we need to fall back
    original_image = image.copy()
    original_annotations = annotation.copy()
    
    # Create a random permutation of transformations
    transform_permutation = np.random.permutation(transforms)

    # Decide how many transformations to apply (1 to 4)
    num_transforms = randint(1, len(transforms))
    
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
    resized = cv2.resize(original_image, (320, 320))
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
    """Apply a single transformation to an image."""
    h, w = image.shape[:2]
    
    if transformation_type == "translate":
        dx, dy = randint(-50, 50), randint(-50, 50)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        transformed_image = cv2.warpAffine(image, matrix, (w, h))
        transformed_annotations = [
            [id_, x + dx, y + dy] for id_, x, y in annotation
            if 0 <= x + dx < w and 0 <= y + dy < h
        ]
    elif transformation_type == "rotate":
        angle = uniform(-45, 45)
        
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        transformed_image = cv2.warpAffine(image, matrix, (w, h))
        transformed_annotations = []
        for id_, x, y in annotation:
            coords = np.dot(matrix[:, :2], [x, y]) + matrix[:, 2]
            if 0 <= coords[0] < w and 0 <= coords[1] < h:
                transformed_annotations.append([id_, coords[0], coords[1]])
    elif transformation_type == "vertical_flip":
        transformed_image = cv2.flip(image, 0)
        transformed_annotations = [[id_, x, h - y - 1] for id_, x, y in annotation if 0 <= x < w and 0 <= h - y < h]
    elif transformation_type == "horizontal_flip":
        transformed_image = cv2.flip(image, 1)
        transformed_annotations = [[id_, w - x - 1, y] for id_, x, y in annotation if 0 <= w - x < w and 0 <= y < h]
        
    elif transformation_type == "elastic_transform":
        sigma = np.random.uniform(20, 50)  # Use moderate sigma for realistic deformation
        alpha = np.random.uniform(0, 275)

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

    
    elif transformation_type == "motion_blur":
        degree = 9
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
            # Vary the noise intensity randomly
            sigma = np.random.uniform(0, 3)
            noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
            transformed_image = image.astype(np.float32) + noise
            # Clip values to valid range
            transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
            
        else:  # salt_and_pepper
            # Apply salt and pepper noise - simulates dust/dirt on scanner
            transformed_image = image.copy()
            # Noise density between 0.5% and 5%
            amount = np.random.uniform(0.005, 0.01)
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
            
        
    # Downsample to 320x320
    transformed_image = cv2.resize(transformed_image, (320, 320))
    scale_x, scale_y = 320 / w, 320 / h
    transformed_annotations = [[id_, x * scale_x, y * scale_y] for id_, x, y in transformed_annotations]

    # Center crop to 240x320
    crop_h, crop_w = 240, 320
    start_x = (320 - crop_w) // 2
    start_y = (320 - crop_h) // 2
    cropped_image = transformed_image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    cropped_annotations = [[id_, x - start_x, y - start_y] for id_, x, y in transformed_annotations if start_x <= x < start_x + crop_w and start_y <= y < start_y + crop_h]

    return cropped_image, cropped_annotations
