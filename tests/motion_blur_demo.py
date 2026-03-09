import cv2
import numpy as np
import os
from random import randint
import argparse

def apply_motion_blur(image, degree=None, angle=None):
    """
    Apply motion blur to an image.
    
    Args:
        image: The input image
        degree: The kernel size (odd number). If None, a random value will be used.
        angle: The angle of motion in degrees. If None, a random value will be used.
    
    Returns:
        The blurred image
    """
    h, w = image.shape[:2]
    
    # Use default values if not provided
    if degree is None:
        degree = 2 * randint(2, 4) + 1  # Generate an odd number between 5-9
    if angle is None:
        angle = randint(0, 180)
    
    # Create the motion blur kernel
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_blur_kernel = np.zeros((degree, degree))
    motion_blur_kernel[int((degree-1)/2), :] = 1
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / np.sum(motion_blur_kernel)
    
    # Apply the kernel to create motion blur effect
    blurred_image = cv2.filter2D(image, -1, motion_blur_kernel)
    
    return blurred_image, degree, angle



def draw_annotations(image, annotations, radius=5, color=(0, 255, 0), thickness=2):
    """Draw circles around annotations on an image."""
    annotated_image = image.copy()
    for id_, x, y in annotations:
        # Convert to integers for cv2.circle
        x, y = int(round(x)), int(round(y))
        # Draw circle around the annotation
        cv2.circle(annotated_image, (x, y), radius, color, thickness)
        # Add ID text
        cv2.putText(annotated_image, str(id_), (x+radius, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated_image

def load_sample_image(path=None):
    """Load a sample image and annotations."""
    if path and os.path.exists(path):
        # Load from specified path
        image = cv2.imread(path)
    else:
        # Create a placeholder fingerprint-like image
        image = np.ones((240, 320), dtype=np.uint8) * 200  # Light gray background
        # Add some random lines to simulate fingerprint ridges
        for _ in range(100):
            pt1 = (np.random.randint(0, 320), np.random.randint(0, 240))
            pt2 = (pt1[0] + np.random.randint(-30, 30), pt1[1] + np.random.randint(-30, 30))
            cv2.line(image, pt1, pt2, (100,), 1)
        
        # Add noise
        noise = np.random.normal(0, 10, image.shape).astype(np.int8)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
    # Generate sample annotations (minutiae points)
    h, w = image.shape[:2]
    # Try to load annotations from a TSV file with the same stem as the image
    annotations = []
    if path:
        stem, _ = os.path.splitext(path)
        print(f"Looking for annotations in {stem}.tsv")
        tsv_path = stem + ".tsv"
        if os.path.exists(tsv_path):
            with open(tsv_path, "r") as f:
                counter = 0
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        try:
                            _id = counter
                            x = float(parts[0])
                            y = float(parts[1])
                            if counter % 20 == 0:  # Sample every 20th point
                                annotations.append([ _id, x, y])
                            counter += 1
                        except ValueError:
                            continue
    if not annotations:
        # Generate random annotations if TSV not found
        for i in range(10):
            x = np.random.randint(20, w-20)
            y = np.random.randint(20, h-20)
            annotations.append([i, x, y])

    # Convert to RGB for better visualization if grayscale
    # if len(image.shape) == 2:
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # elif image.shape[2] == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image, annotations

def main():
    # Load image and annotations
    image_path = "/gold/home/dayneguy/Fingerprint/dataset/Synthetic/R1/1_left_loop.jpg"  # Modify this if you have a specific image
    image, annotations = load_sample_image(image_path)
    print(f"Original image shape: {image.shape}")
    print(f"Original annotations: {annotations}")
    
    # Apply motion blur with hardcoded parameters
    degree = 3  # Odd number for kernel size
    angle = 67  # Angle in degrees
    blurred_image, used_degree, used_angle = apply_motion_blur(image, degree, angle)
    print(f"Applied motion blur with degree={used_degree}, angle={used_angle}")
    
    # Draw circles on original and transformed images
    original_image_annotated = draw_annotations(image, annotations)
    blurred_image_annotated = draw_annotations(blurred_image, annotations)
    
    # Save the annotated images
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    original_path = os.path.join(output_dir, "original_image_annotated.png")
    blurred_path = os.path.join(output_dir, "motion_blur_annotated.png")
    cv2.imwrite(original_path, original_image_annotated)
    cv2.imwrite(blurred_path, blurred_image_annotated)
    print(f"Saved original annotated image to {original_path}")
    print(f"Saved blurred annotated image to {blurred_path}")

if __name__ == "__main__":
    main()
