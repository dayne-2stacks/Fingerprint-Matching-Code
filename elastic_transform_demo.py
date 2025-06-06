import cv2
import numpy as np
import os
from utils.augmentation import bilinear_sample_displacement

def apply_elastic_transform(image, annotation, sigma=None, alpha=None):
    """Apply elastic transform to an image and its annotations."""
    h, w = image.shape[:2]
    
    # Use default values if not provided
    if sigma is None:
        sigma = np.random.uniform(3, 6)
    if alpha is None:
        alpha = np.random.uniform(8, 15)

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
                
    return transformed_image, transformed_annotations

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
    
    # Apply elastic transform
    transformed_image, transformed_annotations = apply_elastic_transform(image, annotations, sigma=20, alpha=300)
    print(f"Transformed annotations: {transformed_annotations}")
    
    # Draw circles on original and transformed images
    original_image_annotated = draw_annotations(image, annotations)
    transformed_image_annotated = draw_annotations(transformed_image, transformed_annotations)
    
    # Display original and transformed images
    # Save the annotated images instead of displaying them
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    original_path = os.path.join(output_dir, "original_image_annotated.png")
    transformed_path = os.path.join(output_dir, "transformed_image_annotated.png")
    cv2.imwrite(original_path, cv2.cvtColor(original_image_annotated, cv2.COLOR_RGB2BGR))
    cv2.imwrite(transformed_path, cv2.cvtColor(transformed_image_annotated, cv2.COLOR_RGB2BGR))
    print(f"Saved original annotated image to {original_path}")
    print(f"Saved transformed annotated image to {transformed_path}")

if __name__ == "__main__":
    main()
