from random import randint, uniform
import cv2
import numpy as np


def augment_image(image, annotation):
    transformation_type = np.random.choice(["translate", "rotate"])
    h, w = image.shape[:2]
    
    if transformation_type == "translate":
        dx, dy = randint(-100, 100), randint(-100, 100)
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
