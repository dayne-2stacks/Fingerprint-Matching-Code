from random import randint, uniform
import cv2
import numpy as np


def augment_image(image, annotation):
    transformation_type = np.random.choice(["translate", "rotate"])
    
    if transformation_type == "translate":
        dx, dy = randint(-20, 20), randint(-60, 60)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        transformed_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        transformed_annotations = [
            [id_, x + dx, y + dy] for id_, x, y in annotation
            if 0 <= x + dx < image.shape[1] and 0 <= y + dy < image.shape[0]
        ]
    elif transformation_type == "rotate":
        angle = uniform(-45, 45)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        transformed_image = cv2.warpAffine(image, matrix, (w, h))
        transformed_annotations = []
        for id_, x, y in annotation:
            coords = np.dot(matrix[:, :2], [x, y]) + matrix[:, 2]
            if 0 <= coords[0] < w and 0 <= coords[1] < h:
                transformed_annotations.append([id_, coords[0], coords[1]])

    return transformed_image, transformed_annotations
