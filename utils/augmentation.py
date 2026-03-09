from random import randint, uniform
import cv2
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

from src.model.ngm import CROPSIZE, RESCALE
import torch
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode

from scipy.ndimage import map_coordinates


def add_random(annos, num_points, width, height):
    noise = torch.stack([
        torch.rand(num_points) * width,
        torch.rand(num_points) * height,
    ], dim=1)

    noise_list = [["outlier", x, y] for x, y in noise.cpu().tolist()]

    return annos + noise_list


def bilinear_sample_displacement(disp_map, x, y):
    """Sample displacement at subpixel coordinates using bilinear interpolation."""
    return map_coordinates(disp_map, [[y], [x]], order=1, mode='reflect')[0]

# More realistic session-to-session transform candidates for fingerprints.
transforms = [
    "drop_out",
    "add_pore_noise",
    "affine",               # small rotation/translation/scale combined
    "elastic_transform",    # subtle skin deformation
    "gaussian_blur",        # slight defocus
    "motion_blur",          # finger slip
    "noise",                # sensor noise / dust specks
    "color_jitter",         # photometric jitter via torchvision ColorJitter
    "brightness_contrast_gamma",  # scanner exposure variability
    "clahe",                # local contrast changes due to dryness/wetness
    "jpeg_compress",        # compression artifacts
]

_RESIZE_TO_MODEL = tv_transforms.Resize(RESCALE, interpolation=InterpolationMode.BILINEAR)
_CENTER_CROP_TO_MODEL = tv_transforms.CenterCrop(CROPSIZE)
_RANDOM_AFFINE_APPLY = tv_transforms.RandomApply(
    [
        tv_transforms.RandomAffine(
            degrees=15,
            translate=(0.15, 0.15),
            scale=(0.95, 1.05),
            shear=None,
            interpolation=InterpolationMode.NEAREST,
        )
    ],
    p=0.9,
)
_COLOR_JITTER_APPLY = tv_transforms.RandomApply(
    [tv_transforms.ColorJitter(brightness=0.2, contrast=0.2)],
    p=0.8,
)


def _to_torch_image(image):
    """Convert HxW or HxWxC uint8 numpy image to CxHxW torch tensor."""
    if image.ndim == 2:
        return torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)

    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(image, -1, 0)))


def _from_torch_image(tensor, image_shape):
    """Convert CxHxW torch tensor back to uint8 numpy with original layout."""
    tensor = tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()
    if len(image_shape) == 2:
        return tensor[0]
    return np.moveaxis(tensor, 0, -1)


def _resize_torchvision(image):
    """Resize path using torchvision transforms (avoids PIL round-trips)."""
    tensor = _to_torch_image(image)
    tensor = _RESIZE_TO_MODEL(tensor)
    return _from_torch_image(tensor, image.shape)


def _center_crop_torchvision(image):
    """Center crop path using torchvision transforms."""
    tensor = _to_torch_image(image)
    tensor = _CENTER_CROP_TO_MODEL(tensor)
    return _from_torch_image(tensor, image.shape)


def _as_hw(size):
    """Normalize a torchvision transform size to (h, w)."""
    if isinstance(size, int):
        return int(size), int(size)
    if isinstance(size, (tuple, list)):
        if len(size) == 1:
            return int(size[0]), int(size[0])
        return int(size[0]), int(size[1])
    raise TypeError(f"Unsupported size type: {type(size)}")


def _filter_in_frame_annotations(annotation, h, w):
    """Keep only valid keypoints that lie within image bounds."""
    return [
        [id_, float(x), float(y)]
        for item in annotation
        if len(item) >= 3
        for id_, x, y in [item[:3]]
        if 0.0 <= float(x) < float(w) and 0.0 <= float(y) < float(h)
    ]


def _resize_annotations(annotation, src_h, src_w, dst_h, dst_w):
    """Resize keypoint coordinates from source to destination spatial size."""
    if src_h <= 0 or src_w <= 0:
        return []
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    return [
        [id_, float(x) * sx, float(y) * sy]
        for item in annotation
        if len(item) >= 3
        for id_, x, y in [item[:3]]
    ]


def _center_crop_annotations(annotation, src_h, src_w, crop_h, crop_w):
    """Apply a center-crop transform to keypoints and keep in-frame ones."""
    start_x = (float(src_w) - float(crop_w)) / 2.0
    start_y = (float(src_h) - float(crop_h)) / 2.0
    cropped = [
        [id_, float(x) - start_x, float(y) - start_y]
        for item in annotation
        if len(item) >= 3
        for id_, x, y in [item[:3]]
    ]
    return _filter_in_frame_annotations(cropped, crop_h, crop_w)


def _apply_affine_to_annotations(annotation, h, w, angle_deg, translate, scale):
    """Apply rotation+scale around image center plus translation to keypoints."""
    cx = (float(w) - 1.0) * 0.5
    cy = (float(h) - 1.0) * 0.5
    tx, ty = float(translate[0]), float(translate[1])
    theta = math.radians(float(angle_deg))
    cos_t = math.cos(theta) * float(scale)
    sin_t = math.sin(theta) * float(scale)

    transformed = []
    for item in annotation:
        if len(item) < 3:
            continue
        id_, x, y = item[:3]
        x = float(x)
        y = float(y)
        dx = x - cx
        dy = y - cy
        nx = cos_t * dx - sin_t * dy + cx + tx
        ny = sin_t * dx + cos_t * dy + cy + ty
        transformed.append([id_, nx, ny])
    return transformed


def _sample_affine_params(h, w):
    """Sample affine parameters matching current RandomAffine configuration."""
    angle = float(np.random.uniform(-15.0, 15.0))
    tx = int(round(np.random.uniform(-0.15 * float(w), 0.15 * float(w))))
    ty = int(round(np.random.uniform(-0.15 * float(h), 0.15 * float(h))))
    scale = float(np.random.uniform(0.95, 1.05))
    return angle, [tx, ty], scale


def _apply_elastic_to_image_and_annotations(image_tensor, annotation, sigma, alpha):
    """Apply the same elastic displacement to image and annotation coordinates."""
    h, w = image_tensor.shape[-2:]
    dx = np.random.rand(h, w).astype(np.float32) * 2.0 - 1.0
    dy = np.random.rand(h, w).astype(np.float32) * 2.0 - 1.0
    dx = cv2.GaussianBlur(dx, (0, 0), float(sigma)) * float(alpha)
    dy = cv2.GaussianBlur(dy, (0, 0), float(sigma)) * float(alpha)

    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)

    image_np = image_tensor.cpu().numpy()
    warped = np.empty_like(image_np)
    for c in range(image_np.shape[0]):
        warped[c] = cv2.remap(
            image_np[c],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    transformed = []
    for item in annotation:
        if len(item) < 3:
            continue
        id_, x, y = item[:3]
        x = float(x)
        y = float(y)
        nx = x + float(bilinear_sample_displacement(dx, x, y))
        ny = y + float(bilinear_sample_displacement(dy, x, y))
        transformed.append([id_, nx, ny])

    return torch.from_numpy(warped), transformed


def _build_label_id_maps(annotation):
    """Map arbitrary keypoint labels to contiguous positive integers."""
    label_to_int = {}
    int_to_label = {}
    next_id = 1

    for item in annotation:
        if len(item) < 3:
            continue
        label = item[0]
        if label in label_to_int:
            continue
        label_to_int[label] = next_id
        int_to_label[next_id] = label
        next_id += 1

    return label_to_int, int_to_label


def _annotation_to_channel(annotation, h, w, label_to_int=None):
    """Encode keypoint labels into a single HxW channel."""
    ann_channel = np.zeros((h, w), dtype=np.float32)
    if len(annotation) == 0:
        return ann_channel

    if label_to_int is None:
        label_to_int, _ = _build_label_id_maps(annotation)

    ids = np.asarray([label_to_int.get(a[0], 0) for a in annotation], dtype=np.float32)
    xs = np.rint(np.asarray([a[1] for a in annotation], dtype=np.float32)).astype(np.int64)
    ys = np.rint(np.asarray([a[2] for a in annotation], dtype=np.float32)).astype(np.int64)
    valid = (ids > 0) & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    ann_channel[ys[valid], xs[valid]] = ids[valid]
    return ann_channel


def _channel_to_annotations(ann_channel, int_to_label=None):
    """Decode keypoints from a channel, averaging duplicates per encoded id."""
    ys, xs = np.nonzero(ann_channel > 0)
    if ys.size == 0:
        return []

    ids = np.rint(ann_channel[ys, xs]).astype(np.int64)
    valid = ids > 0
    if not np.any(valid):
        return []

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    ids = ids[valid]
    unique_ids, inv = np.unique(ids, return_inverse=True)
    x_sum = np.bincount(inv, weights=xs)
    y_sum = np.bincount(inv, weights=ys)
    counts = np.bincount(inv)

    annotations = []
    for i, id_ in enumerate(unique_ids):
        encoded_id = int(id_)
        label = int_to_label.get(encoded_id, encoded_id) if int_to_label is not None else encoded_id
        annotations.append([label, float(x_sum[i] / counts[i]), float(y_sum[i] / counts[i])])
    return annotations


def _apply_random_center_crop(stacked):
    """Optionally crop around center with a random keep ratio using RandomApply."""
    h, w = stacked.shape[-2:]
    keep_ratio = uniform(0.85, 1.0)
    crop_h = max(2, min(h, int(h * keep_ratio)))
    crop_w = max(2, min(w, int(w * keep_ratio)))
    random_center_crop = tv_transforms.RandomApply(
        [tv_transforms.CenterCrop((crop_h, crop_w))],
        p=0.5,
    )
    return random_center_crop(stacked)


def _apply_joint_geometric_torchvision(
    image,
    annotation,
    use_affine=False,
    use_center_crop=False,
    use_vflip=False,
    use_hflip=False,
    elastic_sigma=None,
    elastic_alpha=None,
):
    """Apply geometric transforms with explicit keypoint coordinate updates."""
    h, w = image.shape[:2]
    image_tensor = _to_torch_image(image).float()
    transformed_annotations = [
        [item[0], float(item[1]), float(item[2])]
        for item in annotation
        if len(item) >= 3
    ]

    if use_affine and np.random.rand() < 0.9:
        angle, translate, scale = _sample_affine_params(h, w)
        image_tensor = TVF.affine(
            image_tensor,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        transformed_annotations = _apply_affine_to_annotations(
            transformed_annotations, h, w, angle, translate, scale
        )

    if use_center_crop and np.random.rand() < 0.5:
        keep_ratio = float(uniform(0.85, 1.0))
        crop_h = max(2, min(h, int(h * keep_ratio)))
        crop_w = max(2, min(w, int(w * keep_ratio)))
        image_tensor = TVF.center_crop(image_tensor, [crop_h, crop_w])
        transformed_annotations = _center_crop_annotations(
            transformed_annotations, h, w, crop_h, crop_w
        )
        h, w = crop_h, crop_w

    if use_vflip:
        image_tensor = TVF.vflip(image_tensor)
        transformed_annotations = [
            [id_, float(x), float(h - 1) - float(y)]
            for id_, x, y in transformed_annotations
        ]
    if use_hflip:
        image_tensor = TVF.hflip(image_tensor)
        transformed_annotations = [
            [id_, float(w - 1) - float(x), float(y)]
            for id_, x, y in transformed_annotations
        ]

    if elastic_sigma is not None and elastic_alpha is not None:
        image_tensor, transformed_annotations = _apply_elastic_to_image_and_annotations(
            image_tensor, transformed_annotations, elastic_sigma, elastic_alpha
        )

    transformed_annotations = _filter_in_frame_annotations(transformed_annotations, h, w)
    transformed_image = _from_torch_image(image_tensor, image.shape)
    return transformed_image, transformed_annotations

def _standardize_to_model(image, annotation):
    """Resize + center-crop image and keep only keypoints in the final frame."""
    h, w = image.shape[:2]
    resize_h, resize_w = _as_hw(_RESIZE_TO_MODEL.size)
    crop_h, crop_w = _as_hw(_CENTER_CROP_TO_MODEL.size)

    resized = _resize_torchvision(image)
    resized_annotations = _resize_annotations(annotation, h, w, resize_h, resize_w)

    standardized_image = _center_crop_torchvision(resized)
    standardized_annotations = _center_crop_annotations(
        resized_annotations, resize_h, resize_w, crop_h, crop_w
    )
    out_h, out_w = standardized_image.shape[:2]
    standardized_annotations = _filter_in_frame_annotations(
        standardized_annotations, out_h, out_w
    )
    return standardized_image, standardized_annotations


def augment_image(image, annotation, min_points=5):
    """Apply multiple random transformations to an image, each type at most once.

    Notes:
    - Designed for fingerprint images across sessions: mild pose (affine),
      subtle elastic deformation, photometric jitter, slight blur/noise.
    - Preserves keypoints under geometric transforms; photometric transforms
      do not alter keypoint coordinates.
    """
    original_image = image
    original_annotations = annotation

    # Create a random permutation of transformations
    transform_permutation = np.random.permutation(transforms)

    # Decide how many transformations to apply (1 to ~half)
    num_transforms = randint(1, max(2, math.ceil(len(transforms) / 2)))

    # Maximum number of attempts to get valid transformations
    max_attempts = 3
    for attempts in range(max_attempts):
        # Use fewer transforms on later attempts to preserve more keypoints.
        selected_transforms = transform_permutation[:max(1, num_transforms - attempts)]
        temp_image = original_image
        temp_annotations = original_annotations

        for transform_type in selected_transforms:
            temp_image, temp_annotations = apply_single_transform(
                temp_image, temp_annotations, transform_type, finalize=False
            )

        temp_image, temp_annotations = _standardize_to_model(temp_image, temp_annotations)

        # Check if we have enough valid keypoints
        if len(temp_annotations) >= min_points:
            return temp_image, temp_annotations

    # Fallback: still standardize to keep output geometry consistent.
    return _standardize_to_model(original_image, original_annotations)

def apply_single_transform(image, annotation, transformation_type, finalize=True):
    """Apply a single transformation to an image.

    Supported types:
    - "affine": random affine from torchvision (no shear)
    - "rotate", "translate": legacy cases mapped to affine
    - "center_crop": random-apply torchvision CenterCrop
    - "elastic_transform": smooth non-rigid displacement
    - "gaussian_blur": slight defocus
    - "motion_blur": linear motion blur
    - "noise": gaussian or salt-and-pepper
    - "color_jitter", "brightness_contrast_gamma": photometric jitter
    - "clahe": local contrast equalization
    - "jpeg_compress": add JPEG artifacts
    - "vertical_flip", "horizontal_flip": legacy (kept but not preferred)
    """
    if transformation_type in ("affine", "translate", "rotate"):
        transformed_image, transformed_annotations = _apply_joint_geometric_torchvision(
            image, annotation, use_affine=True, use_center_crop=False
        )
    # elif transformation_type == "center_crop":
    #     transformed_image, transformed_annotations = _apply_joint_geometric_torchvision(
    #         image, annotation, use_affine=False, use_center_crop=True
    #     )
    # elif transformation_type == "vertical_flip":
    #     transformed_image, transformed_annotations = _apply_joint_geometric_torchvision(
    #         image, annotation, use_vflip=True
    #     )
    # elif transformation_type == "horizontal_flip":
    #     transformed_image, transformed_annotations = _apply_joint_geometric_torchvision(
    #         image, annotation, use_hflip=True
    #     )
    elif transformation_type == "drop_out":
        mask = torch.rand(len(annotation)) > 0.2
        transformed_annotations = [a for a, m in zip(annotation, mask) if m]
        transformed_image = image
    elif transformation_type == "add_pore_noise":
        transformed_annotations = add_random(annotation, 25, image.shape[0], image.shape[1])
        transformed_image = image
    elif transformation_type == "elastic_transform":
        transformed_image, transformed_annotations = _apply_joint_geometric_torchvision(
            image,
            annotation,
            elastic_sigma=np.random.uniform(8, 20),
            elastic_alpha=np.random.uniform(0, 120),
        )

    elif transformation_type == "gaussian_blur":
        # Slight defocus
        k = np.random.choice([3, 5])
        sigma = uniform(0.1, 1.2)
        img_t = _to_torch_image(image).float() / 255.0
        img_t = TVF.gaussian_blur(img_t, kernel_size=[k, k], sigma=[sigma, sigma])
        transformed_image = _from_torch_image(img_t * 255.0, image.shape)
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

    elif transformation_type in ("color_jitter", "brightness_contrast_gamma"):
        # Photometric jitter (no keypoint movement)
        img_t = _to_torch_image(image).float() / 255.0
        img_t = _COLOR_JITTER_APPLY(img_t)
        transformed_image = _from_torch_image(img_t * 255.0, image.shape)
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
    else:
        transformed_image = image
        transformed_annotations = [[id_, x, y] for id_, x, y in annotation]
            
        
    if finalize:
        return _standardize_to_model(transformed_image, transformed_annotations)
    return transformed_image, transformed_annotations


def augment_image_pair(image, annotation, min_points=5, min_common=4, max_attempts=5, n_jobs=2):
    """Generate two augmented views of the same fingerprint more efficiently.

    - Runs augmentations in parallel threads (OpenCV releases the GIL).
    - Ensures a minimum number of shared keypoint labels across the two views
      (after cropping), falling back to a minimal standardization when needed.

    Returns: (img1, annos1_filtered), (img2, annos2_filtered)
    """
    assert n_jobs >= 2

 

    for _ in range(max_attempts):
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(augment_image, image, annotation, min_points) for _ in range(2)]
            (img1, annos1), (img2, annos2) = futs[0].result(), futs[1].result()
        labels1 = {a[0] for a in annos1}
        labels2 = {a[0] for a in annos2}
        common = labels1.intersection(labels2)
        if len(common) >= min_common:
            return (img1, annos1), (img2, annos2)

    # Fallback: standardized pair with identical geometry
    img1, ann1 = _standardize_to_model(image, annotation)
    img2, ann2 = _standardize_to_model(image, annotation)
    return (img1, ann1), (img2, ann2)


def augment_two_images(image1, annotation1, image2, annotation2, min_points=5, n_jobs=2):
    """Augment two different images in parallel and return their results.

    Returns: (img1_aug, ann1_aug), (img2_aug, ann2_aug)
    """
    with ThreadPoolExecutor(max_workers=max(2, n_jobs)) as ex:
        fut1 = ex.submit(augment_image, image1, annotation1, min_points)
        fut2 = ex.submit(augment_image, image2, annotation2, min_points)
        return fut1.result(), fut2.result()
