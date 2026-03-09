import numpy as np
import cv2

def build_matches(ds_mat, perm_mat):
    """Create OpenCV DMatch objects from similarity and permutation matrices.

    Args:
        ds_mat (np.ndarray or torch.Tensor): Similarity scores matrix of shape
            (N, M).
        perm_mat (np.ndarray or torch.Tensor): Binary permutation matrix of
            shape (N, M) indicating valid matches.
    Returns:
        list[cv2.DMatch]: List of matches suitable for visualization with
            cv2.drawMatches.
    """
    if hasattr(ds_mat, 'detach'):
        ds_mat = ds_mat.detach().cpu().numpy()
    if hasattr(perm_mat, 'detach'):
        perm_mat = perm_mat.detach().cpu().numpy()

    matches = []
    for i in range(ds_mat.shape[0]):
        valid_indices = np.where(perm_mat[i] == 1)[0]
        if valid_indices.size == 0:
            continue
        best_index = valid_indices[np.argmax(ds_mat[i, valid_indices])]
        distance_value = np.squeeze(ds_mat[i, best_index])
        if hasattr(distance_value, 'size') and distance_value.size != 1:
            distance_value = distance_value.flatten()[0]
        distance = float(distance_value)
        matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_index,
                                  _imgIdx=0, _distance=distance))
    return matches
