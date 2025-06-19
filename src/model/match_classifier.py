import torch
import torch.nn as nn

class MatchClassifier(nn.Module):
    """Binary classifier that predicts if a pair of items matches.

    The classifier aggregates similarity scores corresponding to the
    predicted correspondences and outputs a probability in ``[0, 1]``.
    """

    def __init__(self):
        super().__init__()
        # Single fully-connected layer for simplicity
        self.fc = nn.Linear(1, 1)

    def forward(self, sim_mat: torch.Tensor, corr_mat: torch.Tensor) -> torch.Tensor:
        """Compute match probability from similarity and correspondence matrices.

        Args:
            sim_mat: ``(b, n1, n2)`` raw similarity scores.
            corr_mat: ``(b, n1, n2)`` binary correspondence matrix.

        Returns:
            ``(b, 1)`` probability that each pair is a match.
        """
        # Aggregate original similarity scores using the correspondence mask
        matched_scores = (sim_mat * corr_mat).sum(dim=(1, 2))
        denom = corr_mat.sum(dim=(1, 2)).clamp(min=1)
        avg_score = matched_scores / denom
        out = torch.sigmoid(self.fc(avg_score.unsqueeze(1)))
        return out
