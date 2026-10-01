from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from sftl54_pytorch_dataset import SFTL54PairDataset


ROOT = Path(__file__).parent
EDGEFACE_REPO = ROOT / "otroshi-edgeface-ce86851"
CHECKPOINT_PATH = EDGEFACE_REPO / "checkpoints" / "edgeface_xs_gamma_06.pt"
BATCH_SIZE = 32


def load_edgeface() -> torch.nn.Module:
    """Load the frozen pretrained EdgeFace model from the local clone."""
    sys.path.insert(0, str(EDGEFACE_REPO))
    from backbones import get_model

    model = get_model("edgeface_xs_gamma_06")
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.eval()


def extract_test_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract normalized RGB/thermal embeddings and their subject IDs."""
    rgb_embeddings = []
    thermal_embeddings = []
    rgb_subject_ids = []
    thermal_subject_ids = []

    with torch.inference_mode():
        for batch in loader:
            rgb_embeddings.append(functional.normalize(model(batch["rgb"]), dim=1))
            thermal_embeddings.append(functional.normalize(model(batch["thermal"]), dim=1))
            rgb_subject_ids.append(batch["label"])
            thermal_subject_ids.append(batch["label"])

    return (
        torch.cat(rgb_embeddings),
        torch.cat(thermal_embeddings),
        torch.cat(rgb_subject_ids),
        torch.cat(thermal_subject_ids),
    )


def evaluate_identity_matching(
    rgb_embeddings: torch.Tensor,
    thermal_embeddings: torch.Tensor,
    rgb_subject_ids: torch.Tensor,
    thermal_subject_ids: torch.Tensor,
) -> dict[str, float]:
    """Evaluate RGB-to-thermal nearest-identity matching with cosine scores."""
    similarity_matrix = rgb_embeddings @ thermal_embeddings.T
    ranking = similarity_matrix.argsort(dim=1, descending=True)
    ranked_subject_ids = thermal_subject_ids[ranking]
    target_ids = rgb_subject_ids.unsqueeze(1)

    rank_1 = (ranked_subject_ids[:, :1] == target_ids).any(dim=1).float().mean()
    rank_5 = (ranked_subject_ids[:, :5] == target_ids).any(dim=1).float().mean()

    same_identity = rgb_subject_ids[:, None] == thermal_subject_ids[None, :]
    genuine_scores = similarity_matrix[same_identity]
    impostor_scores = similarity_matrix[~same_identity]

    return {
        "rank_1": rank_1.item(),
        "rank_5": rank_5.item(),
        "genuine_mean": genuine_scores.mean().item(),
        "impostor_mean": impostor_scores.mean().item(),
        "genuine_std": genuine_scores.std().item(),
        "impostor_std": impostor_scores.std().item(),
    }


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    test_dataset = SFTL54PairDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = load_edgeface()

    rgb_embeddings, thermal_embeddings, rgb_ids, thermal_ids = extract_test_embeddings(
        model,
        test_loader,
    )
    metrics = evaluate_identity_matching(
        rgb_embeddings,
        thermal_embeddings,
        rgb_ids,
        thermal_ids,
    )

    print("Test pairs:", len(test_dataset))
    print("RGB embeddings:", tuple(rgb_embeddings.shape))
    print("Thermal embeddings:", tuple(thermal_embeddings.shape))
    print("Embedding dimension:", rgb_embeddings.shape[1])
    print("\nPretrained EdgeFace baseline:")
    print(f"Rank-1 accuracy: {metrics['rank_1']:.4f}")
    print(f"Rank-5 accuracy: {metrics['rank_5']:.4f}")
    print(f"Genuine cosine mean: {metrics['genuine_mean']:.4f}")
    print(f"Genuine cosine std: {metrics['genuine_std']:.4f}")
    print(f"Impostor cosine mean: {metrics['impostor_mean']:.4f}")
    print(f"Impostor cosine std: {metrics['impostor_std']:.4f}")
