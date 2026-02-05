import os
import random
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from vipe.priors.embedding import DinoBackboneFamily, EmbeddingsPipeline
from vipe.priors.embedding.dinov2 import DinoV2Variant
from vipe.priors.embedding.dinov3 import DinoV3Variant
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Dict


def sample_replica_images(
    data_dir,
    samples_per_scene=100,
    seed=42
):
    random.seed(seed)

    sampled_images = []

    # list all scene directories
    scenes = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    for scene in scenes:
        results_dir = os.path.join(data_dir, scene, "results")
        if not os.path.isdir(results_dir):
            continue

        # collect all frame images
        images = sorted(glob(os.path.join(results_dir, "frame*.jpg")))

        if len(images) == 0:
            continue

        # sample images
        if len(images) > samples_per_scene:
            images = random.sample(images, samples_per_scene)

        sampled_images.extend(images)

        print(f"{scene}: selected {len(images)} images")

    print(f"\nTotal sampled images: {len(sampled_images)}")
    return sampled_images


def load_and_normalize_images(image_paths,imagenet_transform):
    images = []

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = imagenet_transform(img)
        images.append(img)
    images = torch.stack(images, dim=0)
    return images


import torch
from typing import Dict

def compute_pca_from_features(
    feats: torch.Tensor,
    target_dim: int = 64
) -> Dict[str, torch.Tensor]:
    """
    Compute PCA from feature tensor of shape (B, H, W, D)

    Returns a dict with:
      - mean: (D,)
      - components: (target_dim, D)
      - metadata: dict
    """

    assert feats.dim() == 4, "Expected input shape (B, H, W, D)"

    B, H, W, D = feats.shape
    device = feats.device
    dtype = feats.dtype

    # --------------------------------------------------
    # 1. Flatten to (N, D)
    # --------------------------------------------------
    X = feats.reshape(-1, D)  # (B*H*W, D)
    num_samples = X.shape[0]

    # --------------------------------------------------
    # 2. Compute mean
    # --------------------------------------------------
    mean = X.mean(dim=0)  # (D,)

    # --------------------------------------------------
    # 3. Center data
    # --------------------------------------------------
    X_centered = X - mean

    # --------------------------------------------------
    # 4. PCA via SVD
    # --------------------------------------------------
    # X_centered = U S V^T
    # principal components = rows of V^T
    _, _, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:target_dim]  # (target_dim, D)

    # --------------------------------------------------
    # 5. Package result
    # --------------------------------------------------
    pca_dict = {
        "mean": mean.detach(),
        "components": components.detach(),
        "metadata": {
            "samples": int(num_samples),
            "target_dim": int(target_dim),
        }
    }

    return pca_dict


data_dir = "/data/Replica"

image_list = sample_replica_images(
    data_dir=data_dir,
    samples_per_scene=100
)

imagenet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

normalized_images = load_and_normalize_images(image_list,imagenet_transform)

model_family: DinoBackboneFamily = DinoBackboneFamily.DINOV2
model_variant: str | DinoV3Variant | DinoV2Variant = DinoV2Variant.VITB_REG
weights_dir = "/home/user/km-vipe/weights/dinov2"
embedder = EmbeddingsPipeline(
    model_family=model_family,
    model_variant=model_variant,
    weights_dir=weights_dir,
    pca_max_samples=2000000,
    device= 'cpu'
)

features = []
for image in normalized_images:
    features.append(embedder.process_image(image)[0])

features = torch.stack(features, dim=0)
pca_dict = compute_pca_from_features(features,64)
torch.save(pca_dict, "/home/user/km-vipe/pca_basis.pt")
