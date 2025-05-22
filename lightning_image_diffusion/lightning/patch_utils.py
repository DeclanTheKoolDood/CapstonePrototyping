import torch
import torch.nn.functional as F


def extract_patches(img, patch_size=32, overlap=8):
    """
    Extract overlapping patches from an image tensor.
    Args:
        img: (B, C, H, W) tensor
        patch_size: size of each patch (int)
        overlap: overlap between patches (int)
    Returns:
        patches: (num_patches, C, patch_size, patch_size)
        positions: list of (y, x) positions for each patch
        out_shape: (H, W)
    """
    B, C, H, W = img.shape
    stride = patch_size - overlap
    patches = []
    positions = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[:, :, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    patches = torch.cat(patches, dim=0)
    return patches, positions, (H, W)


def reconstruct_from_patches(patches, positions, out_shape, patch_size=32, overlap=8):
    """
    Reconstruct image from overlapping patches by averaging overlaps.
    Args:
        patches: (num_patches, C, patch_size, patch_size)
        positions: list of (y, x) positions
        out_shape: (H, W)
    Returns:
        img: (1, C, H, W) tensor
    """
    device = patches.device
    C = patches.shape[1]
    H, W = out_shape
    img = torch.zeros((1, C, H, W), device=device)
    count = torch.zeros((1, 1, H, W), device=device)
    for i, (y, x) in enumerate(positions):
        img[:, :, y:y+patch_size, x:x+patch_size] += patches[i:i+1]
        count[:, :, y:y+patch_size, x:x+patch_size] += 1
    img = img / count.clamp(min=1)
    return img
