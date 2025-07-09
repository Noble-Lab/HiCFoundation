"""
Unit test for masking

Author: Xumeng Zhang (xumzhang@uw.edu)

Test random masking for patches.
Focus on diagonal symmetric masking as well.

random_masking(self, x,  mask_ratio,diag=None):
x: [N, L, D], sequence (here L is without additional tokens)
mask_ratio: float, masking ratio
diag: [N,1] diagonal position to symmetrical masking, if None, then random masking
"""

import pytest
import os
import sys
import inspect

# add HiCFoundation dir into path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from model.models_hicfoundation import Models_HiCFoundation, apply_symmectric_noise


import torch

batch_size = 3
mask_ratio = 0.6
image_size = 128
patch_size = 16


# initialize model class
@pytest.fixture
def model():
    return Models_HiCFoundation(
        img_size=(image_size, image_size), patch_size=patch_size
    )


# feed an example noise as input
@pytest.fixture
def noise():
    noise = torch.zeros([image_size // patch_size, image_size // patch_size])
    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            noise[i, j] = i * 10 + j
    noise = torch.stack([noise] * batch_size)
    print(noise.shape)
    return noise


def test_apply_symmetric_noise_shape_and_symmetry(noise):
    noise_clone = noise.clone()
    diag = torch.tensor([0, -2, 2])

    output = apply_symmectric_noise(noise_clone.clone(), diag)

    assert output.shape == noise.shape

    # Case 1: diag = 0, i.e. full matrix should be symmetric
    symm_part = output[0]
    assert torch.allclose(symm_part, symm_part.T, atol=1e-6)

    # Case 2: diag = -2
    affected = output[1, 2:, :-2]
    transposed = affected.T
    assert torch.allclose(affected, transposed, atol=1e-6)

    # Case 3: diag = 2
    affected = output[2, :-2, 2:]
    transposed = affected.T
    assert torch.allclose(affected, transposed, atol=1e-6)


def test_random_masking_shape(model):
    """
    test Models_HiCFoundation.random_masking
    x is the (batch, length, dim) tensor

    """
    x = torch.randn(
        batch_size, model.num_patches, model.embed_dim
    )  # [batch, tokens, dim]
    print(x.shape)

    x_masked, mask, ids_restore = model.random_masking(x, mask_ratio)

    L = x.shape[1]
    len_keep = int(L * (1 - mask_ratio))

    assert x_masked.shape == (batch_size, len_keep, model.embed_dim)
    assert mask.shape == (batch_size, L)
    assert ids_restore.shape == (batch_size, L)
    assert (mask.sum(dim=1) == L - len_keep).all()  # confirm number of masked positions


def test_random_masking_deterministic_output_on_seed(model):
    torch.manual_seed(42)
    x = torch.randn(1, model.num_patches, model.embed_dim)
    x_masked1, mask1, _ = model.random_masking(x, mask_ratio)

    torch.manual_seed(42)
    x = torch.randn(1, model.num_patches, model.embed_dim)
    x_masked2, mask2, _ = model.random_masking(x, mask_ratio)

    # They should match since the seed and input match
    assert torch.allclose(x_masked1, x_masked2), (
        "The same random seed leads to different masking"
    )
    assert torch.allclose(mask1, mask2), (
        "The same random seed leads to different masking"
    )

    torch.manual_seed(42)
    x = torch.randn(1, model.num_patches, model.embed_dim)
    x_masked1, mask1, _ = model.random_masking(x, mask_ratio)
    x_masked2, mask2, _ = model.random_masking(x, mask_ratio)

    # They should match since the seed and input match
    assert not torch.allclose(x_masked1, x_masked2), (
        "Different sampling leads to the same masking"
    )
    assert not torch.allclose(mask1, mask2), (
        "Different sampling leads to the same masking"
    )
