"""
Unit test for patchify and unpatchify
With a focus on the order of each patchify channels

Author: Xumeng Zhang (xumzhang@uw.edu)

Models_HiCFoundation.patchify(self, imgs, in_chans=None)
imgs: (N, 3, H, W)
x: (N, L, H*W *self.in_chans)

Models_HiCFoundation.unpatchify(self, x, in_chans=None):
x: (N, L, patch_size**2 *self.in_chans)

"""

import pytest
import os
import sys
import inspect

# add HiCFoundation dir into path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from model.models_hicfoundation import Models_HiCFoundation


import torch

image_size = 4
patch_size = 2
batch_size = 2

# test for in_chans both 1 and 3 each time
in_chans = 3


# feed an easily interpretable example image
def make_example_image(in_chans):
    """
    make an example image that is easy for as to predict the patchify result
    This should be how it look like for count matrix or R channel of rgb image
    tensor([[1., 1., 2., 2.],
          [1., 1., 2., 2.],
          [3., 3., 4., 4.],
          [3., 3., 4., 4.]])
    G channel should be all 0
    B channel should be negative of R channel

    Returns:
        tensor: [B, C, H, W], image of size [batch, channel, height, weight]
    """
    n_patch_row = image_size // patch_size

    patches = []
    for i in range(n_patch_row * n_patch_row):
        if in_chans == 1:
            # make each patch a [patch_size, patch_size] square with all 1's or all 2's or ...
            patch = torch.full(
                (1, patch_size, patch_size), fill_value=(i + 1), dtype=torch.float32
            )
        elif in_chans == 3:
            # make each patch a [patch_size, patch_size] square
            # Channel R: i + 1 (e.g., 1, 2, 3, …)
            # Channel G: always 0
            # Channel B:  -(i + 1) (e.g., -1, -2, -3, …)
            chan0 = torch.full(
                (patch_size, patch_size), fill_value=(i + 1), dtype=torch.float32
            )
            chan1 = torch.zeros((patch_size, patch_size), dtype=torch.float32)
            chan2 = torch.full(
                (patch_size, patch_size), fill_value=-(i + 1), dtype=torch.float32
            )
            patch = torch.stack([chan0, chan1, chan2])
            assert patch.shape == (in_chans, patch_size, patch_size), f"{patch.shape}"
        patches.append(patch)

    patches = torch.stack(patches)  # [num_patches, inchan, patch_size, patch_size]
    patches = patches.reshape(
        n_patch_row, n_patch_row, in_chans, patch_size, patch_size
    )
    img = patches.permute(
        2, 0, 3, 1, 4
    )  # [inchan, n_patch_row, patch_size, n_patch_row, patch_size]
    img = img.reshape(in_chans, image_size, image_size)  # [C, H, W]
    # check the first patch is all 1
    assert torch.all(img[0, :patch_size, :patch_size] == 1)
    if in_chans == 3:
        assert torch.all(img[1, :, :] == 0)
        assert torch.all(img[2, :patch_size, :patch_size] == -1)
    # check the last patch is all n_patch_row**2
    assert torch.all(img[0, -patch_size:, -patch_size:] == n_patch_row**2)
    if in_chans == 3:
        assert torch.all(img[1, :, :] == 0)
        assert torch.all(img[2, -patch_size:, -patch_size:] == -(n_patch_row**2))

    return torch.cat([img.unsqueeze(0), img.unsqueeze(0)], dim=0)  # [B, C, H, W]


# initialize model class
@pytest.fixture
def model():
    return Models_HiCFoundation(
        img_size=(image_size, image_size), patch_size=patch_size
    )


@pytest.mark.parametrize("in_chans", [1, 3])
def test_patchify_output(model, in_chans):
    example_image = make_example_image(in_chans=in_chans)
    patches = model.patchify(example_image, in_chans=in_chans)
    # test output shape
    assert patches.shape == (
        batch_size,
        model.num_patches,
        model.patch_size**2 * in_chans,
    )
    assert patches.shape == (
        batch_size,
        (image_size / patch_size) ** 2,
        patch_size**2 * in_chans,
    )  # another way calculating

    # test whether output is as expected
    # the two batches are the same
    assert torch.all(patches[0] == patches[1])

    # check the three channels in the first patch
    for i in range(model.num_patches):
        if in_chans == 1:
            assert torch.all(
                patches[0, i, :] == torch.tensor([i + 1] * patch_size**2)
            ), f"patch No. {i} patchify not expected"
        else:
            assert torch.all(
                patches[0, i, :] == torch.tensor([i + 1, 0, -(i + 1)] * patch_size**2)
            ), f"patch No. {i} patchify not expected"


# test unpatchify logic is simple, after patchify and unpatchify if it returns the same thing then its great
@pytest.mark.parametrize("in_chans", [1, 3])
def test_patchify_unpatchify_equivalence(model, in_chans):
    example_image = make_example_image(in_chans=in_chans)
    patches = model.patchify(example_image, in_chans=in_chans)
    reconstructed = model.unpatchify(patches, in_chans=in_chans)

    assert reconstructed.shape == example_image.shape
    assert torch.allclose(example_image, reconstructed, atol=1e-6), (
        "Reconstructed patchified image doesn't match original"
    )
