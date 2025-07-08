"""
Unit test for model forward

Author: Xumeng Zhang (xumzhang@uw.edu)

Test if the forward functions in HiCFoundation model work
Models_HiCFoundation.forward_encoder()
Models_HiCFoundation.forward_decoder()
Models_HiCFoundation.forward()

"""

import pytest
import os
import sys
import inspect
import numpy as np
import torch

# add HiCFoundation dir into path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from model.models_hicfoundation import Models_HiCFoundation


batch_size = 3  # batch size
in_chans = 3  # channels (1 or 3)
image_size = 4  # image size
patch_size = 2


# initialize model
@pytest.fixture
def model():
    return Models_HiCFoundation(
        img_size=(image_size, image_size), patch_size=patch_size
    )


#
@pytest.fixture
def example_input():
    imgs = torch.randn(batch_size, in_chans, image_size, image_size)
    # sample 1 and 3: no sparsity
    imgs_mask = torch.ones(batch_size, 1, image_size, image_size)
    # sample 2: all 0 matrix
    imgs_mask[1] = 0
    total_count = torch.tensor([1000000, 1000000, float("nan")])
    diag = torch.tensor([0, 0, 0]).float()
    return imgs, imgs_mask, total_count, diag


def test_forward_encoder(model, example_input):
    imgs, imgs_mask, total_count, diag = example_input

    latent, mask, ids_restore = model.forward_encoder(
        imgs, total_count=total_count, diag=diag, mask_ratio=0.6
    )
    assert latent.shape[0] == imgs.shape[0]
    assert mask.shape[0] == imgs.shape[0]
    assert ids_restore.shape[0] == imgs.shape[0]
    assert torch.isnan(latent[2]).all(), (
        "input nan total count should result in nan encoder output"
    )

    # test case if total_count=None
    # here total_count=None means not inputing any value for total_counts
    # different than having nan in total_count tensor
    # nan in total_count tensor will lead to nan loss
    latent, mask, ids_restore = model.forward_encoder(
        imgs, diag=diag, mask_ratio=0.6
    )
    assert latent.shape[0] == imgs.shape[0]
    assert mask.shape[0] == imgs.shape[0]
    assert ids_restore.shape[0] == imgs.shape[0]
    assert not torch.isnan(latent).any(), "encoder output contains NaNs"


def test_forward_decoder(model, example_input):
    imgs, imgs_mask, total_count, diag = example_input

    latent, mask, ids_restore = model.forward_encoder(
        imgs, total_count=total_count, diag=diag, mask_ratio=0.6
    )
    count_pred, patch_pred = model.forward_decoder(latent, ids_restore)
    assert count_pred.shape[0] == imgs.shape[0]
    assert patch_pred.shape[0] == imgs.shape[0]
    assert patch_pred.shape[1] == (image_size // patch_size) ** 2
    assert patch_pred.shape[2] == patch_size * patch_size * in_chans
    assert torch.isnan(count_pred[2]), (
        "input nan total count should result in nan count pred"
    )


def test_forward(model, example_input):
    imgs, imgs_mask, total_count, diag = example_input

    ssim_loss, contrastive_loss, count_pred, pred_img, mask = model.forward(
        imgs,
        imgs_mask,
        total_count=total_count,
        diag=diag,
        mask_ratio=0.4,
    )
    assert torch.isnan(ssim_loss), "input nan total count should result in nan ssimloss"
    assert torch.isnan(contrastive_loss), (
        "input nan total count should result in nan contrastive loss"
    )
    assert count_pred.shape[0] == imgs.shape[0]
    assert pred_img.shape == imgs.shape

    # remove the last sample whose total_count is nan
    imgs = imgs[:-1]
    imgs_mask = imgs_mask[:-1]
    total_count = total_count[:-1]
    diag = diag[:-1]
    ssim_loss, contrastive_loss, count_pred, pred_img, mask = model.forward(
        imgs,
        imgs_mask,
        total_count=total_count,
        diag=diag,
        mask_ratio=0.4
    )
    assert ssim_loss.item() >= 0
    assert contrastive_loss.item() >= 0
    assert count_pred.shape[0] == imgs.shape[0]
    assert pred_img.shape == imgs.shape
    assert mask.shape == pred_img.shape
    for i in range(batch_size - 1):
        assert torch.equal(mask[i, 0], mask[i, 1]) and torch.equal(
            mask[i, 1], mask[i, 2]
        ), f"mask differ amont channels for sample {i}"

