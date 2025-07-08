"""
Unit test for loss functions

Author: Xumeng Zhang (xumzhang@uw.edu)

Models_HiCFoundation.forward_loss(self, imgs, imgs_mask, pred, mask):
imgs: [N, 3, H, W]
imgs_mask: [N, 1, H, W] indicate those 0 regions and mask them in target
pred: [N, L, D], sequence of embeddings
mask: [N, L], binary mask, 0 is keep, 1 is remove

"""

import pytest
import os
import sys
import inspect
import numpy as np

# add HiCFoundation dir into path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from model.models_hicfoundation import Models_HiCFoundation
from data_processing.pretrain_dataset import Pretrain_Dataset

import torchvision.transforms as transforms

import torch

image_size = 4
patch_size = 2
batch_size = 2
mask_ratio = 0.6
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]


# initialize model
@pytest.fixture
def model():
    return Models_HiCFoundation(
        img_size=(image_size, image_size), patch_size=patch_size
    )


# pseudo dataset class for preprocessing
@pytest.fixture
def dataset():
    # Create a temporary empty folder
    empty_folder_path = os.path.join(os.getcwd(), "temp_empty_folder")
    assert not os.path.exists(empty_folder_path) or not os.listdir(empty_folder_path), "temporary empty folder exists and also not empty, try change the folder path for testing"
    os.makedirs(empty_folder_path, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std),
    ])

    dataset = Pretrain_Dataset(data_list=[empty_folder_path], transform=transform_train)

    # Remove the folder after dataset creation (or after using it)
    # to be safe, only remove it if the folder is empty
    if os.path.exists(empty_folder_path) and not os.listdir(empty_folder_path):
        os.rmdir(empty_folder_path)

    return dataset


# Test when input is rgb
@pytest.mark.parametrize("in_chans", [3])
def test_forward_loss_rgb(model, dataset, in_chans):
    """
    test ssim loss and contrastive loss,
    here the ssim loss include the R channel loss
    as input of model.forward_loss, the imgs and pred should both be after normalization

    Args:
        model: initialized example model
        dataset: pseudo dataset
    """
    torch.manual_seed(42)

    B, C, H, W = batch_size, in_chans, image_size, image_size  # small test case
    L = (H // patch_size) * (W // patch_size)

    # simulate data preprocessing steps before inputing into loss func
    # random example
    imgs = np.random.randint(low=0, high=100, size=(H, W))
    imgs_converted = dataset.convert_rgb(imgs, max_value=np.max(imgs))  # (H, W, C)
    imgs_converted = dataset.transform(imgs_converted)  # (C, H, W)
    imgs_converted = torch.concat(
        [imgs_converted.unsqueeze(0)] * batch_size, dim=0
    )  # (B, C, H, W) repeat across batch

    # patched used later as true prediction
    imgs_patched = model.patchify(imgs_converted)  # (B, L, D)

    # sparsity
    imgs_mask = (torch.rand(B, 1, H, W) * 0.2).float()

    # random prediction
    pred = torch.rand(B, L, patch_size * patch_size * C)
    mask = (torch.rand(B, L) > mask_ratio).float()

    ssim_loss, contrastive_loss = model.forward_loss(
        imgs_converted, imgs_mask, pred, mask
    )

    # sanity check
    assert isinstance(ssim_loss, torch.Tensor), "SSIM loss should be a tensor"
    assert isinstance(contrastive_loss, torch.Tensor), (
        "Contrastive loss should be a tensor"
    )
    assert not torch.isnan(ssim_loss), "SSIM loss shouldn't be NaN"
    assert not torch.isnan(contrastive_loss), "Contrastive loss shouldn't be NaN"
    assert ssim_loss.item() >= 0, "SSIM loss should be non-negative"
    assert contrastive_loss.item() >= 0, "Contrastive loss should be non-negative"
    assert ssim_loss.item() != 0, "SSIM loss with random prediction should not be 0"
    assert contrastive_loss.item() != 0, (
        "Contrastive loss with random prediction should not be 0"
    )

    # loss with itself
    # the prediction here should be after convert rgb
    ssim_loss, contrastive_loss = model.forward_loss(
        imgs_converted, imgs_mask, imgs_patched, mask
    )
    assert ssim_loss.item() == 0, (
        f"ssim loss with itself should be 0, but it is {ssim_loss}"
    )
    # actually in Xiao's contrastive loss, this number will never be 0
    # it should be as close as to 1
    # assert contrastive_loss.item() == 0, (
    #     f"contrastive loss with itself should be 0, but it is {contrastive_loss}"
    # )


# Test when input is count matrix
@pytest.mark.parametrize("in_chans", [1])
def test_forward_loss_count(model, in_chans):
    """
    test ssim loss and contrastive loss for HiC count matrix. i.e. in_chans=1
    here the ssim loss include the R channel loss


    Args:
        model: initialized example model
        dataset: pseudo dataset
    """
    torch.manual_seed(42)

    B, C, H, W = batch_size, in_chans, image_size, image_size  # small test case
    L = (H // patch_size) * (W // patch_size)

    # random example
    img_np = np.random.randint(low=0, high=100, size=(H, W))
    img_torch = torch.tensor(img_np).unsqueeze(0).float()  # (1,H,W)
    imgs_torch = torch.concat(
        [img_torch.unsqueeze(0)] * batch_size, dim=0
    )  # (B, C, H, W)

    # tune the model into 1 in_chans mode
    model.in_chans = in_chans

    imgs_patched = model.patchify(imgs_torch)  # (B, L, D)

    imgs_mask = (torch.rand(B, 1, H, W) * 0.2).float()

    # random prediction
    pred = torch.rand(B, L, patch_size * patch_size * C)
    mask = (torch.rand(B, L) > mask_ratio).float()

    ssim_loss, contrastive_loss = model.forward_loss(imgs_torch, imgs_mask, pred, mask)

    # sanity check
    assert isinstance(ssim_loss, torch.Tensor), "SSIM loss should be a tensor"
    assert isinstance(contrastive_loss, torch.Tensor), (
        "Contrastive loss should be a tensor"
    )
    assert not torch.isnan(ssim_loss), "SSIM loss shouldn't be NaN"
    assert not torch.isnan(contrastive_loss), "Contrastive loss shouldn't be NaN"
    assert ssim_loss.item() >= 0, "SSIM loss should be non-negative"
    assert contrastive_loss.item() >= 0, "Contrastive loss should be non-negative"
    assert ssim_loss.item() != 0, "SSIM loss with random prediction should not be 0"
    assert contrastive_loss.item() != 0, (
        "Contrastive loss with random prediction should not be 0"
    )
