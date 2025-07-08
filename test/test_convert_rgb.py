"""
Unit test for convert RGB

Author: Xumeng Zhang (xumzhang@uw.edu)

This function takes only one np window from one npz file
does not handle batch

"""

import pytest
import os
import sys
import inspect

# add HiCFoundation dir into path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_processing.pretrain_dataset import Pretrain_Dataset


import numpy as np
import torchvision.transforms as transforms

batch_size = 2
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]


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


# example hic matrix
@pytest.fixture
def example_hic():
    # should be np matrix [window_size, window_size]
    hic = np.array([[0.5, 1.0], [0.0, 0.75]])
    return hic


def test_convert_rgb_shape_and_dtype(dataset, example_hic):
    out = dataset.convert_rgb(example_hic, np.max(example_hic))

    assert out.shape == (*example_hic.shape, 3)  # B,
    assert out.dtype == np.float32


def test_convert_rgb_values(dataset, example_hic):
    """
    check if the converted values are as expected
    """
    out = dataset.convert_rgb(example_hic, np.max(example_hic))
    red = out[:, :, 0]
    green = out[:, :, 1]
    blue = out[:, :, 2]

    # the theoritical value for both blue and green channel
    expected_gb = (np.max(example_hic) - example_hic) / np.max(example_hic)

    assert np.allclose(red, np.ones_like(red)), "red channel should be all 1"
    assert np.allclose(green, expected_gb), "Green channel is not as expected"
    assert np.allclose(blue, expected_gb), "blue channel is not as expected"


def test_log10_convert(dataset, example_hic):
    """
    Test is the function handles data after log10 well
    """
    log10_input = np.log10(example_hic + 1)
    log10_max_val = np.log10(np.max(example_hic) + 1)
    out = dataset.convert_rgb(log10_input, log10_max_val)
    red = out[:, :, 0]
    green = out[:, :, 1]
    blue = out[:, :, 2]

    # the theoritical value for both blue and green channel
    expected_gb = (log10_max_val - log10_input) / log10_max_val

    assert np.allclose(red, np.ones_like(red))
    assert np.allclose(green, expected_gb)
    assert np.allclose(blue, expected_gb)


def test_post_convert_transform(dataset, example_hic):
    """
    Test if the post-process transformation reproduce the same matrix
    """
    log10_input = np.log10(example_hic + 1)
    log10_max_val = np.log10(np.max(example_hic) + 1)

    out = dataset.convert_rgb(log10_input, log10_max_val)
    out = dataset.transform(out)  # [3, window_size, window_size]
    assert out.shape[0] == 3, (
        "After transformation, the first dimension should be 3, then the window_size by window_size"
    )
    red_inverse = out[0, :, :] * transform_std[0] + transform_mean[0]
    green_inverse = out[1, :, :] * transform_std[1] + transform_mean[1]
    blue_inverse = out[2, :, :] * transform_std[2] + transform_mean[2]

    expected_gb = (log10_max_val - log10_input) / log10_max_val

    assert np.allclose(red_inverse, np.ones_like(red_inverse), atol=1e-6)
    assert np.allclose(green_inverse, expected_gb, atol=1e-6)
    assert np.allclose(blue_inverse, expected_gb, atol=1e-6)
