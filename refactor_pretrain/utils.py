
"""
from ops.sparse_ops import array_to_coo
"""
from scipy.sparse import triu,coo_matrix
import numpy as np
import torch
import numpy as np
import torch.nn as nn
import pickle
import os
import json 


def array_to_coo(array):
    """
    Convert a regular 2D NumPy array to a scipy.sparse.coo_matrix.

    Parameters:
    - array (numpy.ndarray): The input 2D array.

    Returns:
    - scipy.sparse.coo_matrix: The converted COO matrix.
    """
    # Find the non-zero elements in the array
    row, col = np.nonzero(array)

    # Get the values of the non-zero elements
    data = array[row, col]

    # Create the COO matrix
    coo_mat = coo_matrix((data, (row, col)), shape=array.shape)

    return coo_mat

"""
from ops.io_utils import load_pickle
"""

def load_pickle(path):
    with open(path,'rb') as file:
        data=pickle.load(file)
    return data

"""
from data_processing.finetune_dataset import to_tensor, list_to_tensor
"""


def to_tensor(x):
    """
    Convert the input to tensor
    Args:
        x: the input data
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif x is None:
        x = None
    #if already tensor, do nothing
    elif isinstance(x, torch.Tensor):
        pass
    #if float, convert to tensor
    elif isinstance(x, float):
        x = torch.tensor(x)
    elif isinstance(x, int):
        x = torch.tensor(x)
    return x

def list_to_tensor(x):
    """
    Convert the list to tensor
    Args:
        x: the input list
    """
    y=[]
    for i in x:
        y.append(to_tensor(i))
    return y




"""
from ops.train_utils import list_to_device, to_value, create_image, torch_to_nparray
"""

def list_to_device(data_list, device):

    def to_device(data, device):
        if data is not None:
            new_data = data.to(device,non_blocking=True)
        else:
            new_data = None
        return new_data

    new_data_list = []
    for data in data_list:
        data = to_device(data, device)
        if data is not None:
            data = data.float()
        new_data_list.append(data)
    return new_data_list

def to_value(data):
    if isinstance(data, torch.Tensor):
        return data.item()
    else:
        return data

def create_image(samples):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    imagenet_mean = torch.tensor(imagenet_mean,device=samples.device)
    imagenet_std = torch.tensor(imagenet_std,device=samples.device)
    new_samples = torch.einsum("bchw,c->bchw",samples,imagenet_std)
    new_samples = torch.clip((new_samples+ imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) * 255, 0, 255)
    return new_samples

def torch_to_nparray(data):
    #https://github.com/pytorch/pytorch/blob/main/torch/utils/tensorboard/summary.py
    #image take n,c,h,w,
    """
    'tensor' can either have values in [0, 1] (float32) or [0, 255] (uint8).
        The image() function will scale the image values to [0, 255] by applying
        a scale factor of either 1 (uint8) or 255 (float32). Out-of-range values
        will be clipped.

    """
    data = data.cpu().numpy()
   #data = data.transpose(0,2,3,1)
    data=np.array(data,dtype=np.uint8)
    return data


import torch
def collate_fn(batch):
    # Transpose the batch (list of lists) to group elements by position
    batch_transposed = list(zip(*batch))
    
    # Process each position across the batch
    processed_batch = []
    for tensors in batch_transposed:
        if all(t is None for t in tensors):  # If all are None, keep None
            processed_batch.append(None)
        else:  # Otherwise, stack non-None tensors and replace None with zero tensors
            #make sure no None element in the tensors
            any_none = any(t is None for t in tensors)
            assert not any_none, "None element in a list of tensors"
            stacked = [
                t for t in tensors
            ]
            processed_batch.append(torch.stack(stacked))
    
    return processed_batch


"""
from ops.io_utils import write_log
"""
def write_log(log_dir,status_flag,log_stats):
    cur_log_path = os.path.join(log_dir,status_flag+".log")
    with open(cur_log_path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_stats) + "\n")