"""
    Code here is borrowed from the https://github.com/facebookresearch/deit/blob/main/utils.py
    This file includes utility scripts for handling distributed training
"""

import os
import torch
import resource
import datetime
import builtins
import numpy as np
import torch.distributed as dist

def is_dist_avail_and_initialized():
    """
        This function checks a distributed process, whether its initialized properly
        It returns true, if all the requisites, rank, world_size parameters, seeds etc.. are setup 
        else it returns false
        Args: 
            None
        Returns:
          bool: True if distributed backend is usable.

    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    """
        This function fetches the rank of a distributed process in a distributed processes group. 
        Ranks range from [0 - (world_size -1)], rank 0 is assigned to the main (master) worker
        In a non-distributed setting the rank of the main process is also 0
        Args: 
            None
        Returns:
          int: Rank of the process

    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    """
        This function fetches the world_size which is the total number of workers in a distributed group. 
        In a non-distributed setting the world size is 1 and the main process's rank is 0
        Args: 
            None
        Returns:
          int: world_size (number of processes in a distributed group)

    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def all_reduce_mean(x):
    """
        This function fetches value of a particular variable 'x' from all the distributed processes and 
        returns a mean of mean of that value. Specifically, it could be used to fetch loss for a batch of input 
        values across the distributed processes and can return the mean of that loss for the subsequent backprop.
        This can be used to gather evaluation metrics from other processes too. 
        Args: 
            x (float): A scalar value to be averaged.
        Returns:
            float: Average of x across all the distributed processes
    """
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def is_main_process():
    """
        A binary check for testing whether the process we are operating on is the main process
        The check is simple, the rank of the current process should be 0 for it to considered main
        Args: 
            None
        Returns:
            bool: True if main process
    """

    return get_rank() == 0


def setup_for_distributed(is_master):
    """
        This is a function that overrides the python print function to be only 
        accessible from the main process. This convention is followed to keep the logs clean
        Args: 
            is_master (bool): Boolean input
        Returns:
            None
    """
    builtin_print = builtins.print
    def print(*args, **kwargs):
        if is_master:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)
    builtins.print = print

def init_distributed_mode(gpu, ngpus_per_node, args):
    """
        Initialize distributed training environment by setting a few os paramters and then setup
        distributed training across provided number of gpus across the provided number of nodes. 
        Args:
            gpu (int): GPU index for current process.
            ngpus_per_node (int): Number of GPUs per node.
            args (Namespace): Arguments object containing attributes:
                - rank (int): Base rank of the node.
                - world_size (int): Total number of processes.
                - dist_url (str): Initialization URL for distributed training.
                - seed (int): Seed for random number generators.
        Returns:
            None
    """

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    os.environ['LOCAL_RANK'] = str(args.gpu)
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    print("make sure the distributed mode is ",args.dist_url)



    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         timeout=datetime.timedelta(seconds=36000),
                                         world_size=args.world_size, rank=args.rank)

    setup_for_distributed(args.rank == 0)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
