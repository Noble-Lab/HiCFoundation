# My code has references to the following repositories:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# AdPE: https://github.com/maple-research-lab/AdPE
# --------------------------------------------------------
import os
from argparser import  argparser_pretrain
import torch
import torch.multiprocessing as mp
import timm
assert timm.__version__ == "0.3.2" # version check


def main(args):
    # collect multi-gpu info
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("local ip: ",local_ip)

    # number of process per node
    ngpus_per_node = torch.cuda.device_count()
    # total number of process to run across all gpus
    # user gives number of nodes to use through args
    # here we multiply process per nodes
    args.world_size = args.world_size*ngpus_per_node
    from main_worker import main_worker
    if ngpus_per_node==1:
        main_worker(args.gpu,ngpus_per_node,args)#if you only have one gpu
    else:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,  args))




if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096*2, rlimit[1]))
    limit_in_b = 900 * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_DATA, (limit_in_b, limit_in_b))
    use_cuda = torch.cuda.is_available()
    print("starting check cuda status",use_cuda)
    #assert cuda is available
    assert use_cuda == True, "CUDA is not available, pre-training requires CUDA support to run!"
    parser = argparser_pretrain()
    args = parser.parse_args()
    #If you have many GPU on your server, but you only want to use few of them
    # run command line to configure the environment:
    # export CUDA_VISIBLE_DEVICES="0,1,2,3"
    # Here you can specify the GPU you want to use
    #check the specied input size, must be a multiple of args.patch_size
    if args.input_row_size%args.patch_size!=0 or args.input_col_size%args.patch_size!=0:
        print("args configuration error: input_row_size and input_col_size must be a multiple of patch_size")
        exit(1)
    main(args)
