import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import datetime
import json
import numpy as np
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory

from ops.distribute_utils import init_distributed_mode,get_world_size,get_rank,is_main_process
from ops.Logger import print_important_info,print_warning_info
from data_processing.pretrain_dataset import Pretrain_Dataset
from data_processing.collate_fn import collate_fn
import model.models_hicfoundation as models_hicfoundation
from model.NativeScaler import NativeScalerWithGradNormCount as NativeScaler
from model.model_utils import load_model,save_checkpoint,save_model2path
from ops.io_utils import write_log
from pretrain.train_epoch import train_epoch
from pretrain.val_epoch import val_epoch

def parse_text(config_file, data_dir):
    train_list=[]
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            line = line.replace('\n', '')
            if len(line) == 0:
                continue
            current_path = os.path.join(data_dir, line)
            if not os.path.exists(current_path):
                print("The sub-directory {} does not exist in the data directory".format(current_path))
                print("Please check the sub-directory name in the {} file".format(config_file))
                continue
            train_list.append(current_path)
    return train_list

def configure_data_loader(args):
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_dir=os.path.abspath(args.data_path)
    train_config= os.path.abspath(args.train_config)
    train_list = parse_text(train_config, data_dir)
    val_config= os.path.abspath(args.valid_config)
    val_list = parse_text(val_config, data_dir)
    input_row_size = args.input_row_size
    input_col_size = args.input_col_size
    sparsity_filter = float(args.sparsity_ratio)
    patch_size = args.patch_size

    dataset_train = Pretrain_Dataset(train_list,transform=transform_train,
                                     sparsity_filter=sparsity_filter,patch_size=patch_size,
                                     window_height=input_row_size,window_width=input_col_size)
    dataset_val = Pretrain_Dataset(val_list,transform=transform_train,
                                   sparsity_filter=sparsity_filter, patch_size=patch_size,
                                        window_height=input_row_size,window_width=input_col_size)

    if  args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)
        global_rank = -1
    sample_batch_size = args.batch_size
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=sample_batch_size, sampler=sampler_train, 
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
        collate_fn=collate_fn
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=sample_batch_size, sampler=sampler_val, 
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
        collate_fn=collate_fn
    )
    return data_loader_train, data_loader_val

def config_writer(output_dir,tensorboard_log):
    tensorboard_dir = os.path.join(output_dir,'tensorboard')
    os.makedirs(tensorboard_dir,exist_ok=True)
    if tensorboard_log:
        from torch.utils.tensorboard import SummaryWriter
        log_writer = SummaryWriter(tensorboard_dir)
    else:
        log_writer = None
    return log_writer
def main_worker(gpu, ngpus_per_node,args):
    if ngpus_per_node>1:
        init_distributed_mode(gpu,ngpus_per_node,args)
    else:
        args.distributed=False
        print_warning_info("The distributed mode is disabled.\n For pre-training, one GPU may take very long to train!")
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    if  args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
    else:
        global_rank = -1
        num_tasks = 1
    output_dir = os.path.abspath(args.output)
    if global_rank==0:
        os.makedirs(output_dir,exist_ok=True)
        log_writer =config_writer(output_dir,args.tensorboard)
    elif args.distributed:
        log_writer = None
    else:
        os.makedirs(output_dir,exist_ok=True)
        log_writer = config_writer(output_dir,args.tensorboard)

    cudnn.benchmark = True
    device = torch.device(args.device)

    # Data loading code
    data_loader_train, data_loader_val = configure_data_loader(args)
    print("Data loader is configured!")

    # Configure the model
    patch_wise_size = (args.input_row_size//args.patch_size,args.input_col_size//args.patch_size)
    model = models_hicfoundation.__dict__[args.model](img_size=(args.input_row_size,args.input_col_size))
    
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        #not necessary for current setting, since all param with grad
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp =model
    #print out model information
    print("Model is configured!")

     #configure batch size, learning rate, weight decay
    eff_batch_size = args.batch_size * args.accum_iter*get_world_size()
    args.lr = args.blr * eff_batch_size / 256
    print("Learning rate: %.6f"%args.lr)
    print("Accumulative grad iteration: %d"%args.accum_iter)
    print("Effective batch size: %d"%eff_batch_size)
    print("Base learning rate: %.6f"%args.blr)

    #configure optimizer
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    #print(param_groups) #too long printing
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    #resume model if it is loading from checkpoint
    print("Optimizer is configured!")

    resume_path=os.path.abspath(args.resume)
    load_model(resume_path,args, model_without_ddp, optimizer, loss_scaler)

    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir,exist_ok=True)
    log_dir = os.path.join(output_dir, 'log')
    os.makedirs(log_dir,exist_ok=True)

    epochs = int(args.epochs)
    start_epoch = int(args.start_epoch)
    start_time = time.time()
    print("Start pre-training from epoch %d"%start_epoch," to epoch %d"%epochs)
    save_freq = args.save_freq
    best_loss = 1e9
    for epoch in range(start_epoch, epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
            )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        if is_main_process():
            write_log(log_dir,"train",log_stats)

        #validation run
        val_stats = val_epoch(
                model, data_loader_val,
                device, epoch,
                log_writer=log_writer,
                args=args
                )
        val_loss = val_stats['loss']
        log_stats_val = {**{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch,}
        if is_main_process():
            write_log(log_dir,"val",log_stats_val)
        if epoch%save_freq==0 or epoch==epochs-1:
            #output_dir, args,epoch, model_without_ddp, optimizer, loss_scaler
            save_checkpoint(model_dir, args,epoch, model_without_ddp, optimizer, loss_scaler)
        
        
        if val_loss < best_loss:
            best_loss = val_loss
            #model_path,args,epoch, model_without_ddp, optimizer, loss_scaler
            model_path = os.path.join(model_dir, 'model_best.pth.tar')
            save_model2path(model_path,args,epoch, model_without_ddp, optimizer, loss_scaler)
    total_time = time.time()-start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Fine-tuning of HiCFoundation is finished!")
    print("The model is saved in {}".format(model_dir))
    print("The log is saved in {}".format(log_dir))
