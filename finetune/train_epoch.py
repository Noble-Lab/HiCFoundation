
import math
import sys
import numpy as np
from typing import Iterable
import torch
import torch.nn.functional as F
import time

from ops.Logger import MetricLogger,SmoothedValue
import finetune.lr_sched as lr_sched
from finetune.loss import configure_loss

def to_device(data, device):
    if data is not None:
        new_data = data.to(device,non_blocking=True)
    else:
        new_data = None
    return new_data

def list_to_device(data_list, device):
    new_data_list = []
    for data in data_list:
        new_data_list.append(to_device(data, device))
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

def convert_gray_rgbimage(samples):
    """
    input: B,H,W
    """
    #add dimension in 1st dim
    if len(samples.shape)==3:
        samples = samples.unsqueeze(1)
    samples = torch.clip(samples, 0, 1)
    red_channel = torch.ones(samples.shape,device=samples.device)
    gb_channel = 1-samples
    new_samples=torch.cat([red_channel,gb_channel,gb_channel],dim=1)*255
    return new_samples

def train_epoch(model, data_loader_train, optimizer, 
                loss_scaler, epoch, device,
                log_writer=None, args=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    if log_writer is not None:
        print('Tensorboard log dir: {}'.format(log_writer.log_dir))
    print("number of iterations: ",len(data_loader_train))
    criterion = configure_loss(args)

    num_iter = len(data_loader_train)
    for data_iter_step, train_data in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)
        input_matrix, total_count, target_matrix, embed_target, target_vector = list_to_device(train_data,device=device)
        output_embedding, output_2d, output_1d = model(input_matrix, total_count)
        
        if embed_target is not None:
            embedding_loss = criterion(output_embedding, embed_target)
        else:
            embedding_loss = 0
        if target_matrix is not None:
            #flatten 2d matrix
            output_2d_flatten = torch.flatten(output_2d, start_dim=1,end_dim=-1)
            target_matrix_flatten = torch.flatten(target_matrix, start_dim=1,end_dim=-1)
            output_2d_loss = criterion(output_2d_flatten, target_matrix_flatten)
        else:
            output_2d_loss = 0
        if target_vector is not None:
            output_1d_loss = criterion(output_1d, target_vector)
        else:
            output_1d_loss = 0
        loss = embedding_loss + output_2d_loss + output_1d_loss #you can adjust the loss function based on your fine-tuning purpose
        #typically, I think you should only finetune for one of the purposes
        metric_logger.update(loss=to_value(loss))
        metric_logger.update(embedding_loss=to_value(embedding_loss))
        metric_logger.update(output_2d_loss=to_value(output_2d_loss))
        metric_logger.update(output_1d_loss=to_value(output_1d_loss))
        if not math.isfinite:
            print("Loss is {}, stopping training".format(to_value(loss)))
            #sys.exit(1)
            optimizer.zero_grad()
            continue
        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize() # Make sure all gradients are finished computing before moving on
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        

        if log_writer is not None and ((data_iter_step + 1) % accum_iter == 0 or data_iter_step==0):
            """ 
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)
            log_writer.add_scalars('Loss/loss', {'train_loss': to_value(loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/embedding_loss', {'train_loss': to_value(embedding_loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/output_2d_loss', {'train_loss': to_value(output_2d_loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/output_1d_loss', {'train_loss': to_value(output_1d_loss)}, epoch_1000x)
            log_writer.add_scalars('LR/lr', {'lr': lr}, epoch_1000x)
            if ((data_iter_step+1)//accum_iter)%50==0 or data_iter_step==0:
                #add visualization for your output and input
                new_samples = create_image(input_matrix)
                select_num = min(8,len(new_samples))
                sample_image = torch_to_nparray(new_samples.clone().detach()[:select_num])
                log_writer.add_images('Input_%s'%"train", sample_image, epoch_1000x)
                output_2d_image = convert_gray_rgbimage(output_2d.clone().detach()[:select_num])
                output_2d_image = torch_to_nparray(output_2d_image)
                log_writer.add_images('Output_2d_%s'%"train", output_2d_image, epoch_1000x)
                for name, param in model.named_parameters():
                    log_writer.add_histogram(name, param, epoch)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}