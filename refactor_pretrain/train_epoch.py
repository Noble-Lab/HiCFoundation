import sys # Unused import
import time # Unused import
import math
import torch

import numpy as np
import torch.nn.functional as F

from typing import Iterable
from Logger import MetricLogger,SmoothedValue
from model_funcs import adjust_learning_rate
from utils import list_to_device, to_value, create_image, torch_to_nparray


def train_epoch(model,data_loader,optimizer,
                device, epoch, loss_scaler,
                log_writer=None,args=None):
    
    """
    Runs one full training epoch
    Args:
        model (torch.nn.Module): HiCFoundation model object
        data_loader (Iterable): Training Dataloader object
        optimizer (torch.optim.Optimizer): Optimizer instance for training.
        device (str): Training Device
        epoch (int): Current epoch
        loss_scaler (callable): NativeScalerWithGradNormCount object
        log_writer (optional): Tensorboard object
        args (Namespace or dict):
            Training configuration with at least:
                print_freq (int)   - logging frequency in steps
                accum_iter (int)   - number of steps to accumulate gradients
                mask_ratio (float) - masking ratio for model input
                loss_alpha (float) - weight for (ssim_loss + count_loss)
    Returns:
        dict[str, float]: Global metric averages upto this epoch
    """

    model.train() # Putting model in the training mode

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}')) # Track the learning rate in the smoothedvalue logger
    header = 'Epoch: [{}]'.format(epoch) 
    print_freq = args.print_freq # Logging frequency from config ??

    accum_iter = args.accum_iter 

    optimizer.zero_grad() # Reset the old gradients 

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir)) # I am not sure why we need to print it every epoch to know where we logging?
    
    print("number of iterations: ",len(data_loader))
    
    num_iter = len(data_loader) # Unused variable

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)): # This allows to only log every "print_freq" steps

        if data_iter_step % accum_iter == 0: # Update learning rate after "accum_iter" steps
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # Move batch to device and unpack fields
        input_matrix, mask_matrix, hic_count, return_diag, matrix_count = list_to_device(data,device=device) 

        with torch.cuda.amp.autocast(): #to enable mixed precision training
            ssim_loss, contrastive_loss, count_pred, pred_image, mask = model( # Forward pass
                input_matrix, mask_matrix, 
                total_count=hic_count,
                diag=return_diag, mask_ratio=args.mask_ratio, 
            )
            
            # Count prediction
            matrix_count = torch.log10(matrix_count+1)
            count_pred = count_pred.flatten()
            count_loss = torch.nn.functional.mse_loss(count_pred, matrix_count)
            loss = args.loss_alpha*(ssim_loss+count_loss) + contrastive_loss # Why alpha for ssim+count_loss and no control for contrastive?

        # Log loss values
        metric_logger.update(loss=to_value(loss))
        metric_logger.update(ssim_loss=to_value(ssim_loss))
        metric_logger.update(count_loss=to_value(count_loss))
        metric_logger.update(contrastive_loss=to_value(contrastive_loss))

        # Do not backprop for NaN or infinite loss (Maybe introduce loss clipping instead)?
        if not math.isfinite(to_value(loss)):
            print("Loss is {}, stopping training".format(to_value(loss)))
            #sys.exit(1)
            optimizer.zero_grad()
            continue
        
        # loss = loss / accum_iter
        loss_scaler(
            loss, optimizer, 
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0
        ) # NativeScalerWithGradNormCount object

        # After each accumulation cycle, clear grads
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # Synchronization step (across multiple workers -- DDP)
        torch.cuda.synchronize() # Make sure all gradients are finished computing before moving on
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)


        # Tensorboard logging 
        if log_writer is not None and ((data_iter_step + 1) % accum_iter == 0 or data_iter_step==0):
            """ 
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalars('Loss/loss', {'train_loss': to_value(loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/ssim_loss', {'train_loss': to_value(ssim_loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/count_loss', {'train_loss': to_value(count_loss)}, epoch_1000x)
            log_writer.add_scalars('Loss/contrastive_loss', {'train_loss': to_value(contrastive_loss)}, epoch_1000x)
            log_writer.add_scalars('LR/lr', {'lr': lr}, epoch_1000x)
            #add visualization
            if ((data_iter_step+1)//accum_iter)%50==0 or data_iter_step==0:
                new_samples = create_image(input_matrix)
                mask_image = new_samples*(1-mask)
                pred_image = create_image(pred_image) 

                select_num = min(8,len(new_samples))
                new_samples = torch_to_nparray(new_samples.clone().detach()[:select_num])
                mask_image = torch_to_nparray(mask_image.clone().detach()[:select_num])
                pred_image = torch_to_nparray(pred_image.clone().detach()[:select_num])
                log_writer.add_images('Target_%s'%"train", new_samples, epoch_1000x)
                log_writer.add_images('Input_%s'%"train", mask_image, epoch_1000x)
                log_writer.add_images('Pred_%s'%"train", pred_image, epoch_1000x)

    # Sync metrics across processes (DDP-safe)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # Return global averages across all logged metrics
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


                



