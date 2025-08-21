import torch
from distribute_utils import is_main_process
from pathlib import Path
import os
from math import inf


def load_model(resume_path, args, model_without_ddp, optimizer, loss_scaler):
    """
    Load the model from the checkpoint
    Args:
        resume_path: the path to the checkpoint
        model_without_ddp: the model
        optimizer: the optimizer
        loss_scaler: the loss scaler
    """
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        if resume_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                resume_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(resume_path, weights_only=False, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("model resume message:{}".format(msg))
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))


def save_on_master(*args, **kwargs):
    """
    Save model
    :param args: positional arguments, torch.save arguments
    :param kwargs: keyword arguments, torch.save arguments
    :return: None
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def save_checkpoint(output_dir, args, epoch, model_without_ddp, optimizer, loss_scaler):
    """
    Save model, optimizer, epoch, scaler and arguments as a checkpoint
    :param output_dir: str, output directory for saving
    :param args: positional arguments
    :param epoch: int, training epoch
    :param model_without_ddp: torch.nn.module, model with no distributed components
    :param optimizer: torch.optimizer, optimizer for model training
    :param loss_scaler: torch.cuda.amp.GradScaler, dynamically scales the loss to prevent underflow when using float16
    :return: None.
    """
    # Save directory
    output_dir = Path(output_dir)
    # Output epoch
    epoch_name = str(epoch)

    # checkpoint path
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
            'args': args,
        }

        save_on_master(to_save, checkpoint_path)


def save_model2path(model_path, args, epoch, model_without_ddp, optimizer, loss_scaler):
    """
    Save model to a certain path
    :param model_path: str, model path directory and file name
    :param args: positional arguments
    :param epoch: int, number of epochs
    :param model_without_ddp: torch.nn.module, model with no distributional parameters
    :param optimizer: torch.optimizer, optimizer for model training
    :param loss_scaler: torch.cuda.amp.GradScaler, dynamically scales the loss to prevent underflow when using float16
    :return: None
    """
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
        'args': args,
    }
    save_on_master(to_save, model_path)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute gradient norm
    :param parameters: torch.tensor, model parameters
    :param norm_type: float, which gradient norm 2.0 -> L2
    :return: total norm
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm
