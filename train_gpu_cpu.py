import os
os.environ['TMPDIR'] = '/tmp'
import argparse
import math
import random
import sys
import time
import csv
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import (
    DCAE_3

)
from torch.utils.tensorboard import SummaryWriter   
import os

from icecream import ic
torch.set_num_threads(8)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

torch.set_default_dtype(torch.float64)

def test_compute_psnr(a, b):
    b = b.to(a.device)
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def test_compute_msssim(a, b):
    b = b.to(a.device)
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def test_compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    bpp_dict = {}
    ans = 0
    for likelihoods in out_net['likelihoods']:
        fsize = out_net['likelihoods'][likelihoods].size()
        num = 1
        for s in fsize:
            num = num * s
        
        print(fsize)
        bpp_dict[likelihoods] = torch.log(out_net['likelihoods'][likelihoods]).sum() / (-math.log(2) * num_pixels)
        print(f"{likelihoods}:{bpp_dict[likelihoods]}")
        ans = ans + bpp_dict[likelihoods]
    print(f"ans:{ans}")
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type
        self.device = "cuda"

    def forward(self, output, target):
        target = target.to(self.device)

        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, train_sampler, type='mse', args=None, lr_scheduler=None
):
    model.train()
    device = "cuda" # next(model.parameters()).device # it gets the device of the first parameter

    if torch.cuda.device_count() > 1:
        train_sampler.set_epoch(epoch)

    pre_time = 0
    now_time = time.time()
    for i, d in enumerate(train_dataloader):

        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm) 
        optimizer.step()
        
        aux_loss = model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        ic(f"running: {i} epoch")

        if (i+1) % 100 == 0:
            pre_time = now_time
            now_time = time.time()
            ic(f"time : {now_time - pre_time}")
            ic(f"lr : {lr_scheduler.get_last_lr()[0]}")
            
            if type == 'mse':
               ic(
                    f"Train epoch {epoch}: ["
                    f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
            else:
                ic(
                    f"Train epoch {epoch}: ["
                    f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )


def test_epoch(epoch, test_dataloader, model, criterion, type='mse', args=None):
    model.eval()
    # device = next(model.parameters()).device
    device = "cuda"
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)
                loss.update(out_criterion["loss"])
                aux_loss.update(model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss())
                mse_loss.update(out_criterion["mse_loss"])
                bpp_loss.update(out_criterion["bpp_loss"])

        ic(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)
                aux_loss.update(model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

        ic(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    return loss.avg

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def save_checkpoint(state, is_best, epoch, save_path, filename):
    '''
    save checkpoint of every epoch and the best loss
    '''
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--drop", action="store_true", default=False,
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim", "l1"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--M", type=int, default=320,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    # torch.autograd.set_detect_anomaly(True)

    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        
    writer = SummaryWriter(save_path + "tensorboard/")

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train_30k", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="train_30k", transform=test_transforms)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    else: 
        device = "cuda"
        
    net = DCAE_3()
    net = net.to(device)

    print(f'I am on {device}')

    # Move everything except g_a to GPU (g_a is already on CPU from model definition)
    for name, module in net.named_children():
        if name == 'g_a':
            module.to("cpu")
    
    # Only check modules with parameters
    for name, param in net.named_parameters():
        print(f"Parameter: {name}, Device: {param.device}")

    if args.cuda and torch.cuda.device_count() > 1:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler if torch.cuda.device_count() > 1 else None,
        shuffle=True if torch.cuda.device_count() == 1 else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler if torch.cuda.device_count() > 1 else None,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)


    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type).to(device)
    last_epoch = 0

    dictory = {}
    if args.checkpoint:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
        print("loaded")

        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            train_sampler if torch.cuda.device_count() > 1 else None,
            type,
            args,
            lr_scheduler
        )


        loss = test_epoch(epoch, test_dataloader, net, criterion, type, args)
        writer.add_scalar('test_loss', loss, epoch)


        global_rank = dist.get_rank() if torch.cuda.device_count() > 1 else 0
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save and global_rank == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    epoch,
                    save_path,
                    save_path + str(epoch) + "_checkpoint.pth.tar",
                )



if __name__ == "__main__":
    main(sys.argv[1:])


# import os
# os.environ['TMPDIR'] = '/tmp'
# import argparse
# import math
# import random
# import sys
# import time
# import csv
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# from torch.utils.data import DataLoader
# from torch import distributed as dist
# from torch.utils.data.distributed import DistributedSampler
# from torchvision import transforms

# from compressai.datasets import ImageFolder
# from compressai.zoo import models
# from pytorch_msssim import ms_ssim

# from models import (
#     DCAE_3
# )
# from torch.utils.tensorboard import SummaryWriter   
# import os
# torch.set_num_threads(8)
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False

# torch.set_default_dtype(torch.float64)

# def test_compute_psnr(a, b):
#     b = b.to(a.device)
#     mse = torch.mean((a - b)**2).item()
#     return -10 * math.log10(mse)

# def test_compute_msssim(a, b):
#     b = b.to(a.device)
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

# def test_compute_bpp(out_net):
#     size = out_net['x_hat'].size()
#     num_pixels = size[0] * size[2] * size[3]
#     bpp_dict = {}
#     ans = 0
#     for likelihoods in out_net['likelihoods']:
#         fsize = out_net['likelihoods'][likelihoods].size()
#         num = 1
#         for s in fsize:
#             num = num * s
        
#         print(fsize)
#         bpp_dict[likelihoods] = torch.log(out_net['likelihoods'][likelihoods]).sum() / (-math.log(2) * num_pixels)
#         print(f"{likelihoods}:{bpp_dict[likelihoods]}")
#         ans = ans + bpp_dict[likelihoods]
#     print(f"ans:{ans}")
#     return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
#               for likelihoods in out_net['likelihoods'].values()).item()


# def compute_msssim(a, b):
#     return ms_ssim(a, b, data_range=1.)


# class RateDistortionLoss(nn.Module):
#     """Custom rate distortion loss with a Lagrangian parameter."""

#     def __init__(self, lmbda=1e-2, type='mse'):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.lmbda = lmbda
#         self.type = type
#         self.device = "cuda"

#     def forward(self, output, target):
#         target = target.to(self.device)

#         N, _, H, W = target.size()
#         out = {}
#         num_pixels = N * H * W

#         out["bpp_loss"] = sum(
#             (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
#             for likelihoods in output["likelihoods"].values()
#         )
#         if self.type == 'mse':
#             out["mse_loss"] = self.mse(output["x_hat"], target)
#             out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
#         else:
#             out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
#             out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

#         return out

# class AverageMeter:
#     """Compute running average."""

#     def __init__(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# class CustomDataParallel(nn.DataParallel):
#     """Custom DataParallel to access the module methods."""

#     def __getattr__(self, key):
#         try:
#             return super().__getattr__(key)
#         except AttributeError:
#             return getattr(self.module, key)

# def configure_optimizers(net, args):
#     """Separate parameters for the main optimizer and the auxiliary optimizer.
#     Return two optimizers"""

#     # Group parameters into CPU and GPU groups
#     gpu_parameters = set()
#     cpu_parameters = set()
#     aux_parameters = set()
    
#     for n, p in net.named_parameters():
#         if n.endswith(".quantiles") and p.requires_grad:
#             aux_parameters.add(n)
#         elif n.startswith("g_a.") and p.requires_grad:
#             cpu_parameters.add(n)
#         elif p.requires_grad:
#             gpu_parameters.add(n)

#     # Make sure we don't have overlapping parameters
#     params_dict = dict(net.named_parameters())
#     assert len(gpu_parameters & cpu_parameters) == 0
#     assert len(gpu_parameters & aux_parameters) == 0
#     assert len(cpu_parameters & aux_parameters) == 0
#     total_params = len(gpu_parameters) + len(cpu_parameters) + len(aux_parameters)
#     assert total_params == len(params_dict)

#     print(f"GPU parameters: {len(gpu_parameters)}")
#     print(f"CPU parameters: {len(cpu_parameters)}")
#     print(f"AUX parameters: {len(aux_parameters)}")
    
#     # Create separate optimizers for CPU and GPU parameters
#     gpu_optimizer = optim.Adam(
#         (params_dict[n] for n in sorted(gpu_parameters)),
#         lr=args.learning_rate,
#     )
    
#     cpu_optimizer = optim.Adam(
#         (params_dict[n] for n in sorted(cpu_parameters)),
#         lr=args.learning_rate,
#     )
    
#     aux_optimizer = optim.Adam(
#         (params_dict[n] for n in sorted(aux_parameters)),
#         lr=args.aux_learning_rate,
#     )
    
#     return gpu_optimizer, cpu_optimizer, aux_optimizer


# def train_one_epoch(
#     model, criterion, train_dataloader, gpu_optimizer, cpu_optimizer, aux_optimizer, epoch, clip_max_norm, train_sampler, type='mse', args=None, lr_scheduler=None
# ):
#     model.train()
#     gpu_device = "cuda"  # Device for the main model
#     cpu_device = "cpu"   # Device for g_a encoder

#     if torch.cuda.device_count() > 1:
#         train_sampler.set_epoch(epoch)

#     pre_time = 0
#     now_time = time.time()
#     for i, d in enumerate(train_dataloader):
#         # Step 1: Split the data processing between CPU and GPU
#         print(f'train data is on {d.device}')
#         d_cpu = d.detach().cpu()  # For encoder on CPU
#         d_gpu = d.to(gpu_device)  # For rest of model on GPU
        
#         # Step 2: Clear gradients on all optimizers
#         gpu_optimizer.zero_grad()
#         cpu_optimizer.zero_grad()
#         aux_optimizer.zero_grad()
        
#         # Step 3: Forward pass - the model's forward method handles CPU/GPU transitions
#         out_net = model(d_gpu)  # Model internally manages data transfers
        
#         # Step 4: Compute loss (on GPU)
#         out_criterion = criterion(out_net, d_gpu)
#         print("train/step 4: done")
#         # print(
#         # "  x_hat device:", out_net["x_hat"].device,
#         # "\n  target device:", d_gpu.device,
#         # "\n  loss device:", out_criterion["loss"].device)   
        
#         # Step 5: Backward pass
#         out_criterion["loss"].backward()
#         print("train/step 5: done")

#         # Step 6: Apply gradient clipping if needed
#         if clip_max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
#         print("train/step 6: done")
        
#         # Step 7: Update model parameters with separate optimizers
#         gpu_optimizer.step()
#         cpu_optimizer.step()
#         print("train/step 7: done")

        
#         # Step 8: Handle auxiliary loss for entropy bottleneck
#         aux_loss = model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss()
#         aux_loss.backward()
#         aux_optimizer.step()
#         print("train/step 8: done")

#         # Print progress and statistics
#         if (i+1) % 10 == 0:
#             pre_time = now_time
#             now_time = time.time()
#             print(f'time : {now_time-pre_time}\n', end='')
#             print(f'lr : {lr_scheduler.get_last_lr()[0]}\n', end='')
            
#             if type == 'mse':
#                 print(
#                     f"Train epoch {epoch}: ["
#                     f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
#                     f'\tLoss: {out_criterion["loss"].item():.3f} |'
#                     f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
#                     f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
#                     f"\tAux loss: {aux_loss.item():.2f}"
#                 )
#             else:
#                 print(
#                     f"Train epoch {epoch}: ["
#                     f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}"
#                     f'\tLoss: {out_criterion["loss"].item():.3f} |'
#                     f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
#                     f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
#                     f"\tAux loss: {aux_loss.item():.2f}"
#                 )


# def test_epoch(epoch, test_dataloader, model, criterion, type='mse', args=None):
#     model.eval()
#     gpu_device = "cuda"
    
#     if type == 'mse':
#         loss = AverageMeter()
#         bpp_loss = AverageMeter()
#         mse_loss = AverageMeter()
#         aux_loss = AverageMeter()

#         with torch.no_grad():
#             for d in test_dataloader:
#                 d_cpu = d.detach().cpu()  # For encoder
#                 d_gpu = d.to(gpu_device)  # For criterion
                
#                 out_net = model(d_gpu)
#                 out_criterion = criterion(out_net, d_gpu)
                
#                 loss.update(out_criterion["loss"])
#                 aux_loss.update(model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss())
#                 mse_loss.update(out_criterion["mse_loss"])
#                 bpp_loss.update(out_criterion["bpp_loss"])

#         print(
#             f"Test epoch {epoch}: Average losses:"
#             f"\tLoss: {loss.avg:.3f} |"
#             f"\tMSE loss: {mse_loss.avg:.3f} |"
#             f"\tBpp loss: {bpp_loss.avg:.2f} |"
#             f"\tAux loss: {aux_loss.avg:.2f}\n"
#         )

#     else:
#         loss = AverageMeter()
#         bpp_loss = AverageMeter()
#         ms_ssim_loss = AverageMeter()
#         aux_loss = AverageMeter()

#         with torch.no_grad():
#             for d in test_dataloader:
#                 d_cpu = d.detach().cpu()  # For encoder
#                 d_gpu = d.to(gpu_device)  # For criterion
                
#                 out_net = model(d_gpu)
#                 out_criterion = criterion(out_net, d_gpu)
                
#                 aux_loss.update(model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss())
#                 bpp_loss.update(out_criterion["bpp_loss"])
#                 loss.update(out_criterion["loss"])
#                 ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

#         print(
#             f"Test epoch {epoch}: Average losses:"
#             f"\tLoss: {loss.avg:.3f} |"
#             f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
#             f"\tBpp loss: {bpp_loss.avg:.2f} |"
#             f"\tAux loss: {aux_loss.avg:.2f}\n"
#         )

#     return loss.avg

# def pad(x, p):
#     h, w = x.size(2), x.size(3)
#     new_h = (h + p - 1) // p * p
#     new_w = (w + p - 1) // p * p
#     padding_left = (new_w - w) // 2
#     padding_right = new_w - w - padding_left
#     padding_top = (new_h - h) // 2
#     padding_bottom = new_h - h - padding_top
#     x_padded = F.pad(
#         x,
#         (padding_left, padding_right, padding_top, padding_bottom),
#         mode="constant",
#         value=0,
#     )
#     return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

# def crop(x, padding):
#     return F.pad(
#         x,
#         (-padding[0], -padding[1], -padding[2], -padding[3]),
#     )

# def save_checkpoint(state, is_best, epoch, save_path, filename):
#     '''
#     save checkpoint of every epoch and the best loss
#     '''
#     torch.save(state, save_path + "checkpoint_latest.pth.tar")
#     if epoch % 5 == 0:
#         torch.save(state, filename)
#     if is_best:
#         torch.save(state, save_path + "checkpoint_best.pth.tar")


# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Example training script.")

#     parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
#     parser.add_argument(
#         "-d", "--dataset", type=str, required=True, help="Training dataset"
#     )
#     parser.add_argument(
#         "-e",
#         "--epochs",
#         default=50,
#         type=int,
#         help="Number of epochs (default: %(default)s)",
#     )
#     parser.add_argument(
#         "-lr",
#         "--learning-rate",
#         default=1e-4,
#         type=float,
#         help="Learning rate (default: %(default)s)",
#     )
#     parser.add_argument(
#         "-n",
#         "--num-workers",
#         type=int,
#         default=20,
#         help="Dataloaders threads (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--lambda",
#         dest="lmbda",
#         type=float,
#         default=3,
#         help="Bit-rate distortion parameter (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
#     )
#     parser.add_argument(
#         "--test-batch-size",
#         type=int,
#         default=8,
#         help="Test batch size (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--aux-learning-rate",
#         default=1e-3,
#         type=float,
#         help="Auxiliary loss learning rate (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--patch-size",
#         type=int,
#         nargs=2,
#         default=(256, 256),
#         help="Size of the patches to be cropped (default: %(default)s)",
#     )
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--save", action="store_true", default=True, help="Save model to disk"
#     )
#     parser.add_argument(
#         "--drop", action="store_true", default=False,
#     )
#     parser.add_argument(
#         "--seed", type=float, default=100, help="Set random seed for reproducibility"
#     )
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
#     parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim", "l1"])
#     parser.add_argument("--save_path", type=str, help="save_path")
#     parser.add_argument(
#         "--N", type=int, default=128,
#     )
#     parser.add_argument(
#         "--M", type=int, default=320,
#     )
#     parser.add_argument(
#         "--lr_epoch", nargs='+', type=int
#     )
#     parser.add_argument(
#         "--continue_train", action="store_true", default=True
#     )
#     args = parser.parse_args(argv)
#     return args


# def main(argv):
#     # torch.autograd.set_detect_anomaly(True)

#     args = parse_args(argv)
#     for arg in vars(args):
#         print(arg, ":", getattr(args, arg))
#     type = args.type
#     save_path = os.path.join(args.save_path, str(args.lmbda))
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#         os.makedirs(save_path + "tensorboard/")

#     if args.seed is not None:
#         torch.manual_seed(args.seed)
#         random.seed(args.seed)
        
#     writer = SummaryWriter(save_path + "tensorboard/")

#     train_transforms = transforms.Compose(
#         [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
#     )
#     test_transforms = transforms.Compose(
#         [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
#     )

#     train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
#     test_dataset = ImageFolder(args.dataset, split="train", transform=test_transforms)

#     if args.local_rank != -1:
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device("cuda", args.local_rank)
#         torch.distributed.init_process_group(backend="nccl", init_method='env://')
#     else: 
#         device = "cuda"
        
#     # Initialize the model
#     net = DCAE_3()
#     net = net.to(device)
#     print(f'I am on {device}')
    
#     # Move everything except g_a to GPU (g_a is already on CPU from model definition)
#     for name, module in net.named_children():
#         if name == 'g_a':
#             module.to("cpu")
#     # # Only check modules with parameters
#     # for name, param in net.named_parameters():
#     #     print(f"Parameter: {name}, Device: {param.device}")

#     # For DDP with mixed CPU/GPU, we need to handle it differently
#     if args.cuda and torch.cuda.device_count() > 1:
#         # When using distributed training with a CPU component, we need a custom approach
#         # Set find_unused_parameters=True since g_a params will be on CPU
#         net = nn.parallel.DistributedDataParallel(
#             net, 
#             device_ids=[args.local_rank], 
#             output_device=args.local_rank, 
#             find_unused_parameters=True
#         )
#         train_sampler = DistributedSampler(train_dataset)
#         test_sampler = DistributedSampler(test_dataset)
#     else:
#         train_sampler = None
#         test_sampler = None
        
#     train_dataloader = DataLoader(
#         train_dataset,
#         sampler=train_sampler if torch.cuda.device_count() > 1 else None,
#         shuffle=True if train_sampler is None else False,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )

#     test_dataloader = DataLoader(
#         test_dataset,
#         sampler=test_sampler if torch.cuda.device_count() > 1 else None,
#         batch_size=args.test_batch_size,
#         num_workers=args.num_workers,
#         shuffle=False,
#         pin_memory=True,
#     )
    
#     # Configure optimizers - separate optimizers for CPU and GPU parameters
#     gpu_optimizer, cpu_optimizer, aux_optimizer = configure_optimizers(net, args)
#     milestones = args.lr_epoch
#     print("milestones: ", milestones)

#     # Set up learning rate scheduler for both optimizers
#     gpu_lr_scheduler = optim.lr_scheduler.MultiStepLR(gpu_optimizer, milestones, gamma=0.1, last_epoch=-1)
#     cpu_lr_scheduler = optim.lr_scheduler.MultiStepLR(cpu_optimizer, milestones, gamma=0.1, last_epoch=-1)

#     criterion = RateDistortionLoss(lmbda=args.lmbda, type=type).to(device)
#     last_epoch = 0

#     # Load checkpoint if provided
#     if args.checkpoint:  
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location=device)
        
#         # Handle loading state dict with potential device differences
#         dictory = {}
#         for k, v in checkpoint["state_dict"].items():
#             k_no_module = k.replace("module.", "")
#             # Check if parameter belongs to g_a (CPU) or rest of model (GPU)
#             if k_no_module.startswith("g_a."):
#                 dictory[k_no_module] = v.cpu()  # Ensure g_a parameters are on CPU
#             else:
#                 dictory[k_no_module] = v.to(device)  # Rest to GPU
                
#         net.load_state_dict(dictory)

#         if args.continue_train:
#             last_epoch = checkpoint["epoch"] + 1
#             # Handle separate optimizers for CPU/GPU
#             if "gpu_optimizer" in checkpoint and "cpu_optimizer" in checkpoint:
#                 gpu_optimizer.load_state_dict(checkpoint["gpu_optimizer"])
#                 cpu_optimizer.load_state_dict(checkpoint["cpu_optimizer"])
#             else:
#                 # Handle backward compatibility with old checkpoints
#                 print("Warning: Loading from a checkpoint with single optimizer.")
#                 print("Optimizer states might not be correctly restored.")
#                 # Attempt to load what we can from the old format
#                 gpu_optimizer.load_state_dict(checkpoint["optimizer"])
#                 cpu_optimizer.load_state_dict(checkpoint["optimizer"])
                
#             aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
#             gpu_lr_scheduler.load_state_dict(checkpoint["gpu_lr_scheduler"] if "gpu_lr_scheduler" in checkpoint else checkpoint["lr_scheduler"])
#             cpu_lr_scheduler.load_state_dict(checkpoint["cpu_lr_scheduler"] if "cpu_lr_scheduler" in checkpoint else checkpoint["lr_scheduler"])

#     best_loss = float("inf")
#     for epoch in range(last_epoch, args.epochs):
#         train_one_epoch(
#             net,
#             criterion,
#             train_dataloader,
#             gpu_optimizer,
#             cpu_optimizer,
#             aux_optimizer,
#             epoch,
#             args.clip_max_norm,
#             train_sampler if torch.cuda.device_count() > 1 else None,
#             type,
#             args,
#             gpu_lr_scheduler  # Just pass the GPU scheduler for printing
#         )

#         loss = test_epoch(epoch, test_dataloader, net, criterion, type, args)
#         writer.add_scalar('test_loss', loss, epoch)

#         global_rank = dist.get_rank() if torch.cuda.device_count() > 1 else 0
        
#         # Step both schedulers
#         gpu_lr_scheduler.step()
#         cpu_lr_scheduler.step()

#         is_best = loss < best_loss
#         best_loss = min(loss, best_loss)

#         if args.save and global_rank == 0:
#             save_checkpoint(
#                 {
#                     "epoch": epoch,
#                     "state_dict": net.state_dict(),
#                     "loss": loss,
#                     "gpu_optimizer": gpu_optimizer.state_dict(),
#                     "cpu_optimizer": cpu_optimizer.state_dict(),
#                     "aux_optimizer": aux_optimizer.state_dict(),
#                     "gpu_lr_scheduler": gpu_lr_scheduler.state_dict(),
#                     "cpu_lr_scheduler": cpu_lr_scheduler.state_dict(),
#                 },
#                 is_best,
#                 epoch,
#                 save_path,
#                 save_path + str(epoch) + "_checkpoint.pth.tar",
#             )

# if __name__ == "__main__":
#     main(sys.argv[1:])