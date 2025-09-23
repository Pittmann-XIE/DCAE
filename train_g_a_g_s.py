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
from pytorch_msssim import ms_ssim

from models.g_a_g_s import SimpleAutoencoder
from torch.utils.tensorboard import SummaryWriter   
import os

torch.set_num_threads(8)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test_compute_psnr(a, b):
    b = b.to(a.device)
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def test_compute_msssim(a, b):
    b = b.to(a.device)
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class ReconstructionLoss(nn.Module):
    """Custom reconstruction loss for SimpleAutoencoder."""

    def __init__(self, loss_type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.loss_type = loss_type

    def forward(self, output, target):
        out = {}
        
        if self.loss_type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = out["mse_loss"]
        elif self.loss_type == 'l1':
            out["l1_loss"] = self.l1(output["x_hat"], target)
            out["loss"] = out["l1_loss"]
        elif self.loss_type == 'ms-ssim':
            out['ms_ssim_loss'] = 1 - compute_msssim(output["x_hat"], target)
            out["loss"] = out['ms_ssim_loss']
        elif self.loss_type == 'mixed':
            # Combination of MSE and MS-SSIM
            mse_loss = self.mse(output["x_hat"], target)
            ms_ssim_loss = 1 - compute_msssim(output["x_hat"], target)
            out["mse_loss"] = mse_loss
            out["ms_ssim_loss"] = ms_ssim_loss
            out["loss"] = 0.84 * ms_ssim_loss + 0.16 * mse_loss  # Commonly used weights
        
        # Calculate PSNR for monitoring
        out["psnr"] = compute_psnr(output["x_hat"], target)
        
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

def configure_optimizers(net, args):
    """Configure optimizer for SimpleAutoencoder (no auxiliary optimizer needed)."""
    
    optimizer = optim.Adam(
        net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    return optimizer

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, train_sampler, loss_type='mse', args=None, lr_scheduler=None
):
    model.train()
    device = next(model.parameters()).device

    if torch.cuda.device_count() > 1:
        train_sampler.set_epoch(epoch)

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    
    if loss_type == 'mse':
        mse_meter = AverageMeter()
    elif loss_type == 'l1':
        l1_meter = AverageMeter()
    elif loss_type == 'ms-ssim':
        ms_ssim_meter = AverageMeter()
    elif loss_type == 'mixed':
        mse_meter = AverageMeter()
        ms_ssim_meter = AverageMeter()

    pre_time = 0
    now_time = time.time()
    
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        
        loss = out_criterion["loss"]
        loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm) 
        
        optimizer.step()
        
        # Update metrics
        batch_size = d.size(0)
        loss_meter.update(loss.item(), batch_size)
        psnr_meter.update(out_criterion["psnr"].item(), batch_size)
        
        if loss_type == 'mse':
            mse_meter.update(out_criterion["mse_loss"].item(), batch_size)
        elif loss_type == 'l1':
            l1_meter.update(out_criterion["l1_loss"].item(), batch_size)
        elif loss_type == 'ms-ssim':
            ms_ssim_meter.update(out_criterion["ms_ssim_loss"].item(), batch_size)
        elif loss_type == 'mixed':
            mse_meter.update(out_criterion["mse_loss"].item(), batch_size)
            ms_ssim_meter.update(out_criterion["ms_ssim_loss"].item(), batch_size)

        if (i+1) % 100 == 0:
            pre_time = now_time
            now_time = time.time()
            print(f'Time: {now_time-pre_time:.2f}s')
            print(f'LR: {lr_scheduler.get_last_lr()[0]:.2e}')
            
            if loss_type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
                    f'\tLoss: {loss.item():.6f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.6f} |'
                    f'\tPSNR: {out_criterion["psnr"].item():.2f} dB'
                )
            elif loss_type == 'l1':
                print(
                    f"Train epoch {epoch}: ["
                    f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
                    f'\tLoss: {loss.item():.6f} |'
                    f'\tL1 loss: {out_criterion["l1_loss"].item():.6f} |'
                    f'\tPSNR: {out_criterion["psnr"].item():.2f} dB'
                )
            elif loss_type == 'ms-ssim':
                print(
                    f"Train epoch {epoch}: ["
                    f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
                    f'\tLoss: {loss.item():.6f} |'
                    f'\tMS-SSIM loss: {out_criterion["ms_ssim_loss"].item():.6f} |'
                    f'\tPSNR: {out_criterion["psnr"].item():.2f} dB'
                )
            elif loss_type == 'mixed':
                print(
                    f"Train epoch {epoch}: ["
                    f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
                    f'\tLoss: {loss.item():.6f} |'
                    f'\tMSE: {out_criterion["mse_loss"].item():.6f} |'
                    f'\tMS-SSIM: {out_criterion["ms_ssim_loss"].item():.6f} |'
                    f'\tPSNR: {out_criterion["psnr"].item():.2f} dB'
                )

def test_epoch(epoch, test_dataloader, model, criterion, loss_type='mse', args=None):
    model.eval()
    device = next(model.parameters()).device
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    
    if loss_type == 'mse':
        mse_meter = AverageMeter()
    elif loss_type == 'l1':
        l1_meter = AverageMeter()
    elif loss_type == 'ms-ssim':
        ms_ssim_meter = AverageMeter()
    elif loss_type == 'mixed':
        mse_meter = AverageMeter()
        ms_ssim_meter = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            
            batch_size = d.size(0)
            loss_meter.update(out_criterion["loss"].item(), batch_size)
            psnr_meter.update(out_criterion["psnr"].item(), batch_size)
            
            if loss_type == 'mse':
                mse_meter.update(out_criterion["mse_loss"].item(), batch_size)
            elif loss_type == 'l1':
                l1_meter.update(out_criterion["l1_loss"].item(), batch_size)
            elif loss_type == 'ms-ssim':
                ms_ssim_meter.update(out_criterion["ms_ssim_loss"].item(), batch_size)
            elif loss_type == 'mixed':
                mse_meter.update(out_criterion["mse_loss"].item(), batch_size)
                ms_ssim_meter.update(out_criterion["ms_ssim_loss"].item(), batch_size)

    if loss_type == 'mse':
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss_meter.avg:.6f} |"
            f"\tMSE loss: {mse_meter.avg:.6f} |"
            f"\tPSNR: {psnr_meter.avg:.2f} dB\n"
        )
    elif loss_type == 'l1':
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss_meter.avg:.6f} |"
            f"\tL1 loss: {l1_meter.avg:.6f} |"
            f"\tPSNR: {psnr_meter.avg:.2f} dB\n"
        )
    elif loss_type == 'ms-ssim':
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss_meter.avg:.6f} |"
            f"\tMS-SSIM loss: {ms_ssim_meter.avg:.6f} |"
            f"\tPSNR: {psnr_meter.avg:.2f} dB\n"
        )
    elif loss_type == 'mixed':
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss_meter.avg:.6f} |"
            f"\tMSE: {mse_meter.avg:.6f} |"
            f"\tMS-SSIM: {ms_ssim_meter.avg:.6f} |"
            f"\tPSNR: {psnr_meter.avg:.2f} dB\n"
        )

    return loss_meter.avg

def save_checkpoint(state, is_best, epoch, save_path, filename):
    """Save checkpoint of every epoch and the best loss."""
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train SimpleAutoencoder")

    parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument(
        "-d", "--dataset", type=str, default='./dataset', help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
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
        "--weight-decay",
        default=1e-5,
        type=float,
        help="Weight decay (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
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
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda", default=True)
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s)",
    )
    parser.add_argument(
        "--dcae-checkpoint", 
        type=str, 
        help="Path to pretrained DCAE checkpoint", 
        default='60.5checkpoint_best.pth.tar'
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to SimpleAutoencoder checkpoint to resume training"
    )
    parser.add_argument(
        "--type", 
        type=str, 
        default='ms-ssim', 
        help="loss type", 
        choices=['mse', "ms-ssim", "l1", "mixed"]
    )
    parser.add_argument("--save_path", type=str, help="save_path", default='./checkpoints/train_simple')
    parser.add_argument(
        "--N", type=int, default=192,
    )
    parser.add_argument(
        "--M", type=int, default=160,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int, default=[50, 80]
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True, help="Continue training if a checkpoint is provided"
    )
    args = parser.parse_args(argv)
    return args


def debug_architecture_mismatch(dcae_checkpoint_path, simple_model):
    """Debug function to compare architectures"""
    checkpoint = torch.load(dcae_checkpoint_path, map_location='cpu')
    dcae_state_dict = checkpoint.get('state_dict', checkpoint)
    simple_state_dict = simple_model.state_dict()
    
    dcae_ga_keys = {k for k in dcae_state_dict.keys() if k.startswith('module.g_a.')}
    dcae_gs_keys = {k for k in dcae_state_dict.keys() if k.startswith('module.g_s.')}
    
    simple_ga_keys = {k for k in simple_state_dict.keys() if k.startswith('g_a.')}
    simple_gs_keys = {k for k in simple_state_dict.keys() if k.startswith('g_s.')}
    
    print(f"DCAE g_a keys: {len(dcae_ga_keys)}")
    print(f"Simple g_a keys: {len(simple_ga_keys)}")
    print(f"DCAE g_s keys: {len(dcae_gs_keys)}")
    print(f"Simple g_s keys: {len(simple_gs_keys)}")
    
    print("\nCommon g_a keys:", len(dcae_ga_keys & simple_ga_keys))
    print("Common g_s keys:", len(dcae_gs_keys & simple_gs_keys))


def inspect_checkpoint(checkpoint_path):
    """Inspect the actual contents of the checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("=== Checkpoint Structure ===")
    print("Main keys:", list(checkpoint.keys()))
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"\nTotal state_dict keys: {len(state_dict)}")
    else:
        state_dict = checkpoint
        print(f"\nTotal checkpoint keys: {len(state_dict)}")
    
    # Remove 'module.' prefix and group by the next level
    clean_prefixes = {}
    for key in state_dict.keys():
        # Remove 'module.' prefix
        clean_key = key.replace('module.', '', 1)
        prefix = clean_key.split('.')[0]
        
        if prefix not in clean_prefixes:
            clean_prefixes[prefix] = []
        clean_prefixes[prefix].append(clean_key)
    
    print("=== Components after removing 'module.' prefix ===")
    for prefix, keys in sorted(clean_prefixes.items()):
        print(f"{prefix}: {len(keys)} keys")
        # Show first few keys
        for key in keys[:3]:
            print(f"  - {key}")
        if len(keys) > 3:
            print("  - ...")
        print()
    
    return state_dict

# Add this to your main function before creating the model


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    
    loss_type = args.type
    save_path = os.path.join(args.save_path, loss_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "/tensorboard/")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        
    writer = SummaryWriter(save_path + "/tensorboard/")

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    else: 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Initialize SimpleAutoencoder from DCAE checkpoint
    # print("=== Inspecting DCAE Checkpoint ===")
    # inspect_checkpoint(args.dcae_checkpoint)
    net = SimpleAutoencoder.from_dcae(args.dcae_checkpoint, head_dim=[8, 16, 32, 32, 16, 8], ignore_shape_mismatch=True, N=args.N, M=args.M)
    net = net.to(device)

    debug_architecture_mismatch(args.dcae_checkpoint, net)

    if args.cuda and torch.cuda.device_count() > 1:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
    else:
        train_sampler = None
        test_sampler = None
        
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    print("LR milestones: ", milestones)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    criterion = ReconstructionLoss(loss_type=loss_type).to(device)
    
    last_epoch = 0
    best_loss = float("inf")

    # Load SimpleAutoencoder checkpoint if specified
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print("Loading SimpleAutoencoder checkpoint", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if torch.cuda.device_count() > 1:
            net.module.load_state_dict(checkpoint["state_dict"])
        else:
            net.load_state_dict(checkpoint["state_dict"])

        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_loss = checkpoint.get("best_loss", float("inf"))

    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            train_sampler,
            loss_type,
            args,
            lr_scheduler
        )

        loss = test_epoch(epoch, test_dataloader, net, criterion, loss_type, args)
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
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path + "/",
                save_path + "/" + str(epoch) + "_checkpoint.pth.tar",
            )

if __name__ == "__main__":
    main(sys.argv[1:])