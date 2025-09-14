# import os
# os.environ['TMPDIR'] = '/tmp'
# import argparse
# import math
# import random
# import sys
# import time

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# from torch.utils.data import DataLoader
# from torch import distributed as dist
# from torch.utils.data.distributed import DistributedSampler
# from torchvision import transforms

# from compressai.datasets import ImageFolder
# from pytorch_msssim import ms_ssim

# from models import DCAE
# from torch.utils.tensorboard import SummaryWriter   

# torch.set_num_threads(8)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# def freeze_decompress_and_shared_components(model):
#     """Freeze all components except compress-unique ones (g_a, h_a)"""
    
#     # Freeze all parameters first
#     for param in model.parameters():
#         param.requires_grad = False
    
#     # Enable gradients only for compress-unique components
#     compress_unique_modules = ['g_a', 'h_a']
    
#     for name, module in model.named_modules():
#         # Check if this module is compress-unique
#         for comp_name in compress_unique_modules:
#             if name.startswith(comp_name) or name.endswith(comp_name):
#                 for param in module.parameters():
#                     param.requires_grad = True
#                 print(f"Enabled gradients for: {name}")
#                 break

# def get_trainable_parameters(model):
#     """Get only the trainable parameters (compress-unique components)"""
#     trainable_params = []
#     total_params = 0
#     trainable_count = 0
    
#     for name, param in model.named_parameters():
#         total_params += param.numel()
#         if param.requires_grad:
#             trainable_params.append(param)
#             trainable_count += param.numel()
#             print(f"Trainable parameter: {name}, shape: {param.shape}")
    
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_count:,}")
#     print(f"Frozen parameters: {total_params - trainable_count:,}")
    
#     return trainable_params

# class RateDistortionLoss(nn.Module):
#     """Custom rate distortion loss with a Lagrangian parameter."""

#     def __init__(self, lmbda=1e-2, type='mse'):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.lmbda = lmbda
#         self.type = type

#     def forward(self, output, target):
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
#             out['ms_ssim_loss'] = ms_ssim(output["x_hat"], target, data_range=1.)
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

# def configure_compress_optimizer(net, args):
#     """Configure optimizer only for compress-unique parameters"""
    
#     # Get only trainable parameters (compress-unique components)
#     trainable_params = get_trainable_parameters(net)
    
#     if not trainable_params:
#         raise ValueError("No trainable parameters found!")
    
#     optimizer = optim.Adam(
#         trainable_params,
#         lr=args.learning_rate,
#     )
    
#     # Note: No aux_optimizer needed since we're not training entropy models
#     return optimizer

# def train_compress_only_epoch(
#     model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, 
#     train_sampler, type='mse', args=None, lr_scheduler=None
# ):
#     model.train()
#     device = next(model.parameters()).device

#     if args.cuda and torch.cuda.device_count() > 1:
#         train_sampler.set_epoch(epoch)

#     pre_time = 0
#     now_time = time.time()
    
#     for i, d in enumerate(train_dataloader):
#         d = d.to(device)
#         optimizer.zero_grad()
        
#         # Forward pass
#         out_net = model(d)
#         out_criterion = criterion(out_net, d)
        
#         # Only backprop through compress-unique components
#         out_criterion["loss"].backward()

#         if clip_max_norm > 0:
#             # Only clip gradients of trainable parameters
#             trainable_params = [p for p in model.parameters() if p.requires_grad]
#             torch.nn.utils.clip_grad_norm_(trainable_params, clip_max_norm) 
        
#         optimizer.step()

#         if (i+1) % 10 == 0:
#             pre_time = now_time
#             now_time = time.time()
#             print(f'time : {now_time-pre_time}')
#             print(f'lr : {lr_scheduler.get_last_lr()[0]}')
            
#             if type == 'mse':
#                 print(
#                     f"Train epoch {epoch}: ["
#                     f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
#                     f'\tLoss: {out_criterion["loss"].item():.3f} |'
#                     f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
#                     f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f}'
#                 )

# def test_epoch(epoch, test_dataloader, model, criterion, type='mse', args=None):
#     model.eval()
#     device = next(model.parameters()).device
    
#     loss = AverageMeter()
#     bpp_loss = AverageMeter()
#     mse_loss = AverageMeter()

#     with torch.no_grad():
#         for d in test_dataloader:
#             d = d.to(device)
#             out_net = model(d)
#             out_criterion = criterion(out_net, d)
#             loss.update(out_criterion["loss"])
#             mse_loss.update(out_criterion["mse_loss"])
#             bpp_loss.update(out_criterion["bpp_loss"])

#     print(
#         f"Test epoch {epoch}: Average losses:"
#         f"\tLoss: {loss.avg:.3f} |"
#         f"\tMSE loss: {mse_loss.avg:.3f} |"
#         f"\tBpp loss: {bpp_loss.avg:.2f}\n"
#     )
#     return loss.avg

# def save_checkpoint(state, is_best, epoch, save_path, filename):
#     """Save checkpoint of every epoch and the best loss"""
#     torch.save(state, save_path + "checkpoint_latest.pth.tar")
#     if epoch % 5 == 0:
#         torch.save(state, filename)
#     if is_best:
#         torch.save(state, save_path + "checkpoint_best.pth.tar")

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Compress-only training script.")

#     parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
#     parser.add_argument("-d", "--dataset", type=str, default='./dataset', help="Training dataset")
#     parser.add_argument("-e", "--epochs", default=50, type=int, help="Number of epochs")
#     parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate")
#     parser.add_argument("-n", "--num-workers", type=int, default=20, help="Dataloaders threads")
#     parser.add_argument("--lambda", dest="lmbda", type=float, default=60.5, help="Rate-distortion parameter")
#     parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
#     parser.add_argument("--test-batch-size", type=int, default=8, help="Test batch size")
#     parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Patch size")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
#     parser.add_argument("--seed", type=float, default=100, help="Random seed")
#     parser.add_argument("--clip_max_norm", default=1.0, type=float, help="Gradient clipping max norm")
#     parser.add_argument("--checkpoint", type=str, default="60.5checkpoint_best.pth.tar", help="Path to pretrained checkpoint")
#     parser.add_argument("--type", type=str, default='mse', help="Loss type", choices=['mse', "ms-ssim"])
#     parser.add_argument("--save_path", type=str, help="Save path", default='./train_compress_only')
#     parser.add_argument("--lr_epoch", nargs='+', type=int, default=[30, 40])
    
#     args = parser.parse_args(argv)
#     return args

# def main(argv):
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

#     # Data loading
#     train_transforms = transforms.Compose([
#         transforms.RandomCrop(args.patch_size), 
#         transforms.ToTensor()
#     ])
#     test_transforms = transforms.Compose([
#         transforms.CenterCrop(args.patch_size), 
#         transforms.ToTensor()
#     ])

#     train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
#     test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

#     # Fix device selection logic
#     if args.local_rank != -1:
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device("cuda", args.local_rank)
#         torch.distributed.init_process_group(backend="nccl", init_method='env://')
#     else: 
#         device = "cuda" if args.cuda else "cpu"
        
#     # Load pretrained model
#     net = DCAE()
#     net = net.to(device)
    
#     print("Loading pretrained model from:", args.checkpoint)
#     checkpoint = torch.load(args.checkpoint, map_location=device)
    
#     # Load state dict
#     state_dict = {}
#     for k, v in checkpoint["state_dict"].items():
#         state_dict[k.replace("module.", "")] = v
#     net.load_state_dict(state_dict)
    
#     # Freeze decompress and shared components, keep only compress-unique trainable
#     freeze_decompress_and_shared_components(net)
    
#     if args.cuda and torch.cuda.device_count() > 1:
#         net = nn.parallel.DistributedDataParallel(
#             net, device_ids=[args.local_rank], 
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
#         sampler=train_sampler,
#         shuffle=True if train_sampler is None else None,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=args.cuda,
#     )

#     test_dataloader = DataLoader(
#         test_dataset,
#         sampler=test_sampler,
#         batch_size=args.test_batch_size,
#         num_workers=args.num_workers,
#         shuffle=False,
#         pin_memory=args.cuda,
#     )

#     # Configure optimizer only for compress-unique components
#     optimizer = configure_compress_optimizer(net, args)
#     milestones = args.lr_epoch
#     print("milestones: ", milestones)

#     lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
#     criterion = RateDistortionLoss(lmbda=args.lmbda, type=type).to(device)

#     best_loss = float("inf")
    
#     for epoch in range(args.epochs):
#         train_compress_only_epoch(
#             net, criterion, train_dataloader, optimizer, epoch,
#             args.clip_max_norm, train_sampler,
#             type, args, lr_scheduler
#         )

#         loss = test_epoch(epoch, test_dataloader, net, criterion, type, args)
#         writer.add_scalar('test_loss', loss, epoch)

#         global_rank = dist.get_rank() if args.cuda and torch.cuda.device_count() > 1 else 0
#         lr_scheduler.step()

#         is_best = loss < best_loss
#         best_loss = min(loss, best_loss)

#         if args.save and global_rank == 0:
#             save_checkpoint(
#                 {
#                     "epoch": epoch,
#                     "state_dict": net.state_dict(),
#                     "loss": loss,
#                     "optimizer": optimizer.state_dict(),
#                     "lr_scheduler": lr_scheduler.state_dict(),
#                 },
#                 is_best, epoch, save_path,
#                 save_path + str(epoch) + "_checkpoint.pth.tar"
#             )

# if __name__ == "__main__":
#     main(sys.argv[1:])
#     print('done')



import os
os.environ['TMPDIR'] = '/tmp'
import argparse
import math
import random
import sys
import time

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

from models import DCAE
from torch.utils.tensorboard import SummaryWriter   

torch.set_num_threads(8)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MixedDeviceDCAE(nn.Module):
    """DCAE wrapper that handles mixed device placement"""
    
    def __init__(self, original_model, cpu_device='cpu', gpu_device='cuda'):
        super().__init__()
        self.cpu_device = cpu_device
        self.gpu_device = gpu_device
        
        # Move compress-unique components to CPU
        self.g_a = original_model.g_a.to(cpu_device)  # Encoder
        self.h_a = original_model.h_a.to(cpu_device)  # Hyper encoder
        
        # Keep decompress-unique components on GPU
        self.g_s = original_model.g_s.to(gpu_device)  # Decoder
        self.h_z_s1 = original_model.h_z_s1.to(gpu_device)  # Hyper decoder 1
        self.h_z_s2 = original_model.h_z_s2.to(gpu_device)  # Hyper decoder 2
        
        # Keep shared components on GPU
        self.dt = original_model.dt.to(gpu_device)
        self.dt_cross_attention = original_model.dt_cross_attention.to(gpu_device)
        
        # Context prediction modules
        self.cc_mean_transforms = original_model.cc_mean_transforms.to(gpu_device)
        self.cc_scale_transforms = original_model.cc_scale_transforms.to(gpu_device)
        self.lrp_transforms = original_model.lrp_transforms.to(gpu_device)
        
        # Entropy models
        self.entropy_bottleneck = original_model.entropy_bottleneck.to(gpu_device)
        self.gaussian_conditional = original_model.gaussian_conditional.to(gpu_device)
        
        # Model parameters
        self.num_slices = original_model.num_slices
        self.max_support_slices = original_model.max_support_slices
        
    def forward(self, x):
        """Forward pass with mixed device handling"""
        b = x.size(0)
        
        # Move input to CPU for compress-unique processing (g_a, h_a)
        x_cpu = x.to(self.cpu_device)
        
        # Encoder (CPU processing)
        y_cpu = self.g_a(x_cpu)
        
        # Hyper encoder (CPU processing)
        z_cpu = self.h_a(y_cpu)
        
        # Move y and z to GPU for the rest of processing
        y = y_cpu.to(self.gpu_device)
        z = z_cpu.to(self.gpu_device)
        y_shape = y.shape[2:]
        
        # Dictionary processing (GPU)
        dt = self.dt.repeat([b, 1, 1])
        
        # Entropy bottleneck processing (GPU)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = self._ste_round(z_tmp) + z_offset
        
        # Hyper decoders (GPU)
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        # Slice processing (GPU)
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)
            
            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)
            
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = self._ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        
        # Decoder (GPU)
        x_hat = self.g_s(y_hat)
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales, "y": y}
        }
    
    def _ste_round(self, x):
        """Straight-through estimator round"""
        return torch.round(x) - x.detach() + x
    
    def aux_loss(self):
        """Auxiliary loss from entropy bottleneck"""
        return self.entropy_bottleneck.loss()
    
    def load_state_dict_mixed_device(self, state_dict, strict=True):
        """Custom state dict loading for mixed device setup"""
        
        # Handle entropy model buffers first (similar to original DCAE)
        from DCAE import update_registered_buffers
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        
        # Clean parameter names (remove 'module.' prefix if present)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            cleaned_key = k.replace("module.", "")
            cleaned_state_dict[cleaned_key] = v
        
        # Load parameters to appropriate devices
        compress_unique_modules = ['g_a', 'h_a']
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in cleaned_state_dict:
                    # Determine target device based on module type
                    is_compress_unique = any(name.startswith(comp_name) for comp_name in compress_unique_modules)
                    target_device = self.cpu_device if is_compress_unique else self.gpu_device
                    
                    # Load parameter to correct device
                    loaded_param = cleaned_state_dict[name].to(target_device)
                    param.copy_(loaded_param)
                    print(f"Loaded {name} -> {target_device}")
                elif strict:
                    raise KeyError(f"Missing key in state_dict: {name}")
        
        # Load buffers
        for name, buffer in self.named_buffers():
            if name in cleaned_state_dict:
                # Determine target device
                is_compress_unique = any(name.startswith(comp_name) for comp_name in compress_unique_modules)
                target_device = self.cpu_device if is_compress_unique else self.gpu_device
                
                loaded_buffer = cleaned_state_dict[name].to(target_device)
                buffer.copy_(loaded_buffer)
                print(f"Loaded buffer {name} -> {target_device}")
            elif strict and not name.startswith('gaussian_conditional'):  # Skip entropy model buffers
                raise KeyError(f"Missing buffer in state_dict: {name}")

def freeze_decompress_and_shared_components(model):
    """Freeze all components except compress-unique ones (g_a, h_a)"""
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable gradients only for compress-unique components (g_a and h_a)
    compress_unique_modules = ['g_a', 'h_a']
    
    for name, module in model.named_modules():
        # Check if this module is compress-unique
        for comp_name in compress_unique_modules:
            if name.startswith(comp_name) or name.endswith(comp_name):
                for param in module.parameters():
                    param.requires_grad = True
                print(f"Enabled gradients for: {name}")
                break

def get_trainable_parameters(model):
    """Get only the trainable parameters (compress-unique components)"""
    trainable_params = []
    total_params = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append(param)
            trainable_count += param.numel()
            print(f"Trainable parameter: {name}, shape: {param.shape}, device: {param.device}")
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Frozen parameters: {total_params - trainable_count:,}")
    
    return trainable_params

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
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
            out['ms_ssim_loss'] = ms_ssim(output["x_hat"], target, data_range=1.)
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

def configure_compress_optimizer(net, args):
    """Configure optimizer only for compress-unique parameters"""
    
    # Get only trainable parameters (compress-unique components)
    trainable_params = get_trainable_parameters(net)
    
    if not trainable_params:
        raise ValueError("No trainable parameters found!")
    
    optimizer = optim.Adam(
        trainable_params,
        lr=args.learning_rate,
    )
    
    return optimizer

def train_compress_only_epoch(
    model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, 
    train_sampler, type='mse', args=None, lr_scheduler=None
):
    model.train()
    
    # Get the primary device (GPU) for moving targets
    gpu_device = next(p for p in model.parameters() if p.device.type == 'cuda').device

    if args.cuda and torch.cuda.device_count() > 1 and train_sampler is not None:
        train_sampler.set_epoch(epoch)

    pre_time = 0
    now_time = time.time()
    
    for i, d in enumerate(train_dataloader):
        # Move input data appropriately - model handles device placement internally
        target_gpu = d.to(gpu_device)  # Target stays on GPU for loss computation
        
        optimizer.zero_grad()
        
        # Forward pass - model handles mixed device internally
        out_net = model(d)  # Input goes to model, which moves it to CPU for g_a
        
        # Compute loss (both outputs and targets on GPU)
        out_criterion = criterion(out_net, target_gpu)
        
        # Backward pass
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            # Only clip gradients of trainable parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, clip_max_norm) 
        
        optimizer.step()

        if (i+1) % 100 == 0:
            pre_time = now_time
            now_time = time.time()
            print(f'time : {now_time-pre_time}')
            print(f'lr : {lr_scheduler.get_last_lr()[0]}')
            
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f}'
                )

def test_epoch(epoch, test_dataloader, model, criterion, type='mse', args=None):
    model.eval()
    
    # Get the primary device (GPU)
    gpu_device = next(p for p in model.parameters() if p.device.type == 'cuda').device
    
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            target_gpu = d.to(gpu_device)
            out_net = model(d)
            out_criterion = criterion(out_net, target_gpu)
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            bpp_loss.update(out_criterion["bpp_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f}\n"
    )
    return loss.avg

def save_checkpoint(state, is_best, epoch, save_path, filename):
    """Save checkpoint of every epoch and the best loss"""
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Mixed-device compress-only training script.")

    parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("-d", "--dataset", type=str, default='./dataset', help="Training dataset")
    parser.add_argument("-e", "--epochs", default=150, type=int, help="Number of epochs")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("-n", "--num-workers", type=int, default=20, help="Dataloaders threads")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=60.5, help="Rate-distortion parameter")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--test-batch-size", type=int, default=8, help="Test batch size")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Patch size")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=float, default=100, help="Random seed")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="Gradient clipping max norm")
    parser.add_argument("--checkpoint", type=str, default="60.5checkpoint_best.pth.tar", help="Path to pretrained checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="Loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="Save path", default='./train_compress_only_mixed_device')
    parser.add_argument("--lr_epoch", nargs='+', type=int, default=[30, 40])
    parser.add_argument("--continue_train", action="store_true", default=False, help="Continue training from checkpoint")
    
    args = parser.parse_args(argv)
    return args

def main(argv):
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

    # Data loading
    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size), 
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.CenterCrop(args.patch_size), 
        transforms.ToTensor()
    ])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    # Device setup
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    else: 
        device = "cuda" if args.cuda else "cpu"
        
    # Load pretrained model
    print("Loading pretrained model from:", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')  # Load to CPU first
    
    # Create original model first
    original_net = DCAE()
    
    # Load pretrained weights into original model
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        state_dict[k.replace("module.", "")] = v
    
    # Load into original model (all on CPU initially)
    original_net.load_state_dict(state_dict)
    print("Successfully loaded pretrained weights into original DCAE model")
    
    # Create mixed device model from pretrained model
    print("\nCreating mixed-device model with pretrained weights...")
    print("Compress-unique components (g_a, h_a) -> CPU")
    print("All other components -> GPU")
    
    if args.cuda:
        net = MixedDeviceDCAE(original_net, cpu_device='cpu', gpu_device=device)
    else:
        # If not using CUDA, put everything on CPU
        net = original_net.to(device)
    
    print("Mixed-device model created successfully with pretrained weights!")
    
    # Print device placement summary
    print("\n=== Device Placement Summary ===")
    for name, param in net.named_parameters():
        print(f"{name}: {param.device}")
    
    # Freeze decompress and shared components, keep only compress-unique trainable
    freeze_decompress_and_shared_components(net)
    
    # Note: No distributed training support for mixed-device model due to complexity
    if args.cuda and torch.cuda.device_count() > 1:
        print("Warning: Distributed training not supported with mixed-device model")
        print("Using single GPU training instead")
        
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.cuda,
    )

    # Configure optimizer only for compress-unique components
    optimizer = configure_compress_optimizer(net, args)
    milestones = args.lr_epoch
    print("milestones: ", milestones)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    
    if args.cuda:
        criterion = RateDistortionLoss(lmbda=args.lmbda, type=type).to(device)
    else:
        criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    # Handle continuing training from checkpoint
    last_epoch = 0
    if args.continue_train and 'epoch' in checkpoint:
        last_epoch = checkpoint["epoch"] + 1
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print(f"Continuing training from epoch {last_epoch}")

    best_loss = float("inf")
    
    print(f"\nStarting mixed-device training from pretrained model:")
    print(f"g_a, h_a (compress-unique) on: CPU")
    print(f"Other components on: {device}")
    print(f"Data flow: Input -> CPU (g_a, h_a) -> GPU (rest) -> Output")
    print(f"Starting from epoch: {last_epoch}")
    
    for epoch in range(last_epoch, args.epochs):
        train_compress_only_epoch(
            net, criterion, train_dataloader, optimizer, epoch,
            args.clip_max_norm, None,  # No distributed sampler for mixed device
            type, args, lr_scheduler
        )

        loss = test_epoch(epoch, test_dataloader, net, criterion, type, args)
        writer.add_scalar('test_loss', loss, epoch)

        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            # Create unified state dict for saving
            unified_state_dict = {}
            
            # Collect all parameters and buffers
            for name, param in net.named_parameters():
                unified_state_dict[name] = param.cpu()
            for name, buffer in net.named_buffers():
                unified_state_dict[name] = buffer.cpu()
            
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": unified_state_dict,
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best, epoch, save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar"
            )

if __name__ == "__main__":
    main(sys.argv[1:])