# ## Train 1
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

# # Import your models
# from models import (
#     CompressModel,
#     DecompressModel,
#     ParameterSync
# )
# from torch.utils.tensorboard import SummaryWriter   
# import os
# torch.set_num_threads(8)
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False

# import numpy as np

# def test_compute_psnr(a, b):
#     b = b.to(a.device)
#     mse = torch.mean((a - b)**2).item()
#     return -10 * math.log10(mse)

# def test_compute_msssim(a, b):
#     b = b.to(a.device)
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

# def compute_msssim(a, b):
#     return ms_ssim(a, b, data_range=1.)

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

# def sync_gradients_cpu_gpu(cpu_model, gpu_model):
#     """Sync gradients for shared parameters from GPU to CPU"""
#     # Dictionary gradients
#     if gpu_model.dt.grad is not None:
#         cpu_model.dt.grad = gpu_model.dt.grad.cpu()
    
#     # Cross-attention gradients and ensure GPU parameters stay on GPU
#     for i, (cpu_module, gpu_module) in enumerate(zip(cpu_model.dt_cross_attention, gpu_model.dt_cross_attention)):
#         for (cpu_name, cpu_param), (gpu_name, gpu_param) in zip(cpu_module.named_parameters(), gpu_module.named_parameters()):
#             if gpu_param.grad is not None:
#                 cpu_param.grad = gpu_param.grad.cpu()
#             # Ensure GPU parameters stay on GPU
#             gpu_param.data = gpu_param.data.cuda()
        
#         # Ensure GPU buffers stay on GPU (including layer norm stats)
#         for gpu_name, gpu_buffer in gpu_module.named_buffers():
#             gpu_buffer.data = gpu_buffer.data.cuda()
    
#     # Apply same pattern to other shared components
#     for cpu_modules, gpu_modules in [
#         (cpu_model.cc_mean_transforms, gpu_model.cc_mean_transforms),
#         (cpu_model.cc_scale_transforms, gpu_model.cc_scale_transforms),
#         (cpu_model.lrp_transforms, gpu_model.lrp_transforms)
#     ]:
#         for cpu_module, gpu_module in zip(cpu_modules, gpu_modules):
#             for (cpu_name, cpu_param), (gpu_name, gpu_param) in zip(cpu_module.named_parameters(), gpu_module.named_parameters()):
#                 if gpu_param.grad is not None:
#                     cpu_param.grad = gpu_param.grad.cpu()
#                 gpu_param.data = gpu_param.data.cuda()
            
#             for gpu_name, gpu_buffer in gpu_module.named_buffers():
#                 gpu_buffer.data = gpu_buffer.data.cuda()
    
#     # Fix hyperprior decoder and entropy models
#     for cpu_modules, gpu_modules in [
#         (cpu_model.h_z_s1, gpu_model.h_z_s1),
#         (cpu_model.h_z_s2, gpu_model.h_z_s2),
#         (cpu_model.entropy_bottleneck, gpu_model.entropy_bottleneck),
#         (cpu_model.gaussian_conditional, gpu_model.gaussian_conditional)
#     ]:
#         for (cpu_name, cpu_param), (gpu_name, gpu_param) in zip(cpu_modules.named_parameters(), gpu_modules.named_parameters()):
#             if gpu_param.grad is not None:
#                 cpu_param.grad = gpu_param.grad.cpu()
#             gpu_param.data = gpu_param.data.cuda()
        
#         for gpu_name, gpu_buffer in gpu_modules.named_buffers():
#             gpu_buffer.data = gpu_buffer.data.cuda()


# def configure_optimizers(compress_model, decompress_model, args):
#     """Configure optimizers for both models"""
    
#     # Compress model parameters (CPU)
#     compress_parameters = {
#         n for n, p in compress_model.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad
#     }
#     compress_aux_parameters = {
#         n for n, p in compress_model.named_parameters()
#         if n.endswith(".quantiles") and p.requires_grad
#     }
    
#     # Decompress model parameters (GPU)
#     decompress_parameters = {
#         n for n, p in decompress_model.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad
#     }
#     decompress_aux_parameters = {
#         n for n, p in decompress_model.named_parameters()
#         if n.endswith(".quantiles") and p.requires_grad
#     }

#     compress_params_dict = dict(compress_model.named_parameters())
#     decompress_params_dict = dict(decompress_model.named_parameters())

#     # Main optimizers
#     compress_optimizer = optim.Adam(
#         (compress_params_dict[n] for n in sorted(compress_parameters)),
#         lr=args.learning_rate,
#     )
#     compress_aux_optimizer = optim.Adam(
#         (compress_params_dict[n] for n in sorted(compress_aux_parameters)),
#         lr=args.aux_learning_rate,
#     )
    
#     decompress_optimizer = optim.Adam(
#         (decompress_params_dict[n] for n in sorted(decompress_parameters)),
#         lr=args.learning_rate,
#     )
#     decompress_aux_optimizer = optim.Adam(
#         (decompress_params_dict[n] for n in sorted(decompress_aux_parameters)),
#         lr=args.aux_learning_rate,
#     )
    
#     return compress_optimizer, compress_aux_optimizer, decompress_optimizer, decompress_aux_optimizer

# def train_one_epoch(
#     compress_model, decompress_model, criterion, train_dataloader, 
#     compress_optimizer, compress_aux_optimizer, decompress_optimizer, decompress_aux_optimizer,
#     epoch, clip_max_norm, train_sampler, type='mse', args=None, lr_schedulers=None
# ):
#     compress_model.train()
#     decompress_model.train()

#     if torch.cuda.device_count() > 1 and train_sampler is not None:
#         train_sampler.set_epoch(epoch)

#     pre_time = 0
#     now_time = time.time()
    
#     for i, d in enumerate(train_dataloader):
#         if (i + 1) % 10 == 0:
#             ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)
#             # Ensure decompress model stays on GPU after sync
#             for param in decompress_model.parameters():
#                 if param.device != torch.device('cuda:0'):
#                     param.data = param.data.cuda()
#         d_cpu = d  # Keep input on CPU
#         d_gpu = d.cuda()  # Copy to GPU for loss computation
        
#         # Zero gradients
#         compress_optimizer.zero_grad()
#         compress_aux_optimizer.zero_grad()
#         decompress_optimizer.zero_grad()
#         decompress_aux_optimizer.zero_grad()
        
#         # Forward pass through compression (CPU)
#         compress_out = compress_model(d_cpu)
#         print(f'compress computed successfully')
        
#         # Transfer to GPU for decompression
#         y_hat_gpu = compress_out["y_hat"].cuda().requires_grad_(True)
#         z_hat_gpu = compress_out["z_hat"].cuda().requires_grad_(True)
        
#         # Forward pass through decompression (GPU)
#         decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)
#         print(f'decompress computed successfully')

        
#         # Combine outputs for loss computation
#         combined_out = {
#             "x_hat": decompress_out["x_hat"],
#             "likelihoods": compress_out["likelihoods"]
#         }
        
#         # Compute loss on GPU
#         out_criterion = criterion(combined_out, d_gpu)
#         out_criterion["loss"].backward()

#         # Sync gradients from GPU to CPU for shared parameters
#         sync_gradients_cpu_gpu(compress_model, decompress_model)

#         print(f'synchronized gradient successfully')
        

#         if clip_max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(compress_model.parameters(), clip_max_norm)
#             torch.nn.utils.clip_grad_norm_(decompress_model.parameters(), clip_max_norm)
        
#         # Update parameters
#         compress_optimizer.step()
#         decompress_optimizer.step()
        
#         # Auxiliary loss
#         compress_aux_loss = compress_model.aux_loss()
#         decompress_aux_loss = decompress_model.aux_loss()
        
#         compress_aux_loss.backward()
#         decompress_aux_loss.backward()
        
#         compress_aux_optimizer.step()
#         decompress_aux_optimizer.step()

#         # Sync parameters every few steps
#         if (i + 1) % 10 == 0:
#             ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#         if (i+1) % 10 == 0:
#             pre_time = now_time
#             now_time = time.time()
#             print(f'time : {now_time-pre_time}\n', end='')
#             if lr_schedulers:
#                 print(f'lr : {lr_schedulers[0].get_last_lr()[0]}\n', end='')
            
#             total_aux_loss = compress_aux_loss.item() + decompress_aux_loss.item()
            
#             if type == 'mse':
#                 print(
#                     f"Train epoch {epoch}: ["
#                     f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
#                     f'\tLoss: {out_criterion["loss"].item():.3f} |'
#                     f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
#                     f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
#                     f"\tAux loss: {total_aux_loss:.2f}"
#                 )
#             else:
#                 print(
#                     f"Train epoch {epoch}: ["
#                     f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
#                     f'\tLoss: {out_criterion["loss"].item():.3f} |'
#                     f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
#                     f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
#                     f"\tAux loss: {total_aux_loss:.2f}, {compress_aux_loss.item()}, {decompress_aux_loss.item()}"
#                 )

# def test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, type='mse', args=None):
#     compress_model.eval()
#     decompress_model.eval()
    
#     if type == 'mse':
#         loss = AverageMeter()
#         bpp_loss = AverageMeter()
#         mse_loss = AverageMeter()
#         aux_loss = AverageMeter()

#         with torch.no_grad():
#             for d in test_dataloader:
#                 d_cpu = d
#                 d_gpu = d.cuda()
                
#                 # Forward pass
#                 compress_out = compress_model(d_cpu)
#                 y_hat_gpu = compress_out["y_hat"].cuda()
#                 z_hat_gpu = compress_out["z_hat"].cuda()
#                 decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)
                
#                 combined_out = {
#                     "x_hat": decompress_out["x_hat"],
#                     "likelihoods": compress_out["likelihoods"]
#                 }
                
#                 out_criterion = criterion(combined_out, d_gpu)
#                 loss.update(out_criterion["loss"])
#                 total_aux = compress_model.aux_loss() + decompress_model.aux_loss()
#                 aux_loss.update(total_aux)
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
#                 d_cpu = d
#                 d_gpu = d.cuda()
                
#                 compress_out = compress_model(d_cpu)
#                 y_hat_gpu = compress_out["y_hat"].cuda()
#                 z_hat_gpu = compress_out["z_hat"].cuda()
#                 decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)
                
#                 combined_out = {
#                     "x_hat": decompress_out["x_hat"],
#                     "likelihoods": compress_out["likelihoods"]
#                 }
                
#                 out_criterion = criterion(combined_out, d_gpu)
#                 total_aux = compress_model.aux_loss() + decompress_model.aux_loss()
#                 aux_loss.update(total_aux)
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

# def save_checkpoint(compress_state, decompress_state, is_best, epoch, save_path, filename):
#     '''Save checkpoint for both models'''
#     checkpoint_data = {
#         'compress_model': compress_state,
#         'decompress_model': decompress_state,
#         'epoch': epoch,
#         'is_best': is_best
#     }
    
#     torch.save(checkpoint_data, save_path + "checkpoint_latest.pth.tar")
#     if epoch % 5 == 0:
#         torch.save(checkpoint_data, filename)
#     if is_best:
#         torch.save(checkpoint_data, save_path + "checkpoint_best.pth.tar")

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="CPU-GPU split training script.")

#     parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
#     parser.add_argument(
#         "-d", "--dataset", type=str, default='./dataset', help="Training dataset"
#     )
#     parser.add_argument(
#         "-e", "--epochs", default=200, type=int,
#         help="Number of epochs (default: %(default)s)",
#     )
#     parser.add_argument(
#         "-lr", "--learning-rate", default=1e-4, type=float,
#         help="Learning rate (default: %(default)s)",
#     )
#     parser.add_argument(
#         "-n", "--num-workers", type=int, default=20,
#         help="Dataloaders threads (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--lambda", dest="lmbda", type=float, default=60.5,
#         help="Bit-rate distortion parameter (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
#     )
#     parser.add_argument(
#         "--test-batch-size", type=int, default=8,
#         help="Test batch size (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--aux-learning-rate", default=1e-3,
#         help="Auxiliary loss learning rate (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--patch-size", type=int, nargs=2, default=(256, 256),
#         help="Size of the patches to be cropped (default: %(default)s)",
#     )
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--save", action="store_true", default=True, help="Save model to disk"
#     )
#     parser.add_argument(
#         "--seed", type=float, default=100, help="Set random seed for reproducibility"
#     )
#     parser.add_argument(
#         "--clip_max_norm", default=1.0, type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
#     parser.add_argument("--type", type=str, default='ms-ssim', help="loss type", choices=['mse', "ms-ssim", "l1"])
#     parser.add_argument("--save_path", type=str, help="save_path", default='./checkpoints/train_5/try_2')
#     parser.add_argument("--N", type=int, default=192)
#     parser.add_argument("--M", type=int, default=320)
#     parser.add_argument("--lr_epoch", nargs='+', type=int)
#     parser.add_argument("--continue_train", action="store_true", default=True)
#     args = parser.parse_args(argv)
#     return args

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


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
#         # torch.manual_seed(args.seed)
#         # random.seed(args.seed)
#         set_seed(args.seed)
        
#     writer = SummaryWriter(save_path + "tensorboard/")

#     train_transforms = transforms.Compose(
#         [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
#     )
#     test_transforms = transforms.Compose(
#         [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
#     )

#     train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
#     test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

#     # Initialize models
#     compress_model = CompressModel(N=args.N, M=args.M)  # Keep on CPU
#     decompress_model = DecompressModel(N=args.N, M=args.M).cuda()  # Move to GPU

#     # Initial parameter synchronization
#     ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)
    
#     # ENSURE ALL DECOMPRESS MODEL PARAMETERS ARE ON GPU
#     decompress_model = decompress_model.cuda()
#     for param in decompress_model.parameters():
#         if param.device != torch.device('cuda:0'):
#             param.data = param.data.cuda()
    
#     # Also ensure buffers are on GPU
#     for buffer in decompress_model.buffers():
#         if buffer.device != torch.device('cuda:0'):
#             buffer.data = buffer.data.cuda()

#     # Update entropy models
#     compress_model.update()
#     decompress_model.update()

#     # Setup data loaders (simplified for single GPU case)
#     train_dataloader = DataLoader(
#         train_dataset,
#         shuffle=True,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )

#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.test_batch_size,
#         num_workers=args.num_workers,
#         shuffle=False,
#         pin_memory=True,
#     )

#     # Configure optimizers for both models
#     compress_optimizer, compress_aux_optimizer, decompress_optimizer, decompress_aux_optimizer = configure_optimizers(
#         compress_model, decompress_model, args
#     )

#     milestones = args.lr_epoch if args.lr_epoch else [30, 40]
#     print("milestones: ", milestones)

#     # Learning rate schedulers
#     compress_lr_scheduler = optim.lr_scheduler.MultiStepLR(compress_optimizer, milestones, gamma=0.1, last_epoch=-1)
#     decompress_lr_scheduler = optim.lr_scheduler.MultiStepLR(decompress_optimizer, milestones, gamma=0.1, last_epoch=-1)

#     criterion = RateDistortionLoss(lmbda=args.lmbda, type=type).cuda()
#     last_epoch = 0

#     # Load checkpoint if provided
#     if args.checkpoint:
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
#         if 'compress_model' in checkpoint:
#             # New format with separate models
#             compress_model.load_state_dict(checkpoint['compress_model']['state_dict'])
#             decompress_model.load_state_dict(checkpoint['decompress_model']['state_dict'])
            
#             if args.continue_train:
#                 last_epoch = checkpoint['epoch'] + 1
#                 compress_optimizer.load_state_dict(checkpoint['compress_model']['optimizer'])
#                 compress_aux_optimizer.load_state_dict(checkpoint['compress_model']['aux_optimizer'])
#                 decompress_optimizer.load_state_dict(checkpoint['decompress_model']['optimizer'])
#                 decompress_aux_optimizer.load_state_dict(checkpoint['decompress_model']['aux_optimizer'])
#                 compress_lr_scheduler.load_state_dict(checkpoint['compress_model']['lr_scheduler'])
#                 decompress_lr_scheduler.load_state_dict(checkpoint['decompress_model']['lr_scheduler'])
#         else:
#             # Legacy format - try to load shared parameters
#             print("Loading from legacy checkpoint format")
#             # You may need to adapt this based on your checkpoint structure

#     best_loss = float("inf")
#     for epoch in range(last_epoch, args.epochs):
#         train_one_epoch(
#             compress_model,
#             decompress_model,
#             criterion,
#             train_dataloader,
#             compress_optimizer,
#             compress_aux_optimizer,
#             decompress_optimizer,
#             decompress_aux_optimizer,
#             epoch,
#             args.clip_max_norm,
#             None,  # train_sampler
#             type,
#             args,
#             [compress_lr_scheduler, decompress_lr_scheduler]
#         )

#         loss = test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, type, args)
#         writer.add_scalar('test_loss', loss, epoch)

#         compress_lr_scheduler.step()
#         decompress_lr_scheduler.step()

#         is_best = loss < best_loss
#         best_loss = min(loss, best_loss)

#         if args.save:
#             compress_state = {
#                 "epoch": epoch,
#                 "state_dict": compress_model.state_dict(),
#                 "loss": loss,
#                 "optimizer": compress_optimizer.state_dict(),
#                 "aux_optimizer": compress_aux_optimizer.state_dict(),
#                 "lr_scheduler": compress_lr_scheduler.state_dict(),
#             }
            
#             decompress_state = {
#                 "epoch": epoch,
#                 "state_dict": decompress_model.state_dict(),
#                 "loss": loss,
#                 "optimizer": decompress_optimizer.state_dict(),
#                 "aux_optimizer": decompress_aux_optimizer.state_dict(),
#                 "lr_scheduler": decompress_lr_scheduler.state_dict(),
#             }
            
#             save_checkpoint(
#                 compress_state,
#                 decompress_state,
#                 is_best,
#                 epoch,
#                 save_path,
#                 save_path + str(epoch) + "_checkpoint.pth.tar",
#             )
            
#             # Also save shared parameters for deployment
#             ParameterSync.save_shared_parameters(compress_model, save_path + "shared_params.pth")

# if __name__ == "__main__":
#     main(sys.argv[1:])




# ## Train 2

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

# # Import your models
# from models import (
#     CompressModel,
#     DecompressModel,
#     ParameterSync
# )
# from torch.utils.tensorboard import SummaryWriter   
# import os
# torch.set_num_threads(8)
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False

# import numpy as np

# def test_compute_psnr(a, b):
#     b = b.to(a.device)
#     mse = torch.mean((a - b)**2).item()
#     return -10 * math.log10(mse)

# def test_compute_msssim(a, b):
#     b = b.to(a.device)
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

# def compute_msssim(a, b):
#     return ms_ssim(a, b, data_range=1.)

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

# def sync_gradients_cpu_gpu(cpu_model, gpu_model):
#     """Sync gradients for shared parameters from GPU to CPU"""
#     # Dictionary gradients
#     if gpu_model.dt.grad is not None:
#         cpu_model.dt.grad = gpu_model.dt.grad.cpu()
    
#     # Cross-attention gradients and ensure GPU parameters stay on GPU
#     for i, (cpu_module, gpu_module) in enumerate(zip(cpu_model.dt_cross_attention, gpu_model.dt_cross_attention)):
#         for (cpu_name, cpu_param), (gpu_name, gpu_param) in zip(cpu_module.named_parameters(), gpu_module.named_parameters()):
#             if gpu_param.grad is not None:
#                 cpu_param.grad = gpu_param.grad.cpu()
#             # Ensure GPU parameters stay on GPU
#             gpu_param.data = gpu_param.data.cuda()
        
#         # Ensure GPU buffers stay on GPU (including layer norm stats)
#         for gpu_name, gpu_buffer in gpu_module.named_buffers():
#             gpu_buffer.data = gpu_buffer.data.cuda()
    
#     # Apply same pattern to other shared components
#     for cpu_modules, gpu_modules in [
#         (cpu_model.cc_mean_transforms, gpu_model.cc_mean_transforms),
#         (cpu_model.cc_scale_transforms, gpu_model.cc_scale_transforms),
#         (cpu_model.lrp_transforms, gpu_model.lrp_transforms)
#     ]:
#         for cpu_module, gpu_module in zip(cpu_modules, gpu_modules):
#             for (cpu_name, cpu_param), (gpu_name, gpu_param) in zip(cpu_module.named_parameters(), gpu_module.named_parameters()):
#                 if gpu_param.grad is not None:
#                     cpu_param.grad = gpu_param.grad.cpu()
#                 gpu_param.data = gpu_param.data.cuda()
            
#             for gpu_name, gpu_buffer in gpu_module.named_buffers():
#                 gpu_buffer.data = gpu_buffer.data.cuda()
    
#     # Fix hyperprior decoder and entropy models
#     for cpu_modules, gpu_modules in [
#         (cpu_model.h_z_s1, gpu_model.h_z_s1),
#         (cpu_model.h_z_s2, gpu_model.h_z_s2),
#         (cpu_model.entropy_bottleneck, gpu_model.entropy_bottleneck),
#         (cpu_model.gaussian_conditional, gpu_model.gaussian_conditional)
#     ]:
#         for (cpu_name, cpu_param), (gpu_name, gpu_param) in zip(cpu_modules.named_parameters(), gpu_modules.named_parameters()):
#             if gpu_param.grad is not None:
#                 cpu_param.grad = gpu_param.grad.cpu()
#             gpu_param.data = gpu_param.data.cuda()
        
#         for gpu_name, gpu_buffer in gpu_modules.named_buffers():
#             gpu_buffer.data = gpu_buffer.data.cuda()


# def configure_optimizers(compress_model, decompress_model, args):
#     """Configure optimizers for both models"""
    
#     # Compress model parameters (CPU)
#     compress_parameters = {
#         n for n, p in compress_model.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad
#     }
#     compress_aux_parameters = {
#         n for n, p in compress_model.named_parameters()
#         if n.endswith(".quantiles") and p.requires_grad
#     }
    
#     # Decompress model parameters (GPU)
#     decompress_parameters = {
#         n for n, p in decompress_model.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad
#     }
#     decompress_aux_parameters = {
#         n for n, p in decompress_model.named_parameters()
#         if n.endswith(".quantiles") and p.requires_grad
#     }

#     compress_params_dict = dict(compress_model.named_parameters())
#     decompress_params_dict = dict(decompress_model.named_parameters())

#     # Main optimizers
#     compress_optimizer = optim.Adam(
#         (compress_params_dict[n] for n in sorted(compress_parameters)),
#         lr=args.learning_rate,
#     )
#     compress_aux_optimizer = optim.Adam(
#         (compress_params_dict[n] for n in sorted(compress_aux_parameters)),
#         lr=args.aux_learning_rate,
#     )
    
#     decompress_optimizer = optim.Adam(
#         (decompress_params_dict[n] for n in sorted(decompress_parameters)),
#         lr=args.learning_rate,
#     )
#     # decompress_aux_optimizer = optim.Adam(
#     #     (decompress_params_dict[n] for n in sorted(decompress_aux_parameters)),
#     #     lr=args.aux_learning_rate,
#     # )
    
#     return compress_optimizer, compress_aux_optimizer, decompress_optimizer

# def train_one_epoch(
#     compress_model, decompress_model, criterion, train_dataloader, 
#     compress_optimizer, compress_aux_optimizer, decompress_optimizer,
#     epoch, clip_max_norm, train_sampler, type='mse', args=None, lr_schedulers=None
# ):
#     compress_model.train()
#     decompress_model.train()

#     if torch.cuda.device_count() > 1 and train_sampler is not None:
#         train_sampler.set_epoch(epoch)

#     pre_time = 0
#     now_time = time.time()
    
#     for i, d in enumerate(train_dataloader):
#         if (i + 1) % 10 == 0:
#             ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)
#             # Ensure decompress model stays on GPU after sync
#             for param in decompress_model.parameters():
#                 if param.device != torch.device('cuda:0'):
#                     param.data = param.data.cuda()
#         d_cpu = d  # Keep input on CPU
#         d_gpu = d.cuda()  # Copy to GPU for loss computation
        
#         # Zero gradients
#         compress_optimizer.zero_grad()
#         compress_aux_optimizer.zero_grad()
#         decompress_optimizer.zero_grad()
#         # decompress_aux_optimizer.zero_grad()
        
#         # Forward pass through compression (CPU)
#         compress_out = compress_model(d_cpu)
#         print(f'compress computed successfully')
        
#         # Transfer to GPU for decompression
#         y_hat_gpu = compress_out["y_hat"].cuda().requires_grad_(True)
#         z_hat_gpu = compress_out["z_hat"].cuda().requires_grad_(True)
        
#         # Forward pass through decompression (GPU)
#         decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)
#         print(f'decompress computed successfully')

        
#         # Combine outputs for loss computation
#         combined_out = {
#             "x_hat": decompress_out["x_hat"],
#             "likelihoods": compress_out["likelihoods"]
#         }
        
#         # Compute loss on GPU
#         out_criterion = criterion(combined_out, d_gpu)
#         out_criterion["loss"].backward()

#         # Sync gradients from GPU to CPU for shared parameters
#         sync_gradients_cpu_gpu(compress_model, decompress_model)

#         print(f'synchronized gradient successfully')
        

#         if clip_max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(compress_model.parameters(), clip_max_norm)
#             torch.nn.utils.clip_grad_norm_(decompress_model.parameters(), clip_max_norm)
        
#         # Update parameters
#         compress_optimizer.step()
#         decompress_optimizer.step()
        
#         # Auxiliary loss
#         compress_aux_loss = compress_model.aux_loss()
#         # decompress_aux_loss = decompress_model.aux_loss()
        
#         compress_aux_loss.backward()
#         # decompress_aux_loss.backward()
        
#         compress_aux_optimizer.step()
#         # decompress_aux_optimizer.step()

#         # Sync parameters every few steps
#         # if (i + 1) % 10 == 0:
#         #     ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)
#         ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#         if (i+1) % 10 == 0:
#             pre_time = now_time
#             now_time = time.time()
#             print(f'time : {now_time-pre_time}\n', end='')
#             if lr_schedulers:
#                 print(f'lr : {lr_schedulers[0].get_last_lr()[0]}\n', end='')
            
#             total_aux_loss = compress_aux_loss.item() 
#             # + decompress_aux_loss.item()
            
#             if type == 'mse':
#                 print(
#                     f"Train epoch {epoch}: ["
#                     f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
#                     f'\tLoss: {out_criterion["loss"].item():.3f} |'
#                     f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
#                     f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
#                     f"\tAux loss: {total_aux_loss:.2f}"
#                 )
#             else:
#                 print(
#                     f"Train epoch {epoch}: ["
#                     f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}]"
#                     f'\tLoss: {out_criterion["loss"].item():.3f} |'
#                     f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
#                     f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
#                     f"\tAux loss: {total_aux_loss:.2f}"
#                 )

# def test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, type='mse', args=None):
#     compress_model.eval()
#     decompress_model.eval()
    
#     if type == 'mse':
#         loss = AverageMeter()
#         bpp_loss = AverageMeter()
#         mse_loss = AverageMeter()
#         aux_loss = AverageMeter()

#         with torch.no_grad():
#             for d in test_dataloader:
#                 d_cpu = d
#                 d_gpu = d.cuda()
                
#                 # Forward pass
#                 compress_out = compress_model(d_cpu)
#                 y_hat_gpu = compress_out["y_hat"].cuda()
#                 z_hat_gpu = compress_out["z_hat"].cuda()
#                 decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)
                
#                 combined_out = {
#                     "x_hat": decompress_out["x_hat"],
#                     "likelihoods": compress_out["likelihoods"]
#                 }
                
#                 out_criterion = criterion(combined_out, d_gpu)
#                 loss.update(out_criterion["loss"])
#                 total_aux = compress_model.aux_loss() 
#                 aux_loss.update(total_aux)
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
#                 d_cpu = d
#                 d_gpu = d.cuda()
                
#                 compress_out = compress_model(d_cpu)
#                 y_hat_gpu = compress_out["y_hat"].cuda()
#                 z_hat_gpu = compress_out["z_hat"].cuda()
#                 decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)
                
#                 combined_out = {
#                     "x_hat": decompress_out["x_hat"],
#                     "likelihoods": compress_out["likelihoods"]
#                 }
                
#                 out_criterion = criterion(combined_out, d_gpu)
#                 total_aux = compress_model.aux_loss()
#                 aux_loss.update(total_aux)
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

# def save_checkpoint(compress_state, decompress_state, is_best, epoch, save_path, filename):
#     '''Save checkpoint for both models'''
#     checkpoint_data = {
#         'compress_model': compress_state,
#         'decompress_model': decompress_state,
#         'epoch': epoch,
#         'is_best': is_best
#     }
    
#     torch.save(checkpoint_data, save_path + "checkpoint_latest.pth.tar")
#     if epoch % 5 == 0:
#         torch.save(checkpoint_data, filename)
#     if is_best:
#         torch.save(checkpoint_data, save_path + "checkpoint_best.pth.tar")

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="CPU-GPU split training script.")

#     parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
#     parser.add_argument(
#         "-d", "--dataset", type=str, default='./dataset', help="Training dataset"
#     )
#     parser.add_argument(
#         "-e", "--epochs", default=200, type=int,
#         help="Number of epochs (default: %(default)s)",
#     )
#     parser.add_argument(
#         "-lr", "--learning-rate", default=1e-4, type=float,
#         help="Learning rate (default: %(default)s)",
#     )
#     parser.add_argument(
#         "-n", "--num-workers", type=int, default=20,
#         help="Dataloaders threads (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--lambda", dest="lmbda", type=float, default=60.5,
#         help="Bit-rate distortion parameter (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
#     )
#     parser.add_argument(
#         "--test-batch-size", type=int, default=8,
#         help="Test batch size (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--aux-learning-rate", default=1e-3,
#         help="Auxiliary loss learning rate (default: %(default)s)",
#     )
#     parser.add_argument(
#         "--patch-size", type=int, nargs=2, default=(256, 256),
#         help="Size of the patches to be cropped (default: %(default)s)",
#     )
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--save", action="store_true", default=True, help="Save model to disk"
#     )
#     parser.add_argument(
#         "--seed", type=float, default=100, help="Set random seed for reproducibility"
#     )
#     parser.add_argument(
#         "--clip_max_norm", default=1.0, type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
#     parser.add_argument("--type", type=str, default='ms-ssim', help="loss type", choices=['mse', "ms-ssim", "l1"])
#     parser.add_argument("--save_path", type=str, help="save_path", default='./checkpoints/train_5/try_2')
#     parser.add_argument("--N", type=int, default=192)
#     parser.add_argument("--M", type=int, default=320)
#     parser.add_argument("--lr_epoch", nargs='+', type=int)
#     parser.add_argument("--continue_train", action="store_true", default=True)
#     args = parser.parse_args(argv)
#     return args

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


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
#         # torch.manual_seed(args.seed)
#         # random.seed(args.seed)
#         set_seed(args.seed)
        
#     writer = SummaryWriter(save_path + "tensorboard/")

#     train_transforms = transforms.Compose(
#         [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
#     )
#     test_transforms = transforms.Compose(
#         [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
#     )

#     train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
#     test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

#     # Initialize models
#     compress_model = CompressModel(N=args.N, M=args.M)  # Keep on CPU
#     decompress_model = DecompressModel(N=args.N, M=args.M).cuda()  # Move to GPU

#     # Initial parameter synchronization
#     ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)
    
#     # ENSURE ALL DECOMPRESS MODEL PARAMETERS ARE ON GPU
#     decompress_model = decompress_model.cuda()
#     for param in decompress_model.parameters():
#         if param.device != torch.device('cuda:0'):
#             param.data = param.data.cuda()
    
#     # Also ensure buffers are on GPU
#     for buffer in decompress_model.buffers():
#         if buffer.device != torch.device('cuda:0'):
#             buffer.data = buffer.data.cuda()

#     # Update entropy models
#     compress_model.update()
#     decompress_model.update()

#     # Setup data loaders (simplified for single GPU case)
#     train_dataloader = DataLoader(
#         train_dataset,
#         shuffle=True,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )

#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.test_batch_size,
#         num_workers=args.num_workers,
#         shuffle=False,
#         pin_memory=True,
#     )

#     # Configure optimizers for both models
#     compress_optimizer, compress_aux_optimizer, decompress_optimizer = configure_optimizers(
#         compress_model, decompress_model, args
#     )

#     milestones = args.lr_epoch if args.lr_epoch else [30, 40]
#     print("milestones: ", milestones)

#     # Learning rate schedulers
#     compress_lr_scheduler = optim.lr_scheduler.MultiStepLR(compress_optimizer, milestones, gamma=0.1, last_epoch=-1)
#     decompress_lr_scheduler = optim.lr_scheduler.MultiStepLR(decompress_optimizer, milestones, gamma=0.1, last_epoch=-1)

#     criterion = RateDistortionLoss(lmbda=args.lmbda, type=type).cuda()
#     last_epoch = 0

#     # Load checkpoint if provided
#     if args.checkpoint:
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
#         if 'compress_model' in checkpoint:
#             # New format with separate models
#             compress_model.load_state_dict(checkpoint['compress_model']['state_dict'])
#             decompress_model.load_state_dict(checkpoint['decompress_model']['state_dict'])
            
#             if args.continue_train:
#                 last_epoch = checkpoint['epoch'] + 1
#                 compress_optimizer.load_state_dict(checkpoint['compress_model']['optimizer'])
#                 compress_aux_optimizer.load_state_dict(checkpoint['compress_model']['aux_optimizer'])
#                 decompress_optimizer.load_state_dict(checkpoint['decompress_model']['optimizer'])
#                 # decompress_aux_optimizer.load_state_dict(checkpoint['decompress_model']['aux_optimizer'])
#                 compress_lr_scheduler.load_state_dict(checkpoint['compress_model']['lr_scheduler'])
#                 decompress_lr_scheduler.load_state_dict(checkpoint['decompress_model']['lr_scheduler'])
#         else:
#             # Legacy format - try to load shared parameters
#             print("Loading from legacy checkpoint format")
#             # You may need to adapt this based on your checkpoint structure

#     best_loss = float("inf")
#     for epoch in range(last_epoch, args.epochs):
#         train_one_epoch(
#             compress_model,
#             decompress_model,
#             criterion,
#             train_dataloader,
#             compress_optimizer,
#             compress_aux_optimizer,
#             decompress_optimizer,
#             # decompress_aux_optimizer,
#             epoch,
#             args.clip_max_norm,
#             None,  # train_sampler
#             type,
#             args,
#             [compress_lr_scheduler, decompress_lr_scheduler]
#         )

#         loss = test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, type, args)
#         writer.add_scalar('test_loss', loss, epoch)

#         compress_lr_scheduler.step()
#         decompress_lr_scheduler.step()

#         is_best = loss < best_loss
#         best_loss = min(loss, best_loss)

#         if args.save:
#             compress_state = {
#                 "epoch": epoch,
#                 "state_dict": compress_model.state_dict(),
#                 "loss": loss,
#                 "optimizer": compress_optimizer.state_dict(),
#                 "aux_optimizer": compress_aux_optimizer.state_dict(),
#                 "lr_scheduler": compress_lr_scheduler.state_dict(),
#             }
            
#             decompress_state = {
#                 "epoch": epoch,
#                 "state_dict": decompress_model.state_dict(),
#                 "loss": loss,
#                 "optimizer": decompress_optimizer.state_dict(),
#                 # "aux_optimizer": decompress_aux_optimizer.state_dict(),
#                 "lr_scheduler": decompress_lr_scheduler.state_dict(),
#             }
            
#             save_checkpoint(
#                 compress_state,
#                 decompress_state,
#                 is_best,
#                 epoch,
#                 save_path,
#                 save_path + str(epoch) + "_checkpoint.pth.tar",
#             )
            
#             # Also save shared parameters for deployment
#             ParameterSync.save_shared_parameters(compress_model, save_path + "shared_params.pth")

# if __name__ == "__main__":
#     main(sys.argv[1:])


# ## Train 3

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

# from torch.utils.data import DataLoader
# from torchvision import transforms

# from compressai.datasets import ImageFolder
# from pytorch_msssim import ms_ssim

# # Import your models
# from models import (
#     CompressModel,
#     DecompressModel,
#     ParameterSync
# )
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np

# torch.set_num_threads(8)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# def compute_msssim(a, b):
#     return ms_ssim(a, b, data_range=1.)

# class RateDistortionLoss(nn.Module):
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
#             out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
#             out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

#         return out

# class AverageMeter:
#     def __init__(self):
#         self.val = 0.0
#         self.avg = 0.0
#         self.sum = 0.0
#         self.count = 0

#     def update(self, val, n=1):
#         v = val.item() if torch.is_tensor(val) else float(val)
#         self.val = v
#         self.sum += v * n
#         self.count += n
#         self.avg = self.sum / max(1, self.count)

# def sync_gradients_cpu_gpu(cpu_model, gpu_model):
#     """Copy grads from GPU shared modules to matching CPU modules (owner)."""
#     # dt
#     if gpu_model.dt.grad is not None:
#         cpu_model.dt.grad = gpu_model.dt.grad.detach().cpu().clone()

#     # module lists
#     def copy_modlist_grads(cpu_list, gpu_list):
#         for cpu_module, gpu_module in zip(cpu_list, gpu_list):
#             for (cn, cp), (gn, gp) in zip(cpu_module.named_parameters(), gpu_module.named_parameters()):
#                 if gp.grad is not None:
#                     if cp.grad is None:
#                         cp.grad = gp.grad.detach().cpu().clone()
#                     else:
#                         cp.grad.copy_(gp.grad.detach().cpu())

#     copy_modlist_grads(cpu_model.dt_cross_attention, gpu_model.dt_cross_attention)
#     copy_modlist_grads(cpu_model.cc_mean_transforms, gpu_model.cc_mean_transforms)
#     copy_modlist_grads(cpu_model.cc_scale_transforms, gpu_model.cc_scale_transforms)
#     copy_modlist_grads(cpu_model.lrp_transforms, gpu_model.lrp_transforms)

#     # single modules
#     def copy_module_grads(cpu_m, gpu_m):
#         for (cn, cp), (gn, gp) in zip(cpu_m.named_parameters(), gpu_m.named_parameters()):
#             if gp.grad is not None:
#                 if cp.grad is None:
#                     cp.grad = gp.grad.detach().cpu().clone()
#                 else:
#                     cp.grad.copy_(gp.grad.detach().cpu())

#     copy_module_grads(cpu_model.h_z_s1, gpu_model.h_z_s1)
#     copy_module_grads(cpu_model.h_z_s2, gpu_model.h_z_s2)
#     copy_module_grads(cpu_model.entropy_bottleneck, gpu_model.entropy_bottleneck)
#     copy_module_grads(cpu_model.gaussian_conditional, gpu_model.gaussian_conditional)

# def configure_optimizers(compress_model, decompress_model, args):
#     """Configure optimizers. GPU optimizer excludes shared params."""
#     def is_shared(name):
#         shared_prefixes = [
#             "dt", "dt_cross_attention", "cc_mean_transforms",
#             "cc_scale_transforms", "lrp_transforms", "h_z_s1",
#             "h_z_s2", "entropy_bottleneck", "gaussian_conditional",
#         ]
#         return any(name == p or name.startswith(p + ".") for p in shared_prefixes)

#     # Compress (owner)
#     compress_parameters = {
#         n for n, p in compress_model.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad
#     }
#     compress_aux_parameters = {
#         n for n, p in compress_model.named_parameters()
#         if n.endswith(".quantiles") and p.requires_grad
#     }

#     # Decompress (decoder-only)
#     decompress_parameters = {
#         n for n, p in decompress_model.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad and not is_shared(n)
#     }

#     compress_params_dict = dict(compress_model.named_parameters())
#     decompress_params_dict = dict(decompress_model.named_parameters())

#     compress_optimizer = optim.Adam(
#         (compress_params_dict[n] for n in sorted(compress_parameters)),
#         lr=args.learning_rate,
#     )
#     compress_aux_optimizer = optim.Adam(
#         (compress_params_dict[n] for n in sorted(compress_aux_parameters)),
#         lr=float(args.aux_learning_rate),
#     )
#     decompress_optimizer = optim.Adam(
#         (decompress_params_dict[n] for n in sorted(decompress_parameters)),
#         lr=args.learning_rate,
#     )
#     return compress_optimizer, compress_aux_optimizer, decompress_optimizer

# def train_one_epoch(
#     compress_model, decompress_model, criterion, train_dataloader, 
#     compress_optimizer, compress_aux_optimizer, decompress_optimizer,
#     epoch, clip_max_norm, train_sampler, loss_type='mse', args=None, lr_schedulers=None
# ):
#     compress_model.train()
#     decompress_model.train()

#     for i, d in enumerate(train_dataloader):
#         d_cpu = d  # on CPU
#         d_gpu = d.cuda(non_blocking=True)

#         # occasional full sync to keep modules/buffers aligned
#         if (i + 1) % 5 == 0:
#             ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#         # zero grads
#         compress_optimizer.zero_grad(set_to_none=True)
#         decompress_optimizer.zero_grad(set_to_none=True)

#         # forward CPU (compressor/entropy)
#         compress_out = compress_model(d_cpu)

#         # move to GPU for decoder and loss
#         y_hat_gpu = compress_out["y_hat"].cuda(non_blocking=True).requires_grad_(True)
#         z_hat_gpu = compress_out["z_hat"].cuda(non_blocking=True).requires_grad_(True)
#         decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)

#         # move likelihoods to GPU for loss
#         likelihoods_gpu = {k: v.cuda(non_blocking=True) for k, v in compress_out["likelihoods"].items()}
#         combined_out = {"x_hat": decompress_out["x_hat"], "likelihoods": likelihoods_gpu}

#         # loss and backward on GPU
#         out_criterion = criterion(combined_out, d_gpu)
#         out_criterion["loss"].backward()

#         # copy grads for shared params back to gpu
#         sync_gradients_cpu_gpu(compress_model, decompress_model)

#         # clip and step
#         if clip_max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(compress_model.parameters(), clip_max_norm)
#             torch.nn.utils.clip_grad_norm_(decompress_model.parameters(), clip_max_norm)
#         compress_optimizer.step()
#         decompress_optimizer.step()

#         # aux optimizer on CPU quantiles
#         compress_aux_optimizer.zero_grad(set_to_none=True)
#         aux = compress_model.aux_loss()
#         aux.backward()
#         compress_aux_optimizer.step()

#         # periodically refresh entropy tables and resync
#         update_freq = 50
#         if (i + 1) % update_freq == 0:
#     #     if (i + 1) % 200 == 0:
#             compress_model.update()
#         ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#         # logging
#         if (i + 1) % 10 == 0:
#             aux_post = compress_model.aux_loss().item()
#             if loss_type == 'mse':
#                 print(
#                     f"Train epoch {epoch}: step {i+1}/{len(train_dataloader)} | "
#                     f"Loss: {out_criterion['loss'].item():.3f} | "
#                     f"MSE: {out_criterion['mse_loss'].item():.3f} | "
#                     f"Bpp: {out_criterion['bpp_loss'].item():.3f} | "
#                     f"Aux: {aux_post:.2f}"
#                 )
#             else:
#                 print(
#                     f"Train epoch {epoch}: step {i+1}/{len(train_dataloader)} | "
#                     f"Loss: {out_criterion['loss'].item():.3f} | "
#                     f"MS-SSIM: {out_criterion['ms_ssim_loss'].item():.4f} | "
#                     f"Bpp: {out_criterion['bpp_loss'].item():.3f} | "
#                     f"Aux: {aux_post:.2f}"
#                 )

# def test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, loss_type='mse', args=None):
#     compress_model.eval()
#     decompress_model.eval()
    
#     loss = AverageMeter()
#     bpp_loss = AverageMeter()
#     aux_meter = AverageMeter()
#     mse_loss = AverageMeter()
#     ms_ssim_loss = AverageMeter()

#     with torch.no_grad():
#         for d in test_dataloader:
#             d_cpu = d
#             d_gpu = d.cuda(non_blocking=True)
#             compress_out = compress_model(d_cpu)
#             y_hat_gpu = compress_out["y_hat"].cuda(non_blocking=True)
#             z_hat_gpu = compress_out["z_hat"].cuda(non_blocking=True)
#             decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)
#             likelihoods_gpu = {k: v.cuda(non_blocking=True) for k, v in compress_out["likelihoods"].items()}
#             combined_out = {"x_hat": decompress_out["x_hat"], "likelihoods": likelihoods_gpu}
#             out_criterion = criterion(combined_out, d_gpu)

#             loss.update(out_criterion["loss"])
#             bpp_loss.update(out_criterion["bpp_loss"])
#             aux_meter.update(compress_model.aux_loss())

#             if loss_type == 'mse':
#                 mse_loss.update(out_criterion["mse_loss"])
#             else:
#                 ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

#     if loss_type == 'mse':
#         print(
#             f"Test epoch {epoch}: "
#             f"Loss: {loss.avg:.3f} | "
#             f"MSE: {mse_loss.avg:.3f} | "
#             f"Bpp: {bpp_loss.avg:.3f} | "
#             f"Aux: {aux_meter.avg:.2f}"
#         )
#     else:
#         print(
#             f"Test epoch {epoch}: "
#             f"Loss: {loss.avg:.3f} | "
#             f"MS-SSIM: {ms_ssim_loss.avg:.4f} | "
#             f"Bpp: {bpp_loss.avg:.3f} | "
#             f"Aux: {aux_meter.avg:.2f}"
#         )

#     return loss.avg

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="CPU-GPU split training script.")
#     parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
#     parser.add_argument("-d", "--dataset", type=str, default='./dataset', help="Training dataset")
#     parser.add_argument("-e", "--epochs", default=200, type=int)
#     parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
#     parser.add_argument("-n", "--num-workers", type=int, default=8)
#     parser.add_argument("--lambda", dest="lmbda", type=float, default=60.5)
#     parser.add_argument("--batch-size", type=int, default=8)
#     parser.add_argument("--test-batch-size", type=int, default=8)
#     parser.add_argument("--aux-learning-rate", default=5e-3)
#     parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument("--save", action="store_true", default=True)
#     parser.add_argument("--seed", type=int, default=100)
#     parser.add_argument("--clip_max_norm", default=1.0, type=float)
#     parser.add_argument("--checkpoint", type=str)
#     parser.add_argument("--type", type=str, default='ms-ssim', choices=['mse', 'ms-ssim', 'l1'])
#     parser.add_argument("--save_path", type=str, default='./checkpoints/train_5/try_4')
#     parser.add_argument("--N", type=int, default=192)
#     parser.add_argument("--M", type=int, default=320)
#     parser.add_argument("--lr_epoch", nargs='+', type=int)
#     parser.add_argument("--continue_train", action="store_true", default=True)
#     args = parser.parse_args(argv)
#     return args

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def main(argv):
#     args = parse_args(argv)
#     for arg in vars(args):
#         print(arg, ":", getattr(args, arg))
    
#     loss_type = args.type
#     save_path = os.path.join(args.save_path, str(args.lmbda))
#     os.makedirs(save_path, exist_ok=True)
#     os.makedirs(os.path.join(save_path, "tensorboard"), exist_ok=True)

#     if args.seed is not None:
#         set_seed(args.seed)
        
#     writer = SummaryWriter(os.path.join(save_path, "tensorboard"))

#     train_transforms = transforms.Compose(
#         [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
#     )
#     test_transforms = transforms.Compose(
#         [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
#     )

#     train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
#     test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

#     compress_model = CompressModel(N=args.N, M=args.M)  # CPU owner
#     decompress_model = DecompressModel(N=args.N, M=args.M).cuda()  # GPU

#     # initial sync
#     ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#     # update entropy tables initially (both models)
#     compress_model.update()
#     decompress_model.update()

#     train_dataloader = DataLoader(
#         train_dataset,
#         shuffle=True,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.test_batch_size,
#         num_workers=args.num_workers,
#         shuffle=False,
#         pin_memory=True,
#     )

#     compress_optimizer, compress_aux_optimizer, decompress_optimizer = configure_optimizers(
#         compress_model, decompress_model, args
#     )

#     milestones = args.lr_epoch if args.lr_epoch else [30, 40]
#     print("milestones:", milestones)

#     compress_lr_scheduler = optim.lr_scheduler.MultiStepLR(compress_optimizer, milestones, gamma=0.1, last_epoch=-1)
#     decompress_lr_scheduler = optim.lr_scheduler.MultiStepLR(decompress_optimizer, milestones, gamma=0.1, last_epoch=-1)

#     criterion = RateDistortionLoss(lmbda=args.lmbda, type=loss_type).cuda()
#     last_epoch = 0

#     if args.checkpoint:
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location='cpu')
#         if 'compress_model' in checkpoint:
#             compress_model.load_state_dict(checkpoint['compress_model']['state_dict'])
#             decompress_model.load_state_dict(checkpoint['decompress_model']['state_dict'])
#             if args.continue_train:
#                 last_epoch = checkpoint['epoch'] + 1
#                 compress_optimizer.load_state_dict(checkpoint['compress_model']['optimizer'])
#                 compress_aux_optimizer.load_state_dict(checkpoint['compress_model']['aux_optimizer'])
#                 decompress_optimizer.load_state_dict(checkpoint['decompress_model']['optimizer'])
#                 compress_lr_scheduler.load_state_dict(checkpoint['compress_model']['lr_scheduler'])
#                 decompress_lr_scheduler.load_state_dict(checkpoint['decompress_model']['lr_scheduler'])
#         else:
#             print("Legacy checkpoint format not supported in this script.")

#     best_loss = float("inf")
#     for epoch in range(last_epoch, args.epochs):
#         train_one_epoch(
#             compress_model,
#             decompress_model,
#             criterion,
#             train_dataloader,
#             compress_optimizer,
#             compress_aux_optimizer,
#             decompress_optimizer,
#             epoch,
#             args.clip_max_norm,
#             None,
#             loss_type,
#             args,
#             [compress_lr_scheduler, decompress_lr_scheduler]
#         )

#         val_loss = test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, loss_type, args)
#         writer.add_scalar('test_loss', val_loss, epoch)

#         compress_lr_scheduler.step()
#         decompress_lr_scheduler.step()

#         is_best = val_loss < best_loss
#         best_loss = min(val_loss, best_loss)

#         if args.save:
#             compress_state = {
#                 "epoch": epoch,
#                 "state_dict": compress_model.state_dict(),
#                 "loss": val_loss,
#                 "optimizer": compress_optimizer.state_dict(),
#                 "aux_optimizer": compress_aux_optimizer.state_dict(),
#                 "lr_scheduler": compress_lr_scheduler.state_dict(),
#             }
#             decompress_state = {
#                 "epoch": epoch,
#                 "state_dict": decompress_model.state_dict(),
#                 "loss": val_loss,
#                 "optimizer": decompress_optimizer.state_dict(),
#                 "lr_scheduler": decompress_lr_scheduler.state_dict(),
#             }
#             torch.save(
#                 {
#                     'compress_model': compress_state,
#                     'decompress_model': decompress_state,
#                     'epoch': epoch,
#                     'is_best': is_best
#                 },
#                 os.path.join(save_path, "checkpoint_latest.pth.tar")
#             )
#             if epoch % 5 == 0:
#                 torch.save(
#                     {
#                         'compress_model': compress_state,
#                         'decompress_model': decompress_state,
#                         'epoch': epoch,
#                         'is_best': is_best
#                     },
#                     os.path.join(save_path, f"{epoch}_checkpoint.pth.tar")
#                 )
#             if is_best:
#                 torch.save(
#                     {
#                         'compress_model': compress_state,
#                         'decompress_model': decompress_state,
#                         'epoch': epoch,
#                         'is_best': is_best
#                     },
#                     os.path.join(save_path, "checkpoint_best.pth.tar")
#                 )
#             ParameterSync.save_shared_parameters(compress_model, os.path.join(save_path, "shared_params.pth"))

# if __name__ == "__main__":
#     main(sys.argv[1:])


# ## Train 4: based on train 3, with wandbimport os
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

# from torch.utils.data import DataLoader
# from torchvision import transforms

# from compressai.datasets import ImageFolder
# from pytorch_msssim import ms_ssim

# # Import your models
# from models import (
#     CompressModel,
#     DecompressModel,
#     ParameterSync
# )
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# import wandb  # Add wandb import

# torch.set_num_threads(8)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# def compute_msssim(a, b):
#     return ms_ssim(a, b, data_range=1.)

# class RateDistortionLoss(nn.Module):
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
#             out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
#             out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

#         return out

# class AverageMeter:
#     def __init__(self):
#         self.val = 0.0
#         self.avg = 0.0
#         self.sum = 0.0
#         self.count = 0

#     def update(self, val, n=1):
#         v = val.item() if torch.is_tensor(val) else float(val)
#         self.val = v
#         self.sum += v * n
#         self.count += n
#         self.avg = self.sum / max(1, self.count)

# def sync_gradients_cpu_gpu(cpu_model, gpu_model):
#     """Copy grads from GPU shared modules to matching CPU modules (owner)."""
#     # dt
#     if gpu_model.dt.grad is not None:
#         cpu_model.dt.grad = gpu_model.dt.grad.detach().cpu().clone()

#     # module lists
#     def copy_modlist_grads(cpu_list, gpu_list):
#         for cpu_module, gpu_module in zip(cpu_list, gpu_list):
#             for (cn, cp), (gn, gp) in zip(cpu_module.named_parameters(), gpu_module.named_parameters()):
#                 if gp.grad is not None:
#                     if cp.grad is None:
#                         cp.grad = gp.grad.detach().cpu().clone()
#                     else:
#                         cp.grad.copy_(gp.grad.detach().cpu())

#     copy_modlist_grads(cpu_model.dt_cross_attention, gpu_model.dt_cross_attention)
#     copy_modlist_grads(cpu_model.cc_mean_transforms, gpu_model.cc_mean_transforms)
#     copy_modlist_grads(cpu_model.cc_scale_transforms, gpu_model.cc_scale_transforms)
#     copy_modlist_grads(cpu_model.lrp_transforms, gpu_model.lrp_transforms)

#     # single modules
#     def copy_module_grads(cpu_m, gpu_m):
#         for (cn, cp), (gn, gp) in zip(cpu_m.named_parameters(), gpu_m.named_parameters()):
#             if gp.grad is not None:
#                 if cp.grad is None:
#                     cp.grad = gp.grad.detach().cpu().clone()
#                 else:
#                     cp.grad.copy_(gp.grad.detach().cpu())

#     copy_module_grads(cpu_model.h_z_s1, gpu_model.h_z_s1)
#     copy_module_grads(cpu_model.h_z_s2, gpu_model.h_z_s2)  
#     copy_module_grads(cpu_model.entropy_bottleneck, gpu_model.entropy_bottleneck)
#     copy_module_grads(cpu_model.gaussian_conditional, gpu_model.gaussian_conditional)

# def configure_optimizers(compress_model, decompress_model, args):
#     """Configure optimizers. GPU optimizer excludes shared params."""
#     def is_shared(name):
#         shared_prefixes = [
#             "dt", "dt_cross_attention", "cc_mean_transforms",
#             "cc_scale_transforms", "lrp_transforms", "h_z_s1",
#             "h_z_s2", "entropy_bottleneck", "gaussian_conditional",
#         ]
#         return any(name == p or name.startswith(p + ".") for p in shared_prefixes)

#     # Compress (owner)
#     compress_parameters = {
#         n for n, p in compress_model.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad
#     }
#     compress_aux_parameters = {
#         n for n, p in compress_model.named_parameters()
#         if n.endswith(".quantiles") and p.requires_grad
#     }

#     # Decompress (decoder-only)
#     decompress_parameters = {
#         n for n, p in decompress_model.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad and not is_shared(n)
#     }

#     compress_params_dict = dict(compress_model.named_parameters())
#     decompress_params_dict = dict(decompress_model.named_parameters())

#     compress_optimizer = optim.Adam(
#         (compress_params_dict[n] for n in sorted(compress_parameters)),
#         lr=args.learning_rate,
#     )
#     compress_aux_optimizer = optim.Adam(
#         (compress_params_dict[n] for n in sorted(compress_aux_parameters)),
#         lr=float(args.aux_learning_rate),
#     )
#     decompress_optimizer = optim.Adam(
#         (decompress_params_dict[n] for n in sorted(decompress_parameters)),
#         lr=args.learning_rate,
#     )
#     return compress_optimizer, compress_aux_optimizer, decompress_optimizer

# def train_one_epoch(
#     compress_model, decompress_model, criterion, train_dataloader, 
#     compress_optimizer, compress_aux_optimizer, decompress_optimizer,
#     epoch, clip_max_norm, train_sampler, loss_type='mse', args=None, lr_schedulers=None
# ):
#     compress_model.train()
#     decompress_model.train()

#     # Initialize meters for epoch-level logging
#     epoch_loss = AverageMeter()
#     epoch_bpp_loss = AverageMeter()
#     epoch_aux_loss = AverageMeter()
#     epoch_mse_loss = AverageMeter()
#     epoch_ms_ssim_loss = AverageMeter()

#     pre_time = time.time()
    
#     for i, d in enumerate(train_dataloader):
#         d_cpu = d  # on CPU
#         d_gpu = d.cuda(non_blocking=True)

#         # occasional full sync to keep modules/buffers aligned
#         if (i + 1) % 5 == 0:
#             ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#         # zero grads
#         compress_optimizer.zero_grad(set_to_none=True)
#         decompress_optimizer.zero_grad(set_to_none=True)

#         # forward CPU (compressor/entropy)
#         compress_out = compress_model(d_cpu)

#         # move to GPU for decoder and loss
#         y_hat_gpu = compress_out["y_hat"].cuda(non_blocking=True).requires_grad_(True)
#         z_hat_gpu = compress_out["z_hat"].cuda(non_blocking=True).requires_grad_(True)
#         decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)

#         # move likelihoods to GPU for loss
#         likelihoods_gpu = {k: v.cuda(non_blocking=True) for k, v in compress_out["likelihoods"].items()}
#         combined_out = {"x_hat": decompress_out["x_hat"], "likelihoods": likelihoods_gpu}

#         # loss and backward on GPU
#         out_criterion = criterion(combined_out, d_gpu)
#         out_criterion["loss"].backward()

#         # copy grads for shared params back to cpu
#         sync_gradients_cpu_gpu(compress_model, decompress_model)

#         # clip and step
#         if clip_max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(compress_model.parameters(), clip_max_norm)
#             torch.nn.utils.clip_grad_norm_(decompress_model.parameters(), clip_max_norm)
#         compress_optimizer.step()
#         decompress_optimizer.step()

#         # aux optimizer on CPU quantiles
#         compress_aux_optimizer.zero_grad(set_to_none=True)
#         aux = compress_model.aux_loss()
#         aux.backward()
#         compress_aux_optimizer.step()

#         # Update epoch meters
#         aux_post = compress_model.aux_loss().item()
#         epoch_loss.update(out_criterion["loss"])
#         epoch_bpp_loss.update(out_criterion["bpp_loss"])
#         epoch_aux_loss.update(aux_post)
        
#         if loss_type == 'mse':
#             epoch_mse_loss.update(out_criterion["mse_loss"])
#         else:
#             epoch_ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

#         # periodically refresh entropy tables and resync
#         update_freq = 50
#         if (i + 1) % update_freq == 0:
#             compress_model.update()
#         ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#         # Enhanced logging similar to code 2
#         if (i + 1) % 10 == 0:
#             now_time = time.time()
#             step_time = now_time - pre_time
#             pre_time = now_time
            
#             current_lr = lr_schedulers[0].get_last_lr()[0] if lr_schedulers else compress_optimizer.param_groups[0]['lr']
            
#             # Log step-level metrics to wandb
#             step = epoch * len(train_dataloader) + i
#             log_dict = {
#                 'train/loss': out_criterion["loss"].item(),
#                 'train/bpp_loss': out_criterion["bpp_loss"].item(),
#                 'train/aux_loss': aux_post,
#                 'train/compress_aux_loss': aux_post,  # Only compress aux in code 1
#                 'train/learning_rate': current_lr,
#                 'train/step_time': step_time,
#                 'epoch': epoch,
#                 'step': step
#             }
            
#             if loss_type == 'mse':
#                 log_dict['train/mse_loss'] = out_criterion["mse_loss"].item()
#                 print(
#                     f"Train epoch {epoch}: step {i+1}/{len(train_dataloader)} | "
#                     f"Loss: {out_criterion['loss'].item():.3f} | "
#                     f"MSE: {out_criterion['mse_loss'].item():.3f} | "
#                     f"Bpp: {out_criterion['bpp_loss'].item():.3f} | "
#                     f"Aux: {aux_post:.2f} | "
#                     f"Time: {step_time:.2f}"
#                 )
#             else:
#                 log_dict['train/ms_ssim_loss'] = out_criterion["ms_ssim_loss"].item()
#                 print(
#                     f"Train epoch {epoch}: step {i+1}/{len(train_dataloader)} | "
#                     f"Loss: {out_criterion['loss'].item():.3f} | "
#                     f"MS-SSIM: {out_criterion['ms_ssim_loss'].item():.4f} | "
#                     f"Bpp: {out_criterion['bpp_loss'].item():.3f} | "
#                     f"Aux: {aux_post:.2f} | "
#                     f"Time: {step_time:.2f}"
#                 )
            
#             wandb.log(log_dict)

#     # Log epoch-level metrics to wandb (keeping this for compatibility)
#     wandb_log = {
#         "epoch": epoch,
#         "train/epoch_loss": epoch_loss.avg,
#         "train/epoch_bpp_loss": epoch_bpp_loss.avg,
#         "train/epoch_aux_loss": epoch_aux_loss.avg,
#         "train/final_learning_rate": compress_optimizer.param_groups[0]['lr']
#     }
    
#     if loss_type == 'mse':
#         wandb_log["train/epoch_mse_loss"] = epoch_mse_loss.avg
#     else:
#         wandb_log["train/epoch_ms_ssim_loss"] = epoch_ms_ssim_loss.avg
    
#     wandb.log(wandb_log)

# def test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, loss_type='mse', args=None):
#     compress_model.eval()
#     decompress_model.eval()
    
#     loss = AverageMeter()
#     bpp_loss = AverageMeter()
#     aux_meter = AverageMeter()
#     mse_loss = AverageMeter()
#     ms_ssim_loss = AverageMeter()

#     with torch.no_grad():
#         for d in test_dataloader:
#             d_cpu = d
#             d_gpu = d.cuda(non_blocking=True)
#             compress_out = compress_model(d_cpu)
#             y_hat_gpu = compress_out["y_hat"].cuda(non_blocking=True)
#             z_hat_gpu = compress_out["z_hat"].cuda(non_blocking=True)
#             decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)
#             likelihoods_gpu = {k: v.cuda(non_blocking=True) for k, v in compress_out["likelihoods"].items()}
#             combined_out = {"x_hat": decompress_out["x_hat"], "likelihoods": likelihoods_gpu}
#             out_criterion = criterion(combined_out, d_gpu)

#             loss.update(out_criterion["loss"])
#             bpp_loss.update(out_criterion["bpp_loss"])
#             aux_meter.update(compress_model.aux_loss())

#             if loss_type == 'mse':
#                 mse_loss.update(out_criterion["mse_loss"])
#             else:
#                 ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

#     # Enhanced logging similar to code 2
#     wandb_log = {
#         "epoch": epoch,
#         "val/loss": loss.avg,
#         "val/bpp_loss": bpp_loss.avg,
#         "val/aux_loss": aux_meter.avg,
#         "val/compress_aux_loss": aux_meter.avg  # Only compress aux in code 1
#     }
    
#     if loss_type == 'mse':
#         wandb_log["val/mse_loss"] = mse_loss.avg
#         print(
#             f"Test epoch {epoch}: Average losses: "
#             f"Loss: {loss.avg:.3f} | "
#             f"MSE loss: {mse_loss.avg:.3f} | "
#             f"Bpp loss: {bpp_loss.avg:.2f} | "
#             f"Aux loss: {aux_meter.avg:.2f}"
#         )
#     else:
#         wandb_log["val/ms_ssim_loss"] = ms_ssim_loss.avg
#         print(
#             f"Test epoch {epoch}: Average losses: "
#             f"Loss: {loss.avg:.3f} | "
#             f"MS-SSIM loss: {ms_ssim_loss.avg:.4f} | "
#             f"Bpp loss: {bpp_loss.avg:.2f} | "
#             f"Aux loss: {aux_meter.avg:.2f}"
#         )
    
#     wandb.log(wandb_log)
#     return loss.avg

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="CPU-GPU split training script.")
#     parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
#     parser.add_argument("-d", "--dataset", type=str, default='./dataset', help="Training dataset")
#     parser.add_argument("-e", "--epochs", default=400, type=int)
#     parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
#     parser.add_argument("-n", "--num-workers", type=int, default=8)
#     parser.add_argument("--lambda", dest="lmbda", type=float, default=60.5)
#     parser.add_argument("--batch-size", type=int, default=8)
#     parser.add_argument("--test-batch-size", type=int, default=8)
#     parser.add_argument("--aux-learning-rate", default=5e-3)
#     parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument("--save", action="store_true", default=True)
#     parser.add_argument("--seed", type=int, default=100)
#     parser.add_argument("--clip_max_norm", default=1.0, type=float)
#     parser.add_argument("--checkpoint", type=str, default='checkpoints/train_5/try_4/60.5/checkpoint_latest.pth.tar')
#     parser.add_argument("--type", type=str, default='ms-ssim', choices=['mse', 'ms-ssim', 'l1'])
#     parser.add_argument("--save_path", type=str, default='./checkpoints/train_5/try_4')
#     parser.add_argument("--N", type=int, default=192)
#     parser.add_argument("--M", type=int, default=320)
#     parser.add_argument("--lr_epoch", nargs='+', type=int)
#     parser.add_argument("--continue_train", action="store_true", default=True)
    
#     # Enhanced wandb arguments similar to code 2
#     parser.add_argument("--wandb_project", type=str, default="Image-compression", help="WandB project name")
#     parser.add_argument("--wandb_run_name", type=str, default='train_5_try_4_20250915_01', help="WandB run name")
#     parser.add_argument("--wandb_tags", nargs='+', type=str, default=[], help="wandb tags")
    
#     args = parser.parse_args(argv)
#     return args

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def main(argv):
#     args = parse_args(argv)
#     for arg in vars(args):
#         print(arg, ":", getattr(args, arg))
    
#     loss_type = args.type
#     save_path = os.path.join(args.save_path, str(args.lmbda))
#     os.makedirs(save_path, exist_ok=True)
#     os.makedirs(os.path.join(save_path, "tensorboard"), exist_ok=True)

#     if args.seed is not None:
#         set_seed(args.seed)

#     # Enhanced wandb initialization similar to code 2
#     wandb_name = args.wandb_run_name or f"lambda_{args.lmbda}_{loss_type}_N{args.N}_M{args.M}"
#     wandb.init(
#         project=args.wandb_project,
#         name=wandb_name,
#         tags=args.wandb_tags,
#         config={
#             "epochs": args.epochs,
#             "learning_rate": args.learning_rate,
#             "aux_learning_rate": args.aux_learning_rate,
#             "batch_size": args.batch_size,
#             "test_batch_size": args.test_batch_size,
#             "lambda": args.lmbda,
#             "loss_type": loss_type,
#             "patch_size": args.patch_size,
#             "N": args.N,
#             "M": args.M,
#             "clip_max_norm": args.clip_max_norm,
#             "seed": args.seed,
#             "lr_epoch": args.lr_epoch,
#             "num_workers": args.num_workers,
#             "dataset": args.dataset,
#             "save_path": args.save_path,
#             "continue_train": args.continue_train,
#             "architecture": "CPU-GPU-Split"
#         }
#     )
        
#     writer = SummaryWriter(os.path.join(save_path, "tensorboard"))

#     train_transforms = transforms.Compose(
#         [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
#     )
#     test_transforms = transforms.Compose(
#         [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
#     )

#     train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
#     test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

#     compress_model = CompressModel(N=args.N, M=args.M)  # CPU owner
#     decompress_model = DecompressModel(N=args.N, M=args.M).cuda()  # GPU

#     # initial sync
#     ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#     # update entropy tables initially (both models)
#     compress_model.update()
#     decompress_model.update()

#     train_dataloader = DataLoader(
#         train_dataset,
#         shuffle=True,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.test_batch_size,
#         num_workers=args.num_workers,
#         shuffle=False,
#         pin_memory=True,
#     )

#     compress_optimizer, compress_aux_optimizer, decompress_optimizer = configure_optimizers(
#         compress_model, decompress_model, args
#     )

#     milestones = args.lr_epoch if args.lr_epoch else [30, 40]
#     print("milestones:", milestones)

#     compress_lr_scheduler = optim.lr_scheduler.MultiStepLR(compress_optimizer, milestones, gamma=0.1, last_epoch=-1)
#     decompress_lr_scheduler = optim.lr_scheduler.MultiStepLR(decompress_optimizer, milestones, gamma=0.1, last_epoch=-1)

#     criterion = RateDistortionLoss(lmbda=args.lmbda, type=loss_type).cuda()
#     last_epoch = 0

#     if args.checkpoint:
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location='cpu')
#         if 'compress_model' in checkpoint:
#             compress_model.load_state_dict(checkpoint['compress_model']['state_dict'])
#             decompress_model.load_state_dict(checkpoint['decompress_model']['state_dict'])
#             if args.continue_train:
#                 last_epoch = checkpoint['epoch'] + 1
#                 compress_optimizer.load_state_dict(checkpoint['compress_model']['optimizer'])
#                 compress_aux_optimizer.load_state_dict(checkpoint['compress_model']['aux_optimizer'])
#                 decompress_optimizer.load_state_dict(checkpoint['decompress_model']['optimizer'])
#                 compress_lr_scheduler.load_state_dict(checkpoint['compress_model']['lr_scheduler'])
#                 decompress_lr_scheduler.load_state_dict(checkpoint['decompress_model']['lr_scheduler'])
#         else:
#             print("Legacy checkpoint format not supported in this script.")

#     best_loss = float("inf")
#     for epoch in range(last_epoch, args.epochs):
#         train_one_epoch(
#             compress_model,
#             decompress_model,
#             criterion,
#             train_dataloader,
#             compress_optimizer,
#             compress_aux_optimizer,
#             decompress_optimizer,
#             epoch,
#             args.clip_max_norm,
#             None,
#             loss_type,
#             args,
#             [compress_lr_scheduler, decompress_lr_scheduler]
#         )

#         val_loss = test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, loss_type, args)
#         writer.add_scalar('test_loss', val_loss, epoch)

#         compress_lr_scheduler.step()
#         decompress_lr_scheduler.step()

#         is_best = val_loss < best_loss
#         best_loss = min(val_loss, best_loss)

#         if args.save:
#             compress_state = {
#                 "epoch": epoch,
#                 "state_dict": compress_model.state_dict(),
#                 "loss": val_loss,
#                 "optimizer": compress_optimizer.state_dict(),
#                 "aux_optimizer": compress_aux_optimizer.state_dict(),
#                 "lr_scheduler": compress_lr_scheduler.state_dict(),
#             }
#             decompress_state = {
#                 "epoch": epoch,
#                 "state_dict": decompress_model.state_dict(),
#                 "loss": val_loss,
#                 "optimizer": decompress_optimizer.state_dict(),
#                 "lr_scheduler": decompress_lr_scheduler.state_dict(),
#             }
#             torch.save(
#                 {
#                     'compress_model': compress_state,
#                     'decompress_model': decompress_state,
#                     'epoch': epoch,
#                     'is_best': is_best
#                 },
#                 os.path.join(save_path, "checkpoint_latest.pth.tar")
#             )
#             if epoch % 5 == 0:
#                 torch.save(
#                     {
#                         'compress_model': compress_state,
#                         'decompress_model': decompress_state,
#                         'epoch': epoch,
#                         'is_best': is_best
#                     },
#                     os.path.join(save_path, f"{epoch}_checkpoint.pth.tar")
#                 )
#             if is_best:
#                 torch.save(
#                     {
#                         'compress_model': compress_state,
#                         'decompress_model': decompress_state,
#                         'epoch': epoch,
#                         'is_best': is_best
#                     },
#                     os.path.join(save_path, "checkpoint_best.pth.tar")
#                 )
#                 # Enhanced best model logging
#                 wandb.log({
#                     "best_val_loss": val_loss, 
#                     "best_epoch": epoch,
#                     "epoch": epoch
#                 })
#             ParameterSync.save_shared_parameters(compress_model, os.path.join(save_path, "shared_params.pth"))

#     wandb.finish()

# if __name__ == "__main__":
#     main(sys.argv[1:])


# ## Train 5: add eponentila learning rate schedule
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

# from torch.utils.data import DataLoader
# from torchvision import transforms

# from compressai.datasets import ImageFolder
# from pytorch_msssim import ms_ssim

# # Import your models
# from models import (
#     CompressModel,
#     DecompressModel,
#     ParameterSync
# )
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# import wandb  # Add wandb import

# torch.set_num_threads(8)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# def compute_msssim(a, b):
#     return ms_ssim(a, b, data_range=1.)

# class ExponentialTargetScheduler:
#     def __init__(self, optimizer, start_loss=3820, target_loss=10, total_epochs=100):
#         self.optimizer = optimizer
#         self.start_loss = start_loss
#         self.target_loss = target_loss
#         self.total_epochs = total_epochs
        
#         # Calculate required decay rate
#         self.decay_rate = (target_loss / start_loss) ** (1.0 / total_epochs)
#         print(f"Target decay rate per epoch: {self.decay_rate:.6f}")
        
#     def step(self, current_aux_loss, main_lr, epoch):
#         """Exponential schedule to reach target in specified epochs"""
        
#         # Expected loss at this epoch if following ideal trajectory
#         expected_loss = self.start_loss * (self.decay_rate ** epoch)
        
#         # If we're behind schedule, boost LR more
#         if current_aux_loss > expected_loss * 1.5:
#             # We're falling behind - boost aggressively
#             boost_factor = (current_aux_loss / expected_loss) * 2
#             multiplier = min(1000, 200 * boost_factor)
#         elif current_aux_loss > expected_loss:
#             # Slightly behind schedule
#             boost_factor = current_aux_loss / expected_loss
#             multiplier = min(500, 100 * boost_factor)
#         else:
#             # On track or ahead - use standard aggressive rate
#             multiplier = max(50, 200 * (current_aux_loss / self.target_loss))
        
#         new_lr = main_lr * multiplier
#         new_lr = min(new_lr, 0.1)  # Safety cap
        
#         self.optimizer.param_groups[0]['lr'] = new_lr
        
#         print(f"Epoch {epoch}: Expected aux loss: {expected_loss:.1f}, "
#               f"Actual: {current_aux_loss:.1f}, Multiplier: {multiplier:.1f}")
        
#         return new_lr, multiplier

# class RateDistortionLoss(nn.Module):
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
#             out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
#             out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

#         return out

# class AverageMeter:
#     def __init__(self):
#         self.val = 0.0
#         self.avg = 0.0
#         self.sum = 0.0
#         self.count = 0

#     def update(self, val, n=1):
#         v = val.item() if torch.is_tensor(val) else float(val)
#         self.val = v
#         self.sum += v * n
#         self.count += n
#         self.avg = self.sum / max(1, self.count)

# def sync_gradients_cpu_gpu(cpu_model, gpu_model):
#     """Copy grads from GPU shared modules to matching CPU modules (owner)."""
#     # dt
#     if gpu_model.dt.grad is not None:
#         cpu_model.dt.grad = gpu_model.dt.grad.detach().cpu().clone()

#     # module lists
#     def copy_modlist_grads(cpu_list, gpu_list):
#         for cpu_module, gpu_module in zip(cpu_list, gpu_list):
#             for (cn, cp), (gn, gp) in zip(cpu_module.named_parameters(), gpu_module.named_parameters()):
#                 if gp.grad is not None:
#                     if cp.grad is None:
#                         cp.grad = gp.grad.detach().cpu().clone()
#                     else:
#                         cp.grad.copy_(gp.grad.detach().cpu())

#     copy_modlist_grads(cpu_model.dt_cross_attention, gpu_model.dt_cross_attention)
#     copy_modlist_grads(cpu_model.cc_mean_transforms, gpu_model.cc_mean_transforms)
#     copy_modlist_grads(cpu_model.cc_scale_transforms, gpu_model.cc_scale_transforms)
#     copy_modlist_grads(cpu_model.lrp_transforms, gpu_model.lrp_transforms)

#     # single modules
#     def copy_module_grads(cpu_m, gpu_m):
#         for (cn, cp), (gn, gp) in zip(cpu_m.named_parameters(), gpu_m.named_parameters()):
#             if gp.grad is not None:
#                 if cp.grad is None:
#                     cp.grad = gp.grad.detach().cpu().clone()
#                 else:
#                     cp.grad.copy_(gp.grad.detach().cpu())

#     copy_module_grads(cpu_model.h_z_s1, gpu_model.h_z_s1)
#     copy_module_grads(cpu_model.h_z_s2, gpu_model.h_z_s2)  
#     copy_module_grads(cpu_model.entropy_bottleneck, gpu_model.entropy_bottleneck)
#     copy_module_grads(cpu_model.gaussian_conditional, gpu_model.gaussian_conditional)

# def configure_optimizers(compress_model, decompress_model, args):
#     """Configure optimizers. GPU optimizer excludes shared params."""
#     def is_shared(name):
#         shared_prefixes = [
#             "dt", "dt_cross_attention", "cc_mean_transforms",
#             "cc_scale_transforms", "lrp_transforms", "h_z_s1",
#             "h_z_s2", "entropy_bottleneck", "gaussian_conditional",
#         ]
#         return any(name == p or name.startswith(p + ".") for p in shared_prefixes)

#     # Compress (owner)
#     compress_parameters = {
#         n for n, p in compress_model.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad
#     }
#     compress_aux_parameters = {
#         n for n, p in compress_model.named_parameters()
#         if n.endswith(".quantiles") and p.requires_grad
#     }

#     # Decompress (decoder-only)
#     decompress_parameters = {
#         n for n, p in decompress_model.named_parameters()
#         if not n.endswith(".quantiles") and p.requires_grad and not is_shared(n)
#     }

#     compress_params_dict = dict(compress_model.named_parameters())
#     decompress_params_dict = dict(decompress_model.named_parameters())

#     compress_optimizer = optim.Adam(
#         (compress_params_dict[n] for n in sorted(compress_parameters)),
#         lr=args.learning_rate,
#     )
#     compress_aux_optimizer = optim.Adam(
#         (compress_params_dict[n] for n in sorted(compress_aux_parameters)),
#         lr=float(args.aux_learning_rate),
#     )
#     decompress_optimizer = optim.Adam(
#         (decompress_params_dict[n] for n in sorted(decompress_parameters)),
#         lr=args.learning_rate,
#     )
#     return compress_optimizer, compress_aux_optimizer, decompress_optimizer

# def train_one_epoch(
#     compress_model, decompress_model, criterion, train_dataloader, 
#     compress_optimizer, compress_aux_optimizer, decompress_optimizer,
#     epoch, clip_max_norm, train_sampler, loss_type='mse', args=None, lr_schedulers=None, aux_scheduler=None
# ):
#     compress_model.train()
#     decompress_model.train()

#     # Initialize meters for epoch-level logging
#     epoch_loss = AverageMeter()
#     epoch_bpp_loss = AverageMeter()
#     epoch_aux_loss = AverageMeter()
#     epoch_mse_loss = AverageMeter()
#     epoch_ms_ssim_loss = AverageMeter()

#     pre_time = time.time()
    
#     for i, d in enumerate(train_dataloader):
#         d_cpu = d  # on CPU
#         d_gpu = d.cuda(non_blocking=True)

#         # occasional full sync to keep modules/buffers aligned
#         if (i + 1) % 5 == 0:
#             ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#         # zero grads
#         compress_optimizer.zero_grad(set_to_none=True)
#         decompress_optimizer.zero_grad(set_to_none=True)

#         # forward CPU (compressor/entropy)
#         compress_out = compress_model(d_cpu)

#         # move to GPU for decoder and loss
#         y_hat_gpu = compress_out["y_hat"].cuda(non_blocking=True).requires_grad_(True)
#         z_hat_gpu = compress_out["z_hat"].cuda(non_blocking=True).requires_grad_(True)
#         decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)

#         # move likelihoods to GPU for loss
#         likelihoods_gpu = {k: v.cuda(non_blocking=True) for k, v in compress_out["likelihoods"].items()}
#         combined_out = {"x_hat": decompress_out["x_hat"], "likelihoods": likelihoods_gpu}

#         # loss and backward on GPU
#         out_criterion = criterion(combined_out, d_gpu)
#         out_criterion["loss"].backward()

#         # copy grads for shared params back to cpu
#         sync_gradients_cpu_gpu(compress_model, decompress_model)

#         # clip and step
#         if clip_max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(compress_model.parameters(), clip_max_norm)
#             torch.nn.utils.clip_grad_norm_(decompress_model.parameters(), clip_max_norm)
#         compress_optimizer.step()
#         decompress_optimizer.step()

#         # aux optimizer on CPU quantiles
#         compress_aux_optimizer.zero_grad(set_to_none=True)
#         aux = compress_model.aux_loss()
#         aux.backward()
        
#         # Apply aux scheduler before optimizer step
#         if aux_scheduler is not None:
#             current_lr = lr_schedulers[0].get_last_lr()[0] if lr_schedulers else compress_optimizer.param_groups[0]['lr']
#             aux_lr, multiplier = aux_scheduler.step(aux.item(), current_lr, epoch)
            
#             # Log aux scheduler metrics
#             wandb.log({
#                 'train/aux_learning_rate': aux_lr,
#                 'train/aux_lr_multiplier': multiplier,
#                 'train/expected_aux_loss': aux_scheduler.start_loss * (aux_scheduler.decay_rate ** epoch),
#                 'epoch': epoch,
#                 'step': epoch * len(train_dataloader) + i
#             })
        
#         compress_aux_optimizer.step()

#         # Update epoch meters
#         aux_post = compress_model.aux_loss().item()
#         epoch_loss.update(out_criterion["loss"])
#         epoch_bpp_loss.update(out_criterion["bpp_loss"])
#         epoch_aux_loss.update(aux_post)
        
#         if loss_type == 'mse':
#             epoch_mse_loss.update(out_criterion["mse_loss"])
#         else:
#             epoch_ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

#         # periodically refresh entropy tables and resync
#         update_freq = 50
#         if (i + 1) % update_freq == 0:
#             compress_model.update()
#         ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#         # Enhanced logging similar to code 2
#         if (i + 1) % 10 == 0:
#             now_time = time.time()
#             step_time = now_time - pre_time
#             pre_time = now_time
            
#             current_lr = lr_schedulers[0].get_last_lr()[0] if lr_schedulers else compress_optimizer.param_groups[0]['lr']
            
#             # Log step-level metrics to wandb
#             step = epoch * len(train_dataloader) + i
#             log_dict = {
#                 'train/loss': out_criterion["loss"].item(),
#                 'train/bpp_loss': out_criterion["bpp_loss"].item(),
#                 'train/aux_loss': aux_post,
#                 'train/compress_aux_loss': aux_post,  # Only compress aux in code 1
#                 'train/learning_rate': current_lr,
#                 'train/step_time': step_time,
#                 'epoch': epoch,
#                 'step': step
#             }
            
#             if loss_type == 'mse':
#                 log_dict['train/mse_loss'] = out_criterion["mse_loss"].item()
#                 print(
#                     f"Train epoch {epoch}: step {i+1}/{len(train_dataloader)} | "
#                     f"Loss: {out_criterion['loss'].item():.3f} | "
#                     f"MSE: {out_criterion['mse_loss'].item():.3f} | "
#                     f"Bpp: {out_criterion['bpp_loss'].item():.3f} | "
#                     f"Aux: {aux_post:.2f} | "
#                     f"Time: {step_time:.2f}"
#                 )
#             else:
#                 log_dict['train/ms_ssim_loss'] = out_criterion["ms_ssim_loss"].item()
#                 print(
#                     f"Train epoch {epoch}: step {i+1}/{len(train_dataloader)} | "
#                     f"Loss: {out_criterion['loss'].item():.3f} | "
#                     f"MS-SSIM: {out_criterion['ms_ssim_loss'].item():.4f} | "
#                     f"Bpp: {out_criterion['bpp_loss'].item():.3f} | "
#                     f"Aux: {aux_post:.2f} | "
#                     f"Time: {step_time:.2f}"
#                 )
            
#             wandb.log(log_dict)

#     # Log epoch-level metrics to wandb (keeping this for compatibility)
#     wandb_log = {
#         "epoch": epoch,
#         "train/epoch_loss": epoch_loss.avg,
#         "train/epoch_bpp_loss": epoch_bpp_loss.avg,
#         "train/epoch_aux_loss": epoch_aux_loss.avg,
#         "train/final_learning_rate": compress_optimizer.param_groups[0]['lr']
#     }
    
#     if loss_type == 'mse':
#         wandb_log["train/epoch_mse_loss"] = epoch_mse_loss.avg
#     else:
#         wandb_log["train/epoch_ms_ssim_loss"] = epoch_ms_ssim_loss.avg
    
#     wandb.log(wandb_log)

# def test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, loss_type='mse', args=None):
#     compress_model.eval()
#     decompress_model.eval()
    
#     loss = AverageMeter()
#     bpp_loss = AverageMeter()
#     aux_meter = AverageMeter()
#     mse_loss = AverageMeter()
#     ms_ssim_loss = AverageMeter()

#     with torch.no_grad():
#         for d in test_dataloader:
#             d_cpu = d
#             d_gpu = d.cuda(non_blocking=True)
#             compress_out = compress_model(d_cpu)
#             y_hat_gpu = compress_out["y_hat"].cuda(non_blocking=True)
#             z_hat_gpu = compress_out["z_hat"].cuda(non_blocking=True)
#             decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)
#             likelihoods_gpu = {k: v.cuda(non_blocking=True) for k, v in compress_out["likelihoods"].items()}
#             combined_out = {"x_hat": decompress_out["x_hat"], "likelihoods": likelihoods_gpu}
#             out_criterion = criterion(combined_out, d_gpu)

#             loss.update(out_criterion["loss"])
#             bpp_loss.update(out_criterion["bpp_loss"])
#             aux_meter.update(compress_model.aux_loss())

#             if loss_type == 'mse':
#                 mse_loss.update(out_criterion["mse_loss"])
#             else:
#                 ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

#     # Enhanced logging similar to code 2
#     wandb_log = {
#         "epoch": epoch,
#         "val/loss": loss.avg,
#         "val/bpp_loss": bpp_loss.avg,
#         "val/aux_loss": aux_meter.avg,
#         "val/compress_aux_loss": aux_meter.avg  # Only compress aux in code 1
#     }
    
#     if loss_type == 'mse':
#         wandb_log["val/mse_loss"] = mse_loss.avg
#         print(
#             f"Test epoch {epoch}: Average losses: "
#             f"Loss: {loss.avg:.3f} | "
#             f"MSE loss: {mse_loss.avg:.3f} | "
#             f"Bpp loss: {bpp_loss.avg:.2f} | "
#             f"Aux loss: {aux_meter.avg:.2f}"
#         )
#     else:
#         wandb_log["val/ms_ssim_loss"] = ms_ssim_loss.avg
#         print(
#             f"Test epoch {epoch}: Average losses: "
#             f"Loss: {loss.avg:.3f} | "
#             f"MS-SSIM loss: {ms_ssim_loss.avg:.4f} | "
#             f"Bpp loss: {bpp_loss.avg:.2f} | "
#             f"Aux loss: {aux_meter.avg:.2f}"
#         )
    
#     wandb.log(wandb_log)
#     return loss.avg

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="CPU-GPU split training script.")
#     parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
#     parser.add_argument("-d", "--dataset", type=str, default='./dataset', help="Training dataset")
#     parser.add_argument("-e", "--epochs", default=4000, type=int)
#     parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
#     parser.add_argument("-n", "--num-workers", type=int, default=8)
#     parser.add_argument("--lambda", dest="lmbda", type=float, default=60.5)
#     parser.add_argument("--batch-size", type=int, default=8)
#     parser.add_argument("--test-batch-size", type=int, default=8)
#     parser.add_argument("--aux-learning-rate", default=5e-3)
#     parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument("--save", action="store_true", default=True)
#     parser.add_argument("--seed", type=int, default=100)
#     parser.add_argument("--clip_max_norm", default=1.0, type=float)
#     parser.add_argument("--checkpoint", type=str, default='checkpoints/train_5/try_6/60.5/checkpoint_best.pth.tar')
#     parser.add_argument("--type", type=str, default='ms-ssim', choices=['mse', 'ms-ssim', 'l1'])
#     parser.add_argument("--save_path", type=str, default='./checkpoints/train_5/try_7')
#     parser.add_argument("--N", type=int, default=192)
#     parser.add_argument("--M", type=int, default=320)
#     parser.add_argument("--lr_epoch", nargs='+', type=int)
#     parser.add_argument("--continue_train", action="store_true", default=True)
    
#     # Enhanced wandb arguments similar to code 2
#     parser.add_argument("--wandb_project", type=str, default="Image-compression", help="WandB project name")
#     parser.add_argument("--wandb_run_name", type=str, default='train_5_try6_20250916_01', help="WandB run name")
#     parser.add_argument("--wandb_tags", nargs='+', type=str, default=[], help="wandb tags")
    
#     # Aux scheduler parameters
#     parser.add_argument("--aux_start_loss", type=float, default=718.0, help="Starting aux loss for scheduler")
#     parser.add_argument("--aux_target_loss", type=float, default=10.0, help="Target aux loss for scheduler")
#     parser.add_argument("--use_aux_scheduler", action="store_true", default=True, help="Use exponential target scheduler for aux loss")
    
#     args = parser.parse_args(argv)
#     return args

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def main(argv):
#     args = parse_args(argv)
#     for arg in vars(args):
#         print(arg, ":", getattr(args, arg))
    
#     loss_type = args.type
#     save_path = os.path.join(args.save_path, str(args.lmbda))
#     os.makedirs(save_path, exist_ok=True)
#     os.makedirs(os.path.join(save_path, "tensorboard"), exist_ok=True)

#     if args.seed is not None:
#         set_seed(args.seed)

#     # Enhanced wandb initialization similar to code 2
#     wandb_name = args.wandb_run_name or f"lambda_{args.lmbda}_{loss_type}_N{args.N}_M{args.M}"
#     wandb.init(
#         project=args.wandb_project,
#         name=wandb_name,
#         tags=args.wandb_tags,
#         config={
#             "epochs": args.epochs,
#             "learning_rate": args.learning_rate,
#             "aux_learning_rate": args.aux_learning_rate,
#             "batch_size": args.batch_size,
#             "test_batch_size": args.test_batch_size,
#             "lambda": args.lmbda,
#             "loss_type": loss_type,
#             "patch_size": args.patch_size,
#             "N": args.N,
#             "M": args.M,
#             "clip_max_norm": args.clip_max_norm,
#             "seed": args.seed,
#             "lr_epoch": args.lr_epoch,
#             "num_workers": args.num_workers,
#             "dataset": args.dataset,
#             "save_path": args.save_path,
#             "continue_train": args.continue_train,
#             "architecture": "CPU-GPU-Split",
#             "use_aux_scheduler": args.use_aux_scheduler,
#             "aux_start_loss": args.aux_start_loss,
#             "aux_target_loss": args.aux_target_loss
#         }
#     )
        
#     writer = SummaryWriter(os.path.join(save_path, "tensorboard"))

#     train_transforms = transforms.Compose(
#         [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
#     )
#     test_transforms = transforms.Compose(
#         [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
#     )

#     train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
#     test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

#     compress_model = CompressModel(N=args.N, M=args.M)  # CPU owner
#     decompress_model = DecompressModel(N=args.N, M=args.M).cuda()  # GPU

#     # initial sync
#     ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

#     # update entropy tables initially (both models)
#     compress_model.update()
#     decompress_model.update()

#     train_dataloader = DataLoader(
#         train_dataset,
#         shuffle=True,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True,
#     )
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.test_batch_size,
#         num_workers=args.num_workers,
#         shuffle=False,
#         pin_memory=True,
#     )

#     compress_optimizer, compress_aux_optimizer, decompress_optimizer = configure_optimizers(
#         compress_model, decompress_model, args
#     )



#     milestones = args.lr_epoch if args.lr_epoch else [30, 40]
#     print("milestones:", milestones)

#     compress_lr_scheduler = optim.lr_scheduler.MultiStepLR(compress_optimizer, milestones, gamma=0.1, last_epoch=-1)
#     decompress_lr_scheduler = optim.lr_scheduler.MultiStepLR(decompress_optimizer, milestones, gamma=0.1, last_epoch=-1)

#     criterion = RateDistortionLoss(lmbda=args.lmbda, type=loss_type).cuda()
#     last_epoch = 0

#     if args.checkpoint:
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location='cpu')
#         if 'compress_model' in checkpoint:
#             compress_model.load_state_dict(checkpoint['compress_model']['state_dict'])
#             decompress_model.load_state_dict(checkpoint['decompress_model']['state_dict'])
#             if args.continue_train:
#                 last_epoch = checkpoint['epoch'] + 1
#                 compress_optimizer.load_state_dict(checkpoint['compress_model']['optimizer'])
#                 compress_aux_optimizer.load_state_dict(checkpoint['compress_model']['aux_optimizer'])
#                 decompress_optimizer.load_state_dict(checkpoint['decompress_model']['optimizer'])
#                 compress_lr_scheduler.load_state_dict(checkpoint['compress_model']['lr_scheduler'])
#                 decompress_lr_scheduler.load_state_dict(checkpoint['decompress_model']['lr_scheduler'])
#         else:
#             print("Legacy checkpoint format not supported in this script.")

#     print(f'last_epoch: {last_epoch}')
#     # Initialize aux scheduler if enabled
#     aux_scheduler = None
#     if args.use_aux_scheduler:
#         aux_scheduler = ExponentialTargetScheduler(
#             compress_aux_optimizer,
#             start_loss=args.aux_start_loss,
#             target_loss=args.aux_target_loss,
#             total_epochs=args.epochs-last_epoch
#         )
#         print(f"Initialized aux scheduler: start_loss={args.aux_start_loss}, target_loss={args.aux_target_loss}")

#     best_loss = float("inf")
#     for epoch in range(last_epoch, args.epochs):
#         train_one_epoch(
#             compress_model,
#             decompress_model,
#             criterion,
#             train_dataloader,
#             compress_optimizer,
#             compress_aux_optimizer,
#             decompress_optimizer,
#             epoch,
#             args.clip_max_norm,
#             None,
#             loss_type,
#             args,
#             [compress_lr_scheduler, decompress_lr_scheduler],
#             aux_scheduler
#         )

#         val_loss = test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, loss_type, args)
#         writer.add_scalar('test_loss', val_loss, epoch)

#         compress_lr_scheduler.step()
#         decompress_lr_scheduler.step()

#         is_best = val_loss < best_loss
#         best_loss = min(val_loss, best_loss)

#         if args.save:
#             compress_state = {
#                 "epoch": epoch,
#                 "state_dict": compress_model.state_dict(),
#                 "loss": val_loss,
#                 "optimizer": compress_optimizer.state_dict(),
#                 "aux_optimizer": compress_aux_optimizer.state_dict(),
#                 "lr_scheduler": compress_lr_scheduler.state_dict(),
#             }
#             decompress_state = {
#                 "epoch": epoch,
#                 "state_dict": decompress_model.state_dict(),
#                 "loss": val_loss,
#                 "optimizer": decompress_optimizer.state_dict(),
#                 "lr_scheduler": decompress_lr_scheduler.state_dict(),
#             }
#             torch.save(
#                 {
#                     'compress_model': compress_state,
#                     'decompress_model': decompress_state,
#                     'epoch': epoch,
#                     'is_best': is_best
#                 },
#                 os.path.join(save_path, "checkpoint_latest.pth.tar")
#             )
#             if epoch % 5 == 0:
#                 torch.save(
#                     {
#                         'compress_model': compress_state,
#                         'decompress_model': decompress_state,
#                         'epoch': epoch,
#                         'is_best': is_best
#                     },
#                     os.path.join(save_path, f"{epoch}_checkpoint.pth.tar")
#                 )
#             if is_best:
#                 torch.save(
#                     {
#                         'compress_model': compress_state,
#                         'decompress_model': decompress_state,
#                         'epoch': epoch,
#                         'is_best': is_best
#                     },
#                     os.path.join(save_path, "checkpoint_best.pth.tar")
#                 )
#                 # Enhanced best model logging
#                 wandb.log({
#                     "best_val_loss": val_loss, 
#                     "best_epoch": epoch,
#                     "epoch": epoch
#                 })
#             ParameterSync.save_shared_parameters(compress_model, os.path.join(save_path, "shared_params.pth"))

#     wandb.finish()

# if __name__ == "__main__":
#     main(sys.argv[1:])


## cross devices
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

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim

# Import your models
from models.dcae_5 import (
    CompressModel,
    DecompressModel,
    ParameterSync
)
from models import DCAE
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb

torch.set_num_threads(8)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

class ExponentialTargetScheduler:
    def __init__(self, optimizer, start_loss=3820, target_loss=10, total_epochs=100):
        self.optimizer = optimizer
        self.start_loss = start_loss
        self.target_loss = target_loss
        self.total_epochs = total_epochs
        
        # Calculate required decay rate
        self.decay_rate = (target_loss / start_loss) ** (1.0 / total_epochs)
        print(f"Target decay rate per epoch: {self.decay_rate:.6f}")
        
    def step(self, current_aux_loss, main_lr, epoch):
        """Exponential schedule to reach target in specified epochs"""
        
        # Expected loss at this epoch if following ideal trajectory
        expected_loss = self.start_loss * (self.decay_rate ** epoch)
        
        # If we're behind schedule, boost LR more
        if current_aux_loss > expected_loss * 1.5:
            # We're falling behind - boost aggressively
            boost_factor = (current_aux_loss / expected_loss) * 2
            multiplier = min(1000, 200 * boost_factor)
        elif current_aux_loss > expected_loss:
            # Slightly behind schedule
            boost_factor = current_aux_loss / expected_loss
            multiplier = min(500, 100 * boost_factor)
        else:
            # On track or ahead - use standard aggressive rate
            multiplier = max(50, 200 * (current_aux_loss / self.target_loss))
        
        new_lr = main_lr * multiplier
        new_lr = min(new_lr, 0.1)  # Safety cap
        
        self.optimizer.param_groups[0]['lr'] = new_lr
        
        print(f"Epoch {epoch}: Expected aux loss: {expected_loss:.1f}, "
              f"Actual: {current_aux_loss:.1f}, Multiplier: {multiplier:.1f}")
        
        return new_lr, multiplier

class RateDistortionLoss(nn.Module):
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
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out

class AverageMeter:
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        v = val.item() if torch.is_tensor(val) else float(val)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

def load_pretrained_to_split_models(checkpoint_path, compress_model, decompress_model, compress_device, decompress_device):
    """Load pretrained unified model weights to split models"""
    print(f"Loading pretrained weights from {checkpoint_path}")
    
    # Load unified model checkpoint on CPU first to avoid memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    unified_state_dict = {}
    
    # Clean up state dict keys
    for k, v in checkpoint["state_dict"].items():
        unified_state_dict[k.replace("module.", "")] = v
    
    # Create temporary unified model on CPU
    temp_unified_model = DCAE()
    temp_unified_model.load_state_dict(unified_state_dict)
    
    # Transfer encoder components to compress model
    compress_model.g_a.load_state_dict(temp_unified_model.g_a.state_dict())
    compress_model.h_a.load_state_dict(temp_unified_model.h_a.state_dict())
    
    # Transfer decoder components to decompress model
    decompress_model.g_s.load_state_dict(temp_unified_model.g_s.state_dict())
    
    # Transfer shared components to both models
    shared_components = [
        'dt', 'h_z_s1', 'h_z_s2', 'cc_mean_transforms', 
        'cc_scale_transforms', 'lrp_transforms', 'dt_cross_attention',
        'entropy_bottleneck', 'gaussian_conditional'
    ]
    
    for component in shared_components:
        if hasattr(temp_unified_model, component):
            if component == 'dt':
                compress_model.dt.data = temp_unified_model.dt.data.clone()
                decompress_model.dt.data = temp_unified_model.dt.data.clone()
            else:
                getattr(compress_model, component).load_state_dict(
                    getattr(temp_unified_model, component).state_dict()
                )
                getattr(decompress_model, component).load_state_dict(
                    getattr(temp_unified_model, component).state_dict()
                )
    
    print("Successfully transferred pretrained weights to split models")
    print(f"Compression model device: {compress_device}")
    print(f"Decompression model device: {decompress_device}")
    del temp_unified_model  # Clean up memory

def sync_gradients_cpu_gpu(cpu_model, gpu_model):
    """Copy grads from GPU shared modules to matching CPU modules (owner)."""
    # dt
    if gpu_model.dt.grad is not None:
        cpu_model.dt.grad = gpu_model.dt.grad.detach().cpu().clone()

    # module lists
    def copy_modlist_grads(cpu_list, gpu_list):
        for cpu_module, gpu_module in zip(cpu_list, gpu_list):
            for (cn, cp), (gn, gp) in zip(cpu_module.named_parameters(), gpu_module.named_parameters()):
                if gp.grad is not None:
                    if cp.grad is None:
                        cp.grad = gp.grad.detach().cpu().clone()
                    else:
                        cp.grad.copy_(gp.grad.detach().cpu())

    copy_modlist_grads(cpu_model.dt_cross_attention, gpu_model.dt_cross_attention)
    copy_modlist_grads(cpu_model.cc_mean_transforms, gpu_model.cc_mean_transforms)
    copy_modlist_grads(cpu_model.cc_scale_transforms, gpu_model.cc_scale_transforms)
    copy_modlist_grads(cpu_model.lrp_transforms, gpu_model.lrp_transforms)

    # single modules
    def copy_module_grads(cpu_m, gpu_m):
        for (cn, cp), (gn, gp) in zip(cpu_m.named_parameters(), gpu_m.named_parameters()):
            if gp.grad is not None:
                if cp.grad is None:
                    cp.grad = gp.grad.detach().cpu().clone()
                else:
                    cp.grad.copy_(gp.grad.detach().cpu())

    copy_module_grads(cpu_model.h_z_s1, gpu_model.h_z_s1)
    copy_module_grads(cpu_model.h_z_s2, gpu_model.h_z_s2)  
    copy_module_grads(cpu_model.entropy_bottleneck, gpu_model.entropy_bottleneck)
    copy_module_grads(cpu_model.gaussian_conditional, gpu_model.gaussian_conditional)

def sync_shared_parameters_during_training(compress_model, decompress_model, epoch):
    """Enhanced parameter synchronization during cross-device training"""
    # Average the parameters of shared components between devices
    shared_components = ['dt', 'dt_cross_attention', 'cc_mean_transforms', 
                        'cc_scale_transforms', 'lrp_transforms', 'h_z_s1', 'h_z_s2']
    
    for component_name in shared_components:
        if component_name == 'dt':
            # Average dictionary parameters
            compress_dt = getattr(compress_model, component_name)
            decompress_dt = getattr(decompress_model, component_name)
            
            # Move to CPU for averaging, then distribute
            avg_dt = (compress_dt.data + decompress_dt.data.cpu()) / 2
            compress_dt.data = avg_dt
            decompress_dt.data = avg_dt.cuda()
        else:
            # Handle module lists and modules
            compress_component = getattr(compress_model, component_name)
            decompress_component = getattr(decompress_model, component_name)
            
            if isinstance(compress_component, nn.ModuleList):
                for i in range(len(compress_component)):
                    sync_module_parameters(compress_component[i], decompress_component[i])
            else:
                sync_module_parameters(compress_component, decompress_component)

def sync_module_parameters(compress_module, decompress_module):
    """Sync individual module parameters between CPU and GPU"""
    compress_params = dict(compress_module.named_parameters())
    decompress_params = dict(decompress_module.named_parameters())
    
    for name in compress_params:
        if name in decompress_params:
            # Average parameters
            compress_param = compress_params[name]
            decompress_param = decompress_params[name]
            
            # Move GPU param to CPU, average, then distribute
            avg_param = (compress_param.data + decompress_param.data.cpu()) / 2
            compress_param.data = avg_param
            decompress_param.data = avg_param.cuda()

def configure_optimizers(compress_model, decompress_model, args):
    """Configure optimizers. GPU optimizer excludes shared params."""
    def is_shared(name):
        shared_prefixes = [
            "dt", "dt_cross_attention", "cc_mean_transforms",
            "cc_scale_transforms", "lrp_transforms", "h_z_s1",
            "h_z_s2", "entropy_bottleneck", "gaussian_conditional",
        ]
        return any(name == p or name.startswith(p + ".") for p in shared_prefixes)

    # Compress (owner)
    compress_parameters = {
        n for n, p in compress_model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    compress_aux_parameters = {
        n for n, p in compress_model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Decompress (decoder-only)
    decompress_parameters = {
        n for n, p in decompress_model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad and not is_shared(n)
    }

    compress_params_dict = dict(compress_model.named_parameters())
    decompress_params_dict = dict(decompress_model.named_parameters())

    compress_optimizer = optim.Adam(
        (compress_params_dict[n] for n in sorted(compress_parameters)),
        lr=args.learning_rate,
    )
    compress_aux_optimizer = optim.Adam(
        (compress_params_dict[n] for n in sorted(compress_aux_parameters)),
        lr=float(args.aux_learning_rate),
    )
    decompress_optimizer = optim.Adam(
        (decompress_params_dict[n] for n in sorted(decompress_parameters)),
        lr=args.learning_rate,
    )
    return compress_optimizer, compress_aux_optimizer, decompress_optimizer

def train_one_epoch(
    compress_model, decompress_model, criterion, train_dataloader, 
    compress_optimizer, compress_aux_optimizer, decompress_optimizer,
    epoch, clip_max_norm, train_sampler, loss_type='mse', args=None, lr_schedulers=None, aux_scheduler=None
):
    compress_model.train()
    decompress_model.train()

    # Initialize meters for epoch-level logging
    epoch_loss = AverageMeter()
    epoch_bpp_loss = AverageMeter()
    epoch_aux_loss = AverageMeter()
    epoch_mse_loss = AverageMeter()
    epoch_ms_ssim_loss = AverageMeter()

    pre_time = time.time()
    
    for i, d in enumerate(train_dataloader):
        d_cpu = d  # on CPU
        d_gpu = d.cuda(non_blocking=True)

        # Enhanced synchronization for cross-device training
        if (i + 1) % 5 == 0:
            ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)
            if args.cross_device_training:
                sync_shared_parameters_during_training(compress_model, decompress_model, epoch)

        # zero grads
        compress_optimizer.zero_grad(set_to_none=True)
        decompress_optimizer.zero_grad(set_to_none=True)

        # forward CPU (compressor/entropy)
        compress_out = compress_model(d_cpu)

        # move to GPU for decoder and loss
        y_hat_gpu = compress_out["y_hat"].cuda(non_blocking=True).requires_grad_(True)
        z_hat_gpu = compress_out["z_hat"].cuda(non_blocking=True).requires_grad_(True)
        decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)

        # move likelihoods to GPU for loss
        likelihoods_gpu = {k: v.cuda(non_blocking=True) for k, v in compress_out["likelihoods"].items()}
        combined_out = {"x_hat": decompress_out["x_hat"], "likelihoods": likelihoods_gpu}

        # Cross-device precision alignment loss (optional regularization)
        if args.cross_device_training and args.precision_regularization:
            device_transfer_noise = torch.randn_like(y_hat_gpu) * 1e-6
            y_hat_gpu_noisy = y_hat_gpu + device_transfer_noise
            decompress_out_noisy = decompress_model(y_hat_gpu_noisy, z_hat_gpu)
            precision_loss = torch.nn.functional.mse_loss(decompress_out["x_hat"], decompress_out_noisy["x_hat"])
            combined_out["precision_loss"] = precision_loss

        # loss and backward on GPU
        out_criterion = criterion(combined_out, d_gpu)
        
        # Add precision regularization to total loss if enabled
        if args.cross_device_training and args.precision_regularization:
            out_criterion["loss"] += 0.001 * combined_out["precision_loss"]
        
        out_criterion["loss"].backward()

        # copy grads for shared params back to cpu
        sync_gradients_cpu_gpu(compress_model, decompress_model)

        # clip and step
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(compress_model.parameters(), clip_max_norm)
            torch.nn.utils.clip_grad_norm_(decompress_model.parameters(), clip_max_norm)
        compress_optimizer.step()
        decompress_optimizer.step()

        # aux optimizer on CPU quantiles
        compress_aux_optimizer.zero_grad(set_to_none=True)
        aux = compress_model.aux_loss()
        aux.backward()
        
        # Apply aux scheduler before optimizer step
        if aux_scheduler is not None:
            current_lr = lr_schedulers[0].get_last_lr()[0] if lr_schedulers else compress_optimizer.param_groups[0]['lr']
            aux_lr, multiplier = aux_scheduler.step(aux.item(), current_lr, epoch)
            
            # Log aux scheduler metrics
            wandb.log({
                'train/aux_learning_rate': aux_lr,
                'train/aux_lr_multiplier': multiplier,
                'train/expected_aux_loss': aux_scheduler.start_loss * (aux_scheduler.decay_rate ** epoch),
                'epoch': epoch,
                'step': epoch * len(train_dataloader) + i
            })
        
        compress_aux_optimizer.step()

        # Update epoch meters
        aux_post = compress_model.aux_loss().item()
        epoch_loss.update(out_criterion["loss"])
        epoch_bpp_loss.update(out_criterion["bpp_loss"])
        epoch_aux_loss.update(aux_post)
        
        if loss_type == 'mse':
            epoch_mse_loss.update(out_criterion["mse_loss"])
        else:
            epoch_ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

        # periodically refresh entropy tables and resync
        update_freq = 50
        if (i + 1) % update_freq == 0:
            compress_model.update()
        ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

        # Enhanced logging
        if (i + 1) % 10 == 0:
            now_time = time.time()
            step_time = now_time - pre_time
            pre_time = now_time
            
            current_lr = lr_schedulers[0].get_last_lr()[0] if lr_schedulers else compress_optimizer.param_groups[0]['lr']
            
            # Log step-level metrics to wandb
            step = epoch * len(train_dataloader) + i
            log_dict = {
                'train/loss': out_criterion["loss"].item(),
                'train/bpp_loss': out_criterion["bpp_loss"].item(),
                'train/aux_loss': aux_post,
                'train/compress_aux_loss': aux_post,
                'train/learning_rate': current_lr,
                'train/step_time': step_time,
                'epoch': epoch,
                'step': step
            }
            
            if args.cross_device_training and args.precision_regularization and "precision_loss" in combined_out:
                log_dict['train/precision_loss'] = combined_out["precision_loss"].item()
            
            if loss_type == 'mse':
                log_dict['train/mse_loss'] = out_criterion["mse_loss"].item()
                print(
                    f"Train epoch {epoch}: step {i+1}/{len(train_dataloader)} | "
                    f"Loss: {out_criterion['loss'].item():.3f} | "
                    f"MSE: {out_criterion['mse_loss'].item():.3f} | "
                    f"Bpp: {out_criterion['bpp_loss'].item():.3f} | "
                    f"Aux: {aux_post:.2f} | "
                    f"Time: {step_time:.2f}"
                )
            else:
                log_dict['train/ms_ssim_loss'] = out_criterion["ms_ssim_loss"].item()
                print(
                    f"Train epoch {epoch}: step {i+1}/{len(train_dataloader)} | "
                    f"Loss: {out_criterion['loss'].item():.3f} | "
                    f"MS-SSIM: {out_criterion['ms_ssim_loss'].item():.4f} | "
                    f"Bpp: {out_criterion['bpp_loss'].item():.3f} | "
                    f"Aux: {aux_post:.2f} | "
                    f"Time: {step_time:.2f}"
                )
            
            wandb.log(log_dict)

    # Log epoch-level metrics to wandb
    wandb_log = {
        "epoch": epoch,
        "train/epoch_loss": epoch_loss.avg,
        "train/epoch_bpp_loss": epoch_bpp_loss.avg,
        "train/epoch_aux_loss": epoch_aux_loss.avg,
        "train/final_learning_rate": compress_optimizer.param_groups[0]['lr']
    }
    
    if loss_type == 'mse':
        wandb_log["train/epoch_mse_loss"] = epoch_mse_loss.avg
    else:
        wandb_log["train/epoch_ms_ssim_loss"] = epoch_ms_ssim_loss.avg
    
    wandb.log(wandb_log)

def test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, loss_type='mse', args=None):
    compress_model.eval()
    decompress_model.eval()
    
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    aux_meter = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d_cpu = d
            d_gpu = d.cuda(non_blocking=True)
            compress_out = compress_model(d_cpu)
            y_hat_gpu = compress_out["y_hat"].cuda(non_blocking=True)
            z_hat_gpu = compress_out["z_hat"].cuda(non_blocking=True)
            decompress_out = decompress_model(y_hat_gpu, z_hat_gpu)
            likelihoods_gpu = {k: v.cuda(non_blocking=True) for k, v in compress_out["likelihoods"].items()}
            combined_out = {"x_hat": decompress_out["x_hat"], "likelihoods": likelihoods_gpu}
            out_criterion = criterion(combined_out, d_gpu)

            loss.update(out_criterion["loss"])
            bpp_loss.update(out_criterion["bpp_loss"])
            aux_meter.update(compress_model.aux_loss())

            if loss_type == 'mse':
                mse_loss.update(out_criterion["mse_loss"])
            else:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

    # Enhanced logging
    wandb_log = {
        "epoch": epoch,
        "val/loss": loss.avg,
        "val/bpp_loss": bpp_loss.avg,
        "val/aux_loss": aux_meter.avg,
        "val/compress_aux_loss": aux_meter.avg
    }
    
    if loss_type == 'mse':
        wandb_log["val/mse_loss"] = mse_loss.avg
        print(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.3f} | "
            f"MSE loss: {mse_loss.avg:.3f} | "
            f"Bpp loss: {bpp_loss.avg:.2f} | "
            f"Aux loss: {aux_meter.avg:.2f}"
        )
    else:
        wandb_log["val/ms_ssim_loss"] = ms_ssim_loss.avg
        print(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.3f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.2f} | "
            f"Aux loss: {aux_meter.avg:.2f}"
        )
    
    wandb.log(wandb_log)
    return loss.avg

def parse_args(argv):
    parser = argparse.ArgumentParser(description="CPU-GPU split training script with cross-device optimization.")
    parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("-d", "--dataset", type=str, default='./dataset', help="Training dataset")
    parser.add_argument("-e", "--epochs", default=400, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=8)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=60.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--aux-learning-rate", default=5e-3)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str, default='60.5checkpoint_best.pth.tar')
    parser.add_argument("--type", type=str, default='ms-ssim', choices=['mse', 'ms-ssim', 'l1'])
    parser.add_argument("--save_path", type=str, default='./checkpoints/train_5/try_9')
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--lr_epoch", nargs='+', type=int)
    parser.add_argument("--continue_train", action="store_true", default=True)
    
    # Cross-device training arguments
    parser.add_argument("--cross_device_training", action="store_true", default=True, 
                       help="Enable cross-device training optimizations")
    parser.add_argument("--precision_regularization", action="store_true", default=False,
                       help="Add precision regularization for device transfer stability")
    parser.add_argument("--load_pretrained_checkpoint", action="store_true", default=False,
                       help="Load from pretrained unified model checkpoint (as used in eval_5.py)")
    
    # Enhanced wandb arguments
    parser.add_argument("--wandb_project", type=str, default="Image-compression", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default='train_5_try8', help="WandB run name")
    parser.add_argument("--wandb_tags", nargs='+', type=str, default=[], help="wandb tags")
    
    # Aux scheduler parameters
    parser.add_argument("--aux_start_loss", type=float, default=7913.0, help="Starting aux loss for scheduler")
    parser.add_argument("--aux_target_loss", type=float, default=10.0, help="Target aux loss for scheduler")
    parser.add_argument("--use_aux_scheduler", action="store_true", default=True, help="Use exponential target scheduler for aux loss")
    
    args = parser.parse_args(argv)
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    
    loss_type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "tensorboard"), exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    # Enhanced wandb initialization
    wandb_name = args.wandb_run_name or f"lambda_{args.lmbda}_{loss_type}_N{args.N}_M{args.M}_cross_device"
    wandb.init(
        project=args.wandb_project,
        name=wandb_name,
        tags=args.wandb_tags + (["cross-device"] if args.cross_device_training else []),
        config={
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "aux_learning_rate": args.aux_learning_rate,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "lambda": args.lmbda,
            "loss_type": loss_type,
            "patch_size": args.patch_size,
            "N": args.N,
            "M": args.M,
            "clip_max_norm": args.clip_max_norm,
            "seed": args.seed,
            "lr_epoch": args.lr_epoch,
            "num_workers": args.num_workers,
            "dataset": args.dataset,
            "save_path": args.save_path,
            "continue_train": args.continue_train,
            "architecture": "CPU-GPU-Split",
            "use_aux_scheduler": args.use_aux_scheduler,
            "aux_start_loss": args.aux_start_loss,
            "aux_target_loss": args.aux_target_loss,
            "cross_device_training": args.cross_device_training,
            "precision_regularization": args.precision_regularization,
            "load_pretrained_checkpoint": args.load_pretrained_checkpoint
        }
    )
        
    writer = SummaryWriter(os.path.join(save_path, "tensorboard"))

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    compress_model = CompressModel(N=args.N, M=args.M)  # CPU owner
    decompress_model = DecompressModel(N=args.N, M=args.M).cuda()  # GPU

    # Load from pretrained checkpoint if specified (like in eval_5.py)
    if args.load_pretrained_checkpoint and args.checkpoint:
        print(f"Loading from pretrained unified checkpoint: {args.checkpoint}")
        load_pretrained_to_split_models(args.checkpoint, compress_model, decompress_model, 
                                       torch.device('cpu'), torch.device('cuda'))
    else:
        # initial sync for regular training
        ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

    # update entropy tables initially (both models)
    compress_model.update()
    decompress_model.update()

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    compress_optimizer, compress_aux_optimizer, decompress_optimizer = configure_optimizers(
        compress_model, decompress_model, args
    )

    milestones = args.lr_epoch if args.lr_epoch else [30, 40]
    print("milestones:", milestones)

    compress_lr_scheduler = optim.lr_scheduler.MultiStepLR(compress_optimizer, milestones, gamma=0.1, last_epoch=-1)
    decompress_lr_scheduler = optim.lr_scheduler.MultiStepLR(decompress_optimizer, milestones, gamma=0.1, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=loss_type).cuda()
    last_epoch = 0

    # Handle checkpoint loading for both pretrained and continue training scenarios
    if args.checkpoint and not args.load_pretrained_checkpoint:
        print("Loading split model checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'compress_model' in checkpoint:
            compress_model.load_state_dict(checkpoint['compress_model']['state_dict'])
            decompress_model.load_state_dict(checkpoint['decompress_model']['state_dict'])
            if args.continue_train:
                last_epoch = checkpoint['epoch'] + 1
                compress_optimizer.load_state_dict(checkpoint['compress_model']['optimizer'])
                compress_aux_optimizer.load_state_dict(checkpoint['compress_model']['aux_optimizer'])
                decompress_optimizer.load_state_dict(checkpoint['decompress_model']['optimizer'])
                compress_lr_scheduler.load_state_dict(checkpoint['compress_model']['lr_scheduler'])
                decompress_lr_scheduler.load_state_dict(checkpoint['decompress_model']['lr_scheduler'])
        else:
            print("Legacy checkpoint format not supported in this script.")

    print(f'last_epoch: {last_epoch}')
    
    # Initialize aux scheduler if enabled
    aux_scheduler = None
    if args.use_aux_scheduler:
        aux_scheduler = ExponentialTargetScheduler(
            compress_aux_optimizer,
            start_loss=args.aux_start_loss,
            target_loss=args.aux_target_loss,
            total_epochs=args.epochs-last_epoch
        )
        print(f"Initialized aux scheduler: start_loss={args.aux_start_loss}, target_loss={args.aux_target_loss}")

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(
            compress_model,
            decompress_model,
            criterion,
            train_dataloader,
            compress_optimizer,
            compress_aux_optimizer,
            decompress_optimizer,
            epoch,
            args.clip_max_norm,
            None,
            loss_type,
            args,
            [compress_lr_scheduler, decompress_lr_scheduler],
            aux_scheduler
        )

        val_loss = test_epoch(epoch, test_dataloader, compress_model, decompress_model, criterion, loss_type, args)
        writer.add_scalar('test_loss', val_loss, epoch)

        compress_lr_scheduler.step()
        decompress_lr_scheduler.step()

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if args.save:
            compress_state = {
                "epoch": epoch,
                "state_dict": compress_model.state_dict(),
                "loss": val_loss,
                "optimizer": compress_optimizer.state_dict(),
                "aux_optimizer": compress_aux_optimizer.state_dict(),
                "lr_scheduler": compress_lr_scheduler.state_dict(),
            }
            decompress_state = {
                "epoch": epoch,
                "state_dict": decompress_model.state_dict(),
                "loss": val_loss,
                "optimizer": decompress_optimizer.state_dict(),
                "lr_scheduler": decompress_lr_scheduler.state_dict(),
            }
            torch.save(
                {
                    'compress_model': compress_state,
                    'decompress_model': decompress_state,
                    'epoch': epoch,
                    'is_best': is_best
                },
                os.path.join(save_path, "checkpoint_latest.pth.tar")
            )
            if epoch % 5 == 0:
                torch.save(
                    {
                        'compress_model': compress_state,
                        'decompress_model': decompress_state,
                        'epoch': epoch,
                        'is_best': is_best
                    },
                    os.path.join(save_path, f"{epoch}_checkpoint.pth.tar")
                )
            if is_best:
                torch.save(
                    {
                        'compress_model': compress_state,
                        'decompress_model': decompress_state,
                        'epoch': epoch,
                        'is_best': is_best
                    },
                    os.path.join(save_path, "checkpoint_best.pth.tar")
                )
                # Enhanced best model logging
                wandb.log({
                    "best_val_loss": val_loss, 
                    "best_epoch": epoch,
                    "epoch": epoch
                })
            ParameterSync.save_shared_parameters(compress_model, os.path.join(save_path, "shared_params.pth"))

    wandb.finish()

if __name__ == "__main__":
    main(sys.argv[1:])