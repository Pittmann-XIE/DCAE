# ## Train 4: based on train 3, with wandbimport os. It uses dcae_2, which approximates the quantization 
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
# from models.dcae_5_2 import CompressModel, DecompressModel, ParameterSync


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
#     parser.add_argument("--checkpoint", type=str, default='checkpoints/train_5/try_4/60.5/checkpoint_best.pth.tar')
#     parser.add_argument("--type", type=str, default='ms-ssim', choices=['mse', 'ms-ssim', 'l1'])
#     parser.add_argument("--save_path", type=str, default='./checkpoints/train_5_2/try_1')
#     parser.add_argument("--N", type=int, default=192)
#     parser.add_argument("--M", type=int, default=320)
#     parser.add_argument("--lr_epoch", nargs='+', type=int)
#     parser.add_argument("--continue_train", action="store_true", default=True)
    
#     # Enhanced wandb arguments similar to code 2
#     parser.add_argument("--wandb_project", type=str, default="Image-compression", help="WandB project name")
#     parser.add_argument("--wandb_run_name", type=str, default='train_5__2_try1_20250915_01', help="WandB run name")
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
from models.dcae_5_2 import CompressModel, DecompressModel, ParameterSync


from torch.utils.tensorboard import SummaryWriter
import numpy as np
import wandb  # Add wandb import

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

        # occasional full sync to keep modules/buffers aligned
        if (i + 1) % 5 == 0:
            ParameterSync.sync_compress_to_decompress(compress_model, decompress_model)

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

        # loss and backward on GPU
        out_criterion = criterion(combined_out, d_gpu)
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

        # Enhanced logging similar to code 2
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
                'train/compress_aux_loss': aux_post,  # Only compress aux in code 1
                'train/learning_rate': current_lr,
                'train/step_time': step_time,
                'epoch': epoch,
                'step': step
            }
            
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

    # Log epoch-level metrics to wandb (keeping this for compatibility)
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

    # Enhanced logging similar to code 2
    wandb_log = {
        "epoch": epoch,
        "val/loss": loss.avg,
        "val/bpp_loss": bpp_loss.avg,
        "val/aux_loss": aux_meter.avg,
        "val/compress_aux_loss": aux_meter.avg  # Only compress aux in code 1
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
    parser = argparse.ArgumentParser(description="CPU-GPU split training script.")
    parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("-d", "--dataset", type=str, default='./dataset', help="Training dataset")
    parser.add_argument("-e", "--epochs", default=400, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=8)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=60.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--aux-learning-rate", default=5e-3)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--checkpoint", type=str, default='checkpoints/train_5/try_4/60.5/checkpoint_best.pth.tar')
    parser.add_argument("--type", type=str, default='ms-ssim', choices=['mse', 'ms-ssim', 'l1'])
    parser.add_argument("--save_path", type=str, default='./checkpoints/train_5_2/try_1')
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--lr_epoch", nargs='+', type=int)
    parser.add_argument("--continue_train", action="store_true", default=True)
    
    # Enhanced wandb arguments similar to code 2
    parser.add_argument("--wandb_project", type=str, default="Image-compression", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default='train_5__2_try1_20250915_01', help="WandB run name")
    parser.add_argument("--wandb_tags", nargs='+', type=str, default=[], help="wandb tags")
    
    # Aux scheduler parameters
    parser.add_argument("--aux_start_loss", type=float, default=3659.0, help="Starting aux loss for scheduler")
    parser.add_argument("--aux_target_loss", type=float, default=1800.0, help="Target aux loss for scheduler")
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

    # Enhanced wandb initialization similar to code 2
    wandb_name = args.wandb_run_name or f"lambda_{args.lmbda}_{loss_type}_N{args.N}_M{args.M}"
    wandb.init(
        project=args.wandb_project,
        name=wandb_name,
        tags=args.wandb_tags,
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
            "aux_target_loss": args.aux_target_loss
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

    # initial sync
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

    if args.checkpoint:
        print("Loading", args.checkpoint)
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