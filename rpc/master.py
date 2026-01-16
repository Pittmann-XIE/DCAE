# --- START OF FILE master.py ---
import os
import sys
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim
from models.dcae_5_fixed import CompressModel

from rpc_shared import (
    MASTER_ADDR, MASTER_PORT, create_remote_decompressor, 
    RPC_TIMEOUT, SHARED_PREFIXES
)

os.environ.setdefault('TP_SOCKET_IFNAME', 'eno8303')
os.environ.setdefault('GLOO_SOCKET_IFNAME', 'eno8303')

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def collect_shared_params(model):
    full_state = model.state_dict()
    shared_state = {}
    for key, value in full_state.items():
        if any(key.startswith(prefix) for prefix in SHARED_PREFIXES):
            shared_state[key] = value.cpu()
    return shared_state

def load_checkpoint_logic(compress_model, decompress_rref, checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    raw_state_dict = checkpoint.get('state_dict', checkpoint)

    if 'compress_model' in raw_state_dict:
        print("Detected Split Checkpoint format.")
        compress_model.load_state_dict(raw_state_dict['compress_model'])
        decompress_rref.rpc_sync().load_full_state_dict(raw_state_dict['decompress_model'])
    else:
        print("Detected Unified/Pretrained Checkpoint format. Adapting weights...")
        clean_state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}
        
        compress_sd = {k: v for k, v in clean_state_dict.items() if not k.startswith("g_s.")}
        decompress_sd = {}
        for k, v in clean_state_dict.items():
            if k.startswith("g_s.") or any(k.startswith(p) for p in SHARED_PREFIXES):
                decompress_sd[k] = v
        
        try:
            compress_model.load_state_dict(compress_sd, strict=True)
            decompress_rref.rpc_sync().load_full_state_dict(decompress_sd)
            print("Weights loaded successfully.")
        except RuntimeError as e:
            print(f"Warning loading weights: {e}")

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, x_hat, target, likelihoods):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        bpp_loss = sum((torch.log(l).sum() / (-math.log(2) * num_pixels)) for l in likelihoods.values())
        
        if self.type == 'mse':
            mse_loss = self.mse(x_hat, target)
            loss = self.lmbda * 255 ** 2 * mse_loss + bpp_loss
            return loss, mse_loss, bpp_loss
        else:
            ms_ssim_loss = ms_ssim(x_hat, target, data_range=1.)
            loss = self.lmbda * (1 - ms_ssim_loss) + bpp_loss
            return loss, ms_ssim_loss, bpp_loss

# --- VALIDATION FUNCTION 1: Standard Forward (Fast) ---
def test_epoch(epoch, test_dataloader, compress_model, decompress_rref, criterion, device):
    compress_model.eval()
    decompress_rref.rpc_sync().set_mode(training=False)
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    
    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            out_enc = compress_model(d)
            x_hat = decompress_rref.rpc_sync().forward(out_enc['y_hat'], out_enc['z_hat'])
            if x_hat.device != device: x_hat = x_hat.to(device)
            
            loss, mse_loss, bpp_loss = criterion(x_hat, d, out_enc['likelihoods'])
            
            loss_meter.update(loss.item())
            bpp_meter.update(bpp_loss.item())
            mse_meter.update(mse_loss.item())

    print(f"Test Epoch {epoch}: Loss {loss_meter.avg:.4f} | Bpp {bpp_meter.avg:.4f}")
    wandb.log({
        "val/loss": loss_meter.avg,
        "val/bpp": bpp_meter.avg,
        "val/mse": mse_meter.avg,
        "epoch": epoch
    })
    
    compress_model.train()
    decompress_rref.rpc_sync().set_mode(training=True)
    return loss_meter.avg

# --- VALIDATION FUNCTION 2: Real Compression (Slow/Accurate) ---
def test_epoch_real(epoch, test_dataloader, compress_model, decompress_rref, device):
    print(f"Running REAL encoding/decoding on epoch {epoch}...")
    compress_model.eval()
    decompress_rref.rpc_sync().set_mode(training=False)
    
    avg_bpp = AverageMeter()
    avg_psnr = AverageMeter()
    avg_mssim = AverageMeter()
    
    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            # Input image
            x = d.to(device)
            b, c, h, w = x.shape
            num_pixels = b * h * w
            
            # 1. Compress (Local)
            # Returns dictionary: {"strings": [...], "shape": ...}
            out_enc = compress_model.compress(x)
            
            # 2. Calculate Real Bitrate
            total_bytes = 0
            for stream_list in out_enc["strings"]:
                for s in stream_list:
                    total_bytes += len(s)
            
            real_bpp = (total_bytes * 8) / num_pixels
            avg_bpp.update(real_bpp, b)
            
            # 3. Decompress (Remote)
            # We pass the byte strings and shape to the worker via RPC
            x_hat = decompress_rref.rpc_sync().decompress(out_enc["strings"], out_enc["shape"])
            
            # Move back to Master for metric calculation
            if x_hat.device != device:
                x_hat = x_hat.to(device)
            
            # 4. Metrics
            mse = F.mse_loss(x_hat, x)
            psnr = 10 * (torch.log10(1 / mse))
            avg_psnr.update(psnr.item(), b)
            
            mssim_val = ms_ssim(x_hat, x, data_range=1.0)
            avg_mssim.update(mssim_val.item(), b)

    print(f"REAL VALIDATION Epoch {epoch}: Bpp: {avg_bpp.avg:.4f} | PSNR: {avg_psnr.avg:.2f}dB | MS-SSIM: {avg_mssim.avg:.4f}")
    
    wandb.log({
        "val_real/bpp": avg_bpp.avg,
        "val_real/psnr": avg_psnr.avg,
        "val_real/ms_ssim": avg_mssim.avg,
        "epoch": epoch
    })
    
    compress_model.train()
    decompress_rref.rpc_sync().set_mode(training=True)

def run_training(args):
    # 1. Init RPC
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        num_worker_threads=16,
        rpc_timeout=RPC_TIMEOUT,
        _transports=["uv"]
    )
    
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        rpc_backend_options.device_maps = {"worker": {device: torch.device("cuda:0")}}

    print(f"Initializing Master on {device}...")
    rpc.init_rpc("master", rank=0, world_size=2, rpc_backend_options=rpc_backend_options)
    
    # 2. Models
    compress_model = CompressModel(N=args.N, M=args.M).to(device)
    compress_model.train()
    
    decompress_rref = rpc.remote(
        "worker",
        create_remote_decompressor,
        args=(args.N, args.M, "cuda:0"), 
        timeout=RPC_TIMEOUT
    )

    if args.checkpoint:
        load_checkpoint_logic(compress_model, decompress_rref, args.checkpoint)

    # 3. Optimizers
    local_params = [p for n, p in compress_model.named_parameters() if not n.endswith(".quantiles") and p.requires_grad]
    local_optimizer = optim.Adam(local_params, lr=args.learning_rate)
    
    aux_params = [p for n, p in compress_model.named_parameters() if n.endswith(".quantiles") and p.requires_grad]
    compress_aux_optimizer = optim.Adam(aux_params, lr=args.aux_learning_rate)

    remote_optimizer = DistributedOptimizer(
        optim.Adam,
        decompress_rref.rpc_sync().get_unique_parameters_rref(),
        lr=args.learning_rate
    )

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=args.type)
    
    # 4. Data
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])
    
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False)

    wandb.init(project=args.wandb_project, config=args, name=f"rpc_train_{args.lmbda}_{int(time.time())}")
    
    print("Starting Training Loop...")
    best_loss = float("inf")
    global_step = 0
    
    for epoch in range(args.epochs):
        compress_model.train()
        
        for i, d in enumerate(train_dataloader):
            d = d.to(device)
            local_optimizer.zero_grad()
            compress_aux_optimizer.zero_grad()
            
            with dist_autograd.context() as context_id:
                out_enc = compress_model(d)
                x_hat = decompress_rref.rpc_sync().forward(out_enc['y_hat'], out_enc['z_hat'])
                if x_hat.device != device: x_hat = x_hat.to(device)
                loss, dist_loss, bpp_loss = criterion(x_hat, d, out_enc['likelihoods'])

                dist_autograd.backward(context_id, [loss])
                
                remote_grads = decompress_rref.rpc_sync().get_shared_gradients()
                for name, p in compress_model.named_parameters():
                    if name in remote_grads:
                        grad_from_worker = remote_grads[name].to(device)
                        if p.grad is None: p.grad = grad_from_worker
                        else: p.grad += grad_from_worker
                
                local_optimizer.step()
                remote_optimizer.step(context_id)
                decompress_rref.rpc_sync().zero_shared_grads()

            aux_loss = compress_model.aux_loss()
            aux_loss.backward()
            compress_aux_optimizer.step()

            decompress_rref.rpc_sync().sync_shared_weights(collect_shared_params(compress_model))

            if (i + 1) % 50 == 0:
                compress_model.update()
                decompress_rref.rpc_sync().update_entropy_tables()

            if i % 10 == 0:
                # Calculate metrics for printing
                current_loss = loss.item()
                current_dist = dist_loss.item()
                current_bpp = bpp_loss.item()
                current_aux = aux_loss.item()

                print(
                    f"Epoch {epoch} [{i}/{len(train_dataloader)}] "
                    f"Loss: {current_loss:.4f} | "
                    f"Dist: {current_dist:.6f} | "
                    f"Bpp: {current_bpp:.4f} | "
                    f"Aux: {current_aux:.4f}"
                )
                
                # Construct log dictionary
                log_dict = {
                    "train/loss": current_loss,
                    "train/bpp": current_bpp,
                    "train/aux_loss": current_aux,  # <-- Added aux loss
                    "train/dist": current_dist,     # <-- Added dist loss
                    "epoch": epoch,
                    "global_step": global_step
                }
                
                # Optional: Make the log key clearer based on the metric type
                if args.type == 'mse':
                    log_dict["train/mse"] = dist_loss.item()
                else:
                    log_dict["train/ms_ssim"] = dist_loss.item()
                wandb.log(log_dict)

            global_step += 1

        # --- VALIDATION LOGIC ---
        # 1. Fast Validation (Estimates)
        val_loss = test_epoch(epoch, test_dataloader, compress_model, decompress_rref, criterion, device)
        
        # Save Best Model
        if args.save and val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(args.save_path, str(args.lmbda))
            os.makedirs(save_path, exist_ok=True)
            remote_state = decompress_rref.rpc_sync().get_state_dict()
            torch.save({
                'compress_model': compress_model.state_dict(),
                'decompress_model': remote_state,
                'epoch': epoch
            }, os.path.join(save_path, "best_checkpoint.pth.tar"))

        # 2. Real Validation (Arithmetic Coding)
        if (epoch + 1) % 10 == 0:
            # Sync tables to ensure decoder has latest CDFs
            compress_model.update(force=True)
            decompress_rref.rpc_sync().update_entropy_tables()
            # Sync parameters one last time
            decompress_rref.rpc_sync().sync_shared_weights(collect_shared_params(compress_model))
            
            test_epoch_real(epoch, test_dataloader, compress_model, decompress_rref, device)

    print("Training Complete.")
    wandb.finish()
    rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPC Distributed Training for DCAE")
    parser.add_argument("-d", "--dataset", type=str, default='../dataset')
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-5, type=float)
    parser.add_argument("--aux-learning-rate", default=1e-4, type=float)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=1) # Reduced for real compression
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--save_path", type=str, default='./checkpoints_rpc')
    parser.add_argument("--type", type=str, default='ms-ssim', choices=['mse', 'ms-ssim'])
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="dcae_rpc_training")
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--load_pretrained", action="store_true")
    parser.add_argument("--cuda", action="store_true", default=True)

    args = parser.parse_args()
    run_training(args)