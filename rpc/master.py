import os
import argparse
import math
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import wandb

from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim

import sys
import os
# Add the project root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dcae_5 import CompressModel

from rpc_shared import (
    MASTER_ADDR, MASTER_PORT, create_remote_decompressor, 
    RPC_TIMEOUT
)

os.environ.setdefault('TP_SOCKET_IFNAME', 'eno8303')
os.environ.setdefault('GLOO_SOCKET_IFNAME', 'eno8303')

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, type='mse', device='cpu'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type
        self.device = device

    def forward(self, x_hat, target, likelihoods):
        N, _, H, W = target.size()
        num_pixels = N * H * W

        # Likelihoods are on Master (CPU/GPU), compute BPP locally
        bpp_loss = sum(
            (torch.log(l).sum() / (-math.log(2) * num_pixels))
            for l in likelihoods.values()
        )

        # Ensure target is on the same device as x_hat (likely CPU if Master is CPU)
        if target.device != x_hat.device:
            target = target.to(x_hat.device)

        if self.type == 'mse':
            mse_loss = self.mse(x_hat, target)
            loss = self.lmbda * 255 ** 2 * mse_loss + bpp_loss
            return loss, mse_loss, bpp_loss
        else:
            ms_ssim_loss = ms_ssim(x_hat, target, data_range=1.)
            loss = self.lmbda * (1 - ms_ssim_loss) + bpp_loss
            return loss, ms_ssim_loss, bpp_loss

def collect_shared_params(model):
    """
    Extracts state_dict of shared components identified in train_5.py
    to send to the worker.
    """
    shared_prefixes = [
        'dt', 'dt_cross_attention', 'cc_mean_transforms', 
        'cc_scale_transforms', 'lrp_transforms', 'h_z_s1', 'h_z_s2',
        'entropy_bottleneck', 'gaussian_conditional'
    ]
    
    full_state = model.state_dict()
    shared_state = {}
    
    for key, value in full_state.items():
        # Check if key starts with any shared prefix
        if any(key.startswith(prefix) for prefix in shared_prefixes):
            shared_state[key] = value.cpu()
            
    return shared_state

def run_training(args):
    # --- 1. Init RPC ---
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        num_worker_threads=16,
        rpc_timeout=RPC_TIMEOUT,
        _transports=["uv"]
    )
    
    # If Master has GPU, map it to Worker GPU for direct transfer
    if args.cuda and torch.cuda.is_available():
        rpc_backend_options.device_maps = {"worker": {"cuda:0": "cuda:0"}}

    print(f"Initializing Master on {MASTER_ADDR}...")
    rpc.init_rpc("master", rank=0, world_size=2, rpc_backend_options=rpc_backend_options)

    # --- 2. Setup Models ---
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Local CompressModel on: {device}")
    
    compress_model = CompressModel(N=args.N, M=args.M).to(device)
    compress_model.train()

    print("Creating Remote Decompressor on Worker...")
    decompress_rref = rpc.remote(
        "worker",
        create_remote_decompressor,
        args=(args.N, args.M, "cuda:0"),
        timeout=RPC_TIMEOUT
    )

    # --- 3. Optimizers ---
    # Wrap local params in RRef for DistributedOptimizer
    local_params = [rpc.RRef(p) for p in compress_model.parameters() if p.requires_grad]
    # Fetch remote params RRefs
    remote_params = decompress_rref.rpc_sync().parameter_rrefs()
    
    # Combined Distributed Optimizer
    # This replaces the separate compress/decompress optimizers + gradient sync
    dist_optimizer = DistributedOptimizer(
        optim.Adam,
        local_params + remote_params,
        lr=args.learning_rate
    )

    # Local Aux Optimizer (Entropy Bottleneck) - Standard Optimizer
    compress_aux_optimizer = optim.Adam(
        (p for n, p in compress_model.named_parameters() if n.endswith(".quantiles")),
        lr=args.aux_learning_rate,
    )

    # --- 4. Dataset & Logging ---
    wandb.init(project=args.wandb_project, name=f"RPC_Split_{args.lmbda}")
    
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers, shuffle=True, pin_memory=True)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=args.type, device=device)

    # --- 5. Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        compress_model.train()
        
        for i, d in enumerate(train_dataloader):
            d = d.to(device)

            # --- Distributed Autograd ---
            with dist_autograd.context() as context_id:
                
                # A. Compress (Local)
                out_enc = compress_model(d)
                y_hat, z_hat = out_enc['y_hat'], out_enc['z_hat']
                
                # B. Decompress (Remote)
                # RPC handles sending y_hat/z_hat to worker and getting x_hat back
                x_hat = decompress_rref.rpc_sync().forward(y_hat, z_hat)
                
                # Ensure x_hat is on local device for loss calculation
                if x_hat.device != device:
                    x_hat = x_hat.to(device)

                # C. Loss
                loss, dist_loss, bpp_loss = criterion(x_hat, d, out_enc['likelihoods'])

                # D. Backward & Step
                dist_autograd.backward(context_id, [loss])
                dist_optimizer.step(context_id)

            # --- Aux Optimizer (Local) ---
            compress_aux_optimizer.zero_grad()
            aux_loss = compress_model.aux_loss()
            aux_loss.backward()
            compress_aux_optimizer.step()

            # --- Periodic Synchronization ---
            # Sync shared weights every 5 steps (same as train_5.py line 258)
            if (i + 1) % 5 == 0:
                # 1. Update entropy tables locally
                if (i + 1) % 50 == 0:
                    compress_model.update()
                    # Trigger remote update
                    decompress_rref.rpc_sync().update_entropy_tables()

                # 2. Sync shared parameters (mimics ParameterSync)
                # We collect local shared weights and force them onto the worker
                shared_weights = collect_shared_params(compress_model)
                decompress_rref.rpc_async().sync_shared_weights(shared_weights)

            if i % 10 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_dataloader)}] "
                      f"Loss: {loss.item():.4f} Bpp: {bpp_loss.item():.4f} Aux: {aux_loss.item():.2f}")
                wandb.log({
                    "train/loss": loss.item(),
                    "train/bpp": bpp_loss.item(),
                    "train/aux": aux_loss.item()
                })

        # Save Checkpoint
        if args.save:
            # We must fetch remote state to save a complete checkpoint
            remote_state = decompress_rref.rpc_sync().model.state_dict()
            torch.save({
                'epoch': epoch,
                'compress_model': compress_model.state_dict(),
                'decompress_model': remote_state,
            }, os.path.join(args.save_path, "checkpoint_rpc.pth.tar"))

    rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--save_path", type=str, default='./checkpoints_rpc')
    parser.add_argument("--type", type=str, default='mse')
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--M", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="RPC_Compression")
    
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    run_training(args)