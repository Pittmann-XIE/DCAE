import os
import sys
import argparse
import math
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim
from models.dcae_5 import CompressModel
from models import DCAE 

from rpc_shared import (
    MASTER_ADDR, MASTER_PORT, create_remote_decompressor, 
    RPC_TIMEOUT
)

os.environ.setdefault('TP_SOCKET_IFNAME', 'eno8303')
os.environ.setdefault('GLOO_SOCKET_IFNAME', 'eno8303')

def load_pretrained_checkpoint(compress_model, decompress_rref, checkpoint_path, is_unified=False):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if is_unified:
        print("-> Type: Unified DCAE Checkpoint")
        unified_state_dict = {}
        raw_state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        for k, v in raw_state.items():
            unified_state_dict[k.replace("module.", "")] = v
        
        temp_unified = DCAE()
        temp_unified.load_state_dict(unified_state_dict)
        
        print("-> Loading Compressor weights locally...")
        compress_model.g_a.load_state_dict(temp_unified.g_a.state_dict())
        compress_model.h_a.load_state_dict(temp_unified.h_a.state_dict())
        
        print("-> Preparing Decompressor weights for RPC...")
        decompress_state_dict = {}
        
        for k, v in temp_unified.g_s.state_dict().items():
            decompress_state_dict[f"g_s.{k}"] = v
            
        shared_components = [
            'dt', 'h_z_s1', 'h_z_s2', 'cc_mean_transforms', 
            'cc_scale_transforms', 'lrp_transforms', 'dt_cross_attention',
            'entropy_bottleneck', 'gaussian_conditional'
        ]
        
        for comp in shared_components:
            if hasattr(compress_model, comp):
                if comp == 'dt':
                    getattr(compress_model, comp).data = getattr(temp_unified, comp).data.clone()
                else:
                    getattr(compress_model, comp).load_state_dict(getattr(temp_unified, comp).state_dict())
            
            if hasattr(temp_unified, comp):
                obj = getattr(temp_unified, comp)
                if comp == 'dt':
                    decompress_state_dict['dt'] = obj.data.clone()
                else:
                    comp_state = obj.state_dict()
                    for k, v in comp_state.items():
                        decompress_state_dict[f"{comp}.{k}"] = v

        print("-> Sending weights to Worker via RPC...")
        decompress_rref.rpc_sync().load_full_state_dict(decompress_state_dict)
        print("-> Checkpoint loaded successfully.")
        del temp_unified

    else:
        print("-> Type: Split/RPC Checkpoint")
        if 'compress_model' in checkpoint:
            compress_model.load_state_dict(checkpoint['compress_model'])
            decompress_rref.rpc_sync().load_full_state_dict(checkpoint['decompress_model'])
        else:
            raise ValueError("Unknown checkpoint format")

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
        bpp_loss = sum((torch.log(l).sum() / (-math.log(2) * num_pixels)) for l in likelihoods.values())
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
    shared_prefixes = [
        'dt', 'dt_cross_attention', 'cc_mean_transforms', 
        'cc_scale_transforms', 'lrp_transforms', 'h_z_s1', 'h_z_s2',
        'entropy_bottleneck', 'gaussian_conditional'
    ]
    full_state = model.state_dict()
    shared_state = {}
    for key, value in full_state.items():
        if any(key.startswith(prefix) for prefix in shared_prefixes):
            shared_state[key] = value.cpu()
    return shared_state

def run_training(args):
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        num_worker_threads=16,
        rpc_timeout=RPC_TIMEOUT,
        _transports=["uv"]
    )
    
    if args.cuda and torch.cuda.is_available():
        rpc_backend_options.device_maps = {
            "worker": {torch.device("cuda:0"): torch.device("cuda:0")}
        }

    print(f"Initializing Master on {MASTER_ADDR}...")
    rpc.init_rpc("master", rank=0, world_size=2, rpc_backend_options=rpc_backend_options)

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

    if args.checkpoint:
        load_pretrained_checkpoint(
            compress_model, 
            decompress_rref, 
            args.checkpoint, 
            is_unified=args.load_pretrained_checkpoint
        )
    
    local_params = [rpc.RRef(p) for p in compress_model.parameters() if p.requires_grad]
    remote_params = decompress_rref.rpc_sync().parameter_rrefs()
    
    dist_optimizer = DistributedOptimizer(
        optim.Adam,
        local_params + remote_params,
        lr=args.learning_rate
    )

    compress_aux_optimizer = optim.Adam(
        (p for n, p in compress_model.named_parameters() if n.endswith(".quantiles")),
        lr=args.aux_learning_rate,
    )

    wandb.init(project=args.wandb_project, name=f"RPC_Split_{args.lmbda}")
    
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers, shuffle=True, pin_memory=True)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=args.type, device=device)

    print("Starting training...")
    for epoch in range(args.epochs):
        compress_model.train()
        
        for i, d in enumerate(train_dataloader):
            d = d.to(device)
            
            with dist_autograd.context() as context_id:
                out_enc = compress_model(d)
                y_hat, z_hat = out_enc['y_hat'], out_enc['z_hat']
                
                x_hat = decompress_rref.rpc_sync().forward(y_hat, z_hat)
                if x_hat.device != device:
                    x_hat = x_hat.to(device)

                loss, dist_loss, bpp_loss = criterion(x_hat, d, out_enc['likelihoods'])

                dist_autograd.backward(context_id, [loss])
                dist_optimizer.step(context_id)

            compress_aux_optimizer.zero_grad()
            aux_loss = compress_model.aux_loss()
            aux_loss.backward()
            compress_aux_optimizer.step()

            if (i + 1) % 5 == 0:
                if (i + 1) % 50 == 0:
                    compress_model.update()
                    decompress_rref.rpc_sync().update_entropy_tables()

                shared_weights = collect_shared_params(compress_model)
                decompress_rref.rpc_sync().sync_shared_weights(shared_weights)

            if i % 10 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_dataloader)}] "
                      f"Loss: {loss.item():.4f} Bpp: {bpp_loss.item():.4f} Aux: {aux_loss.item():.2f}")
                wandb.log({
                    "train/loss": loss.item(),
                    "train/bpp": bpp_loss.item(),
                    "train/dist_loss": dist_loss.item(),
                    "train/aux": aux_loss.item()
                })

        if args.save:
            # --- FIX HERE: Use method call instead of attribute access ---
            remote_state = decompress_rref.rpc_sync().get_model_state_dict()
            torch.save({
                'epoch': epoch,
                'compress_model': compress_model.state_dict(),
                'decompress_model': remote_state,
            }, os.path.join(args.save_path, "checkpoint_rpc.pth.tar"))

    rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='../dataset')
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
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="RPC_Compression")
    parser.add_argument("--checkpoint", type=str, default='../60.5checkpoint_best.pth.tar')
    parser.add_argument("--load_pretrained_checkpoint", action="store_true", default=True)

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    run_training(args)