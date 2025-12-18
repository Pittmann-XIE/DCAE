# rpc_shared.py (Updated)
import torch
import torch.nn as nn
from torch.distributed.rpc import RRef
import logging
import sys

# Assumes models/dcae_5.py exists locally
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dcae_5 import DecompressModel

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

MASTER_ADDR = "172.24.14.195" 
MASTER_PORT = "29500"
RPC_TIMEOUT = 600.0

def get_device(device_str):
    if device_str.lower() == 'cpu':
        return torch.device('cpu')
    return torch.device(device_str)

class RemoteDecompressor(nn.Module):
    def __init__(self, N, M, device_str="cuda:0"):
        super().__init__()
        self.device = get_device(device_str)
        logger.info(f"[RemoteDecompressor] Initializing on {self.device}")
        
        self.model = DecompressModel(N=N, M=M)
        self.model.to(self.device)
        self.model.train()

    def forward(self, y_hat, z_hat):
        if y_hat.device != self.device:
            y_hat = y_hat.to(self.device)
        if z_hat.device != self.device:
            z_hat = z_hat.to(self.device)
        
        out = self.model(y_hat, z_hat)
        return out["x_hat"]

    def parameter_rrefs(self):
        return [RRef(p) for p in self.model.parameters()]

    def sync_shared_weights(self, shared_state_dict):
        try:
            self.model.load_state_dict(shared_state_dict, strict=False)
            return True
        except Exception as e:
            logger.error(f"Error syncing shared weights: {e}")
            return False

    def load_full_state_dict(self, state_dict):
        try:
            logger.info(f"[RemoteDecompressor] Loading state dict with {len(state_dict)} keys...")
            self.model.load_state_dict(state_dict, strict=True) 
            logger.info("[RemoteDecompressor] Weights loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"[RemoteDecompressor] Failed to load weights: {e}")
            raise e

    def update_entropy_tables(self):
        self.model.update()
        return True

    # --- NEW METHOD HERE ---
    def get_model_state_dict(self):
        """Helper to return the internal model state dict via RPC method call."""
        # Move to CPU to save RPC bandwidth and avoid CUDA device mismatches on return
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

def create_remote_decompressor(N, M, device):
    return RemoteDecompressor(N, M, device)