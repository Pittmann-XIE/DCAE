import os
import torch
import torch.nn as nn
from torch.distributed.rpc import RRef
import logging
import sys

# Assumes models/dcae_5.py exists on both machines
import sys
import os
# Add the project root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dcae_5 import DecompressModel

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Configuration ---
MASTER_ADDR = "172.24.14.195"  # Update with your Master IP
MASTER_PORT = "29500"
RPC_TIMEOUT = 600.0

def get_device(device_str):
    if device_str.lower() == 'cpu':
        return torch.device('cpu')
    return torch.device(device_str)

# --- Remote Module Wrapper ---
class RemoteDecompressor(nn.Module):
    """
    Wraps the DecompressModel on the Worker.
    Handles forward pass and parameter synchronization.
    """
    def __init__(self, N, M, device_str="cuda:0"):
        super().__init__()
        self.device = get_device(device_str)
        logger.info(f"[RemoteDecompressor] Initializing on {self.device} with N={N}, M={M}")
        
        self.model = DecompressModel(N=N, M=M)
        self.model.to(self.device)
        self.model.train()

        # List of shared components identified in train_5.py
        self.shared_components = [
            'dt', 'dt_cross_attention', 'cc_mean_transforms', 
            'cc_scale_transforms', 'lrp_transforms', 'h_z_s1', 'h_z_s2',
            'entropy_bottleneck', 'gaussian_conditional'
        ]

    def forward(self, y_hat, z_hat):
        # Ensure inputs are on the correct device
        if y_hat.device != self.device:
            y_hat = y_hat.to(self.device)
        if z_hat.device != self.device:
            z_hat = z_hat.to(self.device)
        
        out = self.model(y_hat, z_hat)
        return out["x_hat"]

    def parameter_rrefs(self):
        """Returns RRefs of parameters for DistributedOptimizer"""
        return [RRef(p) for p in self.model.parameters()]

    def sync_shared_weights(self, shared_state_dict):
        """
        Receives a dictionary of weights from the Master's CompressModel
        and updates the corresponding local DecompressModel components.
        """
        try:
            # We only load the keys present in shared_state_dict
            # strict=False allows loading partial state
            self.model.load_state_dict(shared_state_dict, strict=False)
            return True
        except Exception as e:
            logger.error(f"Error syncing shared weights: {e}")
            return False

    def update_entropy_tables(self):
        """
        Calls .update() on the decompressor (needed for inference/val).
        """
        self.model.update()
        return True

# --- Factory Function ---
def create_remote_decompressor(N, M, device):
    return RemoteDecompressor(N, M, device)