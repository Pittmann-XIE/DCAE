import torch
import torch.nn as nn
from torch.distributed.rpc import RRef
import logging
import sys
import os

# Assumes your project structure has 'models/dcae_5.py'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dcae_5 import DecompressModel

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- CONFIGURATION: UPDATE THESE FOR YOUR NETWORK ---
MASTER_ADDR = "172.24.14.195"  # <--- CHANGE TO MASTER IP
MASTER_PORT = "29500"
RPC_TIMEOUT = 600.0
# ----------------------------------------------------

# Parameters shared between Compressor (Master) and Decompressor (Worker)
# The Master owns these. The Worker just borrows them.
SHARED_PREFIXES = [
    'dt', 'dt_cross_attention', 'cc_mean_transforms', 
    'cc_scale_transforms', 'lrp_transforms', 'h_z_s1', 'h_z_s2',
    'entropy_bottleneck', 'gaussian_conditional'
]

def get_device(device_str):
    if device_str.lower() == 'cpu':
        return torch.device('cpu')
    return torch.device(device_str)

class RemoteDecompressor(nn.Module):
    def __init__(self, N, M, device_str="cuda:0"):
        super().__init__()
        self.device = get_device(device_str)
        logger.info(f"[RemoteDecompressor] Initializing on {self.device}")
        
        # The Worker runs the Decompressor (Heavy GPU load)
        self.model = DecompressModel(N=N, M=M)
        self.model.to(self.device)
        self.model.train()

    def forward(self, y_hat, z_hat):
        # Ensure inputs are on the correct GPU device
        if y_hat.device != self.device:
            y_hat = y_hat.to(self.device)
        if z_hat.device != self.device:
            z_hat = z_hat.to(self.device)
        
        out = self.model(y_hat, z_hat)
        return out["x_hat"]

    def parameter_rrefs(self):
        """
        Returns RRefs ONLY for parameters unique to the Decompressor (g_s).
        We EXCLUDE shared parameters because the Master manages them manually.
        """
        unique_params = []
        for name, p in self.model.named_parameters():
            if not any(name.startswith(prefix) for prefix in SHARED_PREFIXES):
                if p.requires_grad:
                    unique_params.append(RRef(p))
        return unique_params

    def get_shared_gradients(self):
        """
        Critical for Split Training:
        Collects gradients calculated on the Worker (Distortion loss) for the 
        Shared Parameters and sends them to Master to combine with Rate loss.
        """
        grads = {}
        for name, p in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in SHARED_PREFIXES):
                if p.grad is not None:
                    # Move to CPU for RPC transfer
                    grads[name] = p.grad.detach().cpu()
        return grads

    def zero_shared_grads(self):
        """Clean up gradients on Worker after Master has fetched them."""
        for name, p in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in SHARED_PREFIXES):
                p.grad = None

    def sync_shared_weights(self, shared_state_dict):
        """
        Forces Worker to use Master's weights. 
        This prevents 'Split Brain' where Encoder and Decoder drift apart.
        """
        try:
            self.model.load_state_dict(shared_state_dict, strict=False)
            return True
        except Exception as e:
            logger.error(f"Error syncing shared weights: {e}")
            return False

    def load_full_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True) 
        return True

    def update_entropy_tables(self):
        self.model.update()
        return True

    def get_model_state_dict(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

def create_remote_decompressor(N, M, device):
    return RemoteDecompressor(N, M, device)