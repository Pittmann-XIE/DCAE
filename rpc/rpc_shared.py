# --- START OF FILE rpc_shared.py ---
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dcae_5_fixed import DecompressModel, update_registered_buffers

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

MASTER_ADDR = "172.24.14.195" 
MASTER_PORT = "29500"
RPC_TIMEOUT = 600.0

SHARED_PREFIXES = [
    'dt', 'dt_cross_attention', 'cc_mean_transforms', 
    'cc_scale_transforms', 'lrp_transforms', 'h_z_s1', 'h_z_s2',
    'entropy_bottleneck', 'gaussian_conditional'
]

class RemoteDecompressor(nn.Module):
    def __init__(self, N, M, device_str="cuda:0"):
        super().__init__()
        self.device = torch.device(device_str) if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"[RemoteDecompressor] Initializing on {self.device}")
        
        self.model = DecompressModel(N=N, M=M)
        self.model.to(self.device)
        self.model.train()

    def set_mode(self, training=True):
        """Switch between Train and Eval mode"""
        if training:
            self.model.train()
        else:
            self.model.eval()
        return True

    def forward(self, y_hat, z_hat):
        """Standard Forward Pass (for Training/Fast Validation)"""
        if y_hat.device != self.device:
            y_hat = y_hat.to(self.device)
        if z_hat.device != self.device:
            z_hat = z_hat.to(self.device)
        
        out = self.model(y_hat, z_hat)
        return out["x_hat"]

    def decompress(self, strings, shape):
        """Real Decompression (Arithmetic Coding)"""
        # strings is a list of byte-strings sent from Master
        # shape is the tensor shape sent from Master
        with torch.no_grad():
            out = self.model.decompress(strings, shape)
        return out["x_hat"]

    def get_unique_parameters_rref(self):
        unique_params = []
        for name, p in self.model.named_parameters():
            if not any(name.startswith(prefix) for prefix in SHARED_PREFIXES):
                if p.requires_grad:
                    unique_params.append(RRef(p))
        return unique_params

    def get_shared_gradients(self):
        grads = {}
        for name, p in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in SHARED_PREFIXES):
                if p.grad is not None:
                    grads[name] = p.grad.detach().cpu()
        return grads

    def zero_shared_grads(self):
        for name, p in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in SHARED_PREFIXES):
                p.grad = None

    def sync_shared_weights(self, shared_state_dict):
        try:
            self.model.load_state_dict(shared_state_dict, strict=False)
            update_registered_buffers(
                self.model.gaussian_conditional,
                "gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                shared_state_dict,
            )
            return True
        except Exception as e:
            logger.error(f"Error syncing shared weights: {e}")
            return False

    def update_entropy_tables(self):
        self.model.update()
        return True
    
    def load_full_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        return True

    def get_state_dict(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

def create_remote_decompressor(N, M, device):
    return RemoteDecompressor(N, M, device)