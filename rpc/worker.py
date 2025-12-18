import os
import torch
import torch.distributed.rpc as rpc
import argparse
import logging
import sys
from rpc_shared import MASTER_ADDR, MASTER_PORT

os.environ.setdefault('TP_SOCKET_IFNAME', 'enx00e04c6803a6')
os.environ.setdefault('GLOO_SOCKET_IFNAME', 'enx00e04c6803a6')

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("WorkerNode")

def run_worker(rank, world_size, master_addr, master_port, device):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://{master_addr}:{master_port}",
        num_worker_threads=16,
        rpc_timeout=600.0,
        _transports=["uv"]
    )

    if torch.cuda.is_available() and "cuda" in device:
        # --- FIX HERE: Wrap strings in torch.device() ---
        rpc_backend_options.device_maps = {
            "master": {torch.device(device): torch.device(device)} 
        }

    logger.info(f"Initializing RPC Worker on {device}...")
    
    rpc.init_rpc(
        name="worker",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options
    )

    logger.info("Worker ready. Waiting for Master commands...")
    rpc.shutdown()
    logger.info("RPC Shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_worker(args.rank, 2, MASTER_ADDR, MASTER_PORT, args.device)