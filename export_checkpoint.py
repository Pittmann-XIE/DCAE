# export_checkpoint.py
import torch
import os
import argparse
from models import DCAE

def parse_args():
    parser = argparse.ArgumentParser(description="Bake CDF tables into checkpoint")
    parser.add_argument("--input", type=str, default='/home/xie/DCAE/checkpoints/dace_7/60.5checkpoint_best.pth.tar', help="Path to input checkpoint (e.g., 60.5checkpoint_best.pth.tar)")
    parser.add_argument("--output", type=str, default="checkpoint_7_gpu_0dot003.pth", help="Path to save the new checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda'
    # 1. Load the model structure
    print("Creating model...")
    net = DCAE()
    net = net.to(device)
    
    # 2. Load the original weights (which have empty/zeros for CDF tables)
    print(f"Loading weights from {args.input}...")
    checkpoint = torch.load(args.input, map_location=device)
    
    # Handle the 'module.' prefix if it exists from DataParallel training
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
        
    net.load_state_dict(new_state_dict)

    # 3. FORCE UPDATE
    # This performs the floating-point calculations to generate the CDF tables
    # using Machine A's specific hardware/driver math.
    print("Updating CDF tables (Baking)...")
    net.update(force=True)

    # 4. Save the new state_dict
    # This new file now contains the Weights AND the calculated CDFs
    print(f"Saving baked model to {args.output}...")
    torch.save({"state_dict": net.state_dict()}, args.output)
    print("Done! Transfer this new checkpoint to Machine B.")

if __name__ == "__main__":
    main()