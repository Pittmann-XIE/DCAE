import os
import sys
import argparse
import torch
import glob
import time
from torchvision import transforms
from PIL import Image

# Add project root to path to find 'models'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dcae_5 import CompressModel, DecompressModel

def read_image(filepath):
    """Reads image and returns tensor (1, 3, H, W)"""
    img = Image.open(filepath).convert('RGB')
    return transforms.ToTensor()(img).unsqueeze(0)

def save_image(tensor, filepath):
    """Saves tensor (1, 3, H, W) to image file"""
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu().clamp(0, 1))
    img.save(filepath)

def load_weights(model, checkpoint_path, key):
    """Loads specific model weights from the distributed checkpoint"""
    print(f"Loading {key} from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if key in checkpoint:
        # Checkpoint from master.py (contains 'compress_model', 'decompress_model', etc.)
        state_dict = checkpoint[key]
    else:
        # Fallback/Legacy logic
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    
    # Update entropy tables (crucial for inference stability)
    if hasattr(model, 'update'):
        model.update()
    print("Weights loaded successfully.")

def run_compress(args):
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"=== Compression Mode (Device: {device}) ===")
    
    # 1. Initialize Compressor
    model = CompressModel(N=args.N, M=args.M).to(device)
    model.eval()
    
    # 2. Load Weights
    load_weights(model, args.checkpoint, 'compress_model')
    
    # 3. Process Images
    os.makedirs(args.output, exist_ok=True)
    images = sorted(glob.glob(os.path.join(args.input, "*.*")))
    images = [x for x in images if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"No images found in {args.input}")
        return

    print(f"Found {len(images)} images.")
    
    total_time = 0
    with torch.no_grad():
        for img_path in images:
            filename = os.path.basename(img_path).split('.')[0]
            x = read_image(img_path).to(device)
            
            start = time.time()
            # Forward pass to get latents
            out = model(x)
            enc_time = time.time() - start
            total_time += enc_time
            
            # Prepare data package for transfer
            # We save the quantized latents (y_hat, z_hat)
            # In a full production system, you would save bitstreams here.
            compressed_data = {
                'y_hat': out['y_hat'].cpu(),
                'z_hat': out['z_hat'].cpu(),
                'original_shape': x.shape[-2:] # Save H, W for reference
            }
            
            save_path = os.path.join(args.output, f"{filename}.bin")
            torch.save(compressed_data, save_path)
            
            print(f"Encoded: {filename} -> {save_path} ({enc_time*1000:.1f}ms)")

    print(f"Finished. Average encoding time: {(total_time/len(images))*1000:.1f}ms")


def run_decompress(args):
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"=== Decompression Mode (Device: {device}) ===")
    
    # 1. Initialize Decompressor
    model = DecompressModel(N=args.N, M=args.M).to(device)
    model.eval()
    
    # 2. Load Weights
    load_weights(model, args.checkpoint, 'decompress_model')
    
    # 3. Process Bin Files
    os.makedirs(args.output, exist_ok=True)
    bin_files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
    
    if not bin_files:
        print(f"No .bin files found in {args.input}")
        return

    print(f"Found {len(bin_files)} compressed files.")
    
    total_time = 0
    with torch.no_grad():
        for bin_path in bin_files:
            filename = os.path.basename(bin_path).split('.')[0]
            
            # Load compressed data
            data = torch.load(bin_path, map_location=device)
            y_hat = data['y_hat']
            z_hat = data['z_hat']
            
            start = time.time()
            # Decode
            out = model(y_hat, z_hat)
            dec_time = time.time() - start
            total_time += dec_time
            
            # Save Reconstruction
            save_path = os.path.join(args.output, f"{filename}.png")
            save_image(out['x_hat'], save_path)
            
            print(f"Decoded: {filename} -> {save_path} ({dec_time*1000:.1f}ms)")
            
    print(f"Finished. Average decoding time: {(total_time/len(bin_files))*1000:.1f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Split Evaluation Script")
    parser.add_argument("--mode", choices=['compress', 'decompress'], 
                        help="Action to perform")
    parser.add_argument("-i", "--input", type=str, default='/home/xie/datasets/dummy/valid', 
                        help="Input folder (Images for 'compress', .bin files for 'decompress')")
    parser.add_argument("-o", "--output", type=str, default='./compressed', 
                        help="Output folder")
    parser.add_argument("--checkpoint", type=str, default='./checkpoints_rpc/checkpoint_rpc.pth.tar', 
                        help="Path to checkpoint (saved by master.py)")
    
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--cuda", action="store_true", default=True)
    
    args = parser.parse_args()
    
    if args.mode == 'compress':
        run_compress(args)
    elif args.mode == 'decompress':
        run_decompress(args)