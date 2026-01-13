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

def compute_psnr(a, b):
    """Calculates PSNR between two tensors (1, 3, H, W)"""
    mse = torch.mean((a - b)**2)
    if mse == 0:
        return float('inf')
    return -10 * torch.log10(mse)

def load_weights(model, checkpoint_path, key):
    """Loads specific model weights from the distributed checkpoint"""
    print(f"Loading {key} from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if key in checkpoint:
        state_dict = checkpoint[key]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    
    if hasattr(model, 'update'):
        model.update()
    print("Weights loaded successfully.")

# def run_compress(args):
#     device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
#     print(f"=== Compression Mode (Device: {device}) ===")
    
#     model = CompressModel(N=args.N, M=args.M).to(device)
#     model.eval()
#     load_weights(model, args.checkpoint, 'compress_model')
    
#     os.makedirs(args.output, exist_ok=True)
#     images = sorted(glob.glob(os.path.join(args.input, "*.*")))
#     images = [x for x in images if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     if not images:
#         print(f"No images found in {args.input}")
#         return

#     print(f"Found {len(images)} images.")
    
#     total_time = 0
#     with torch.no_grad():
#         for img_path in images:
#             # FIX: Use splitext to handle filenames with multiple dots correctly
#             filename = os.path.splitext(os.path.basename(img_path))[0]
#             x = read_image(img_path).to(device)
            
#             start = time.time()
#             out = model(x)
#             enc_time = time.time() - start
#             total_time += enc_time
            
#             compressed_data = {
#                 'y_hat': out['y_hat'].cpu(),
#                 'z_hat': out['z_hat'].cpu(),
#                 'original_shape': x.shape[-2:]
#             }
            
#             save_path = os.path.join(args.output, f"{filename}.bin")
#             torch.save(compressed_data, save_path)
            
#             print(f"Encoded: {filename} -> {save_path} ({enc_time*1000:.1f}ms)")

#     print(f"Finished. Average encoding time: {(total_time/len(images))*1000:.1f}ms")


# def run_decompress(args):
#     device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
#     print(f"=== Decompression Mode (Device: {device}) ===")
    
#     model = DecompressModel(N=args.N, M=args.M).to(device)
#     model.eval()
#     load_weights(model, args.checkpoint, 'decompress_model')
    
#     os.makedirs(args.output, exist_ok=True)
#     bin_files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
    
#     if not bin_files:
#         print(f"No .bin files found in {args.input}")
#         return

#     print(f"Found {len(bin_files)} compressed files.")
    
#     total_time = 0
#     psnrs = []
#     with torch.no_grad():
#         for bin_path in bin_files:
#             # FIX: Use splitext to handle filenames with multiple dots correctly
#             filename = os.path.splitext(os.path.basename(bin_path))[0]
            
#             data = torch.load(bin_path, map_location=device)
#             y_hat = data['y_hat']
#             z_hat = data['z_hat']
            
#             start = time.time()
#             out = model(y_hat, z_hat)
#             dec_time = time.time() - start
#             total_time += dec_time
            
#             x_hat = out['x_hat'].clamp(0, 1)
#             save_path = os.path.join(args.output, f"{filename}.png")
#             save_image(x_hat, save_path)
            
#             # PSNR Calculation
#             psnr_str = ""
#             if args.original:
#                 orig_path = None
#                 for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
#                     potential_path = os.path.join(args.original, filename + ext)
#                     if os.path.exists(potential_path):
#                         orig_path = potential_path
#                         break
                
#                 if orig_path:
#                     x_original = read_image(orig_path).to(device)
#                     # Handle padding (crop x_hat to original size if necessary)
#                     if x_original.shape != x_hat.shape:
#                         x_hat_eval = x_hat[:, :, :x_original.shape[2], :x_original.shape[3]]
#                     else:
#                         x_hat_eval = x_hat
                        
#                     curr_psnr = compute_psnr(x_hat_eval, x_original).item()
#                     psnrs.append(curr_psnr)
#                     psnr_str = f" | PSNR: {curr_psnr:.2f}dB"
#                 else:
#                     psnr_str = f" | PSNR: Original not found (checked {filename}.png/jpg)"

#             print(f"Decoded: {filename} -> {save_path} ({dec_time*1000:.1f}ms){psnr_str}")
            
#     print(f"\nFinished.")
#     print(f"Average decoding time: {(total_time/len(bin_files))*1000:.1f}ms")
#     if psnrs:
#         print(f"Average PSNR: {sum(psnrs)/len(psnrs):.2f}dB")



def run_compress(args):
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"=== Real Compression Mode (Device: {device}) ===")
    
    model = CompressModel(N=args.N, M=args.M).to(device)
    model.eval()
    load_weights(model, args.checkpoint, 'compress_model')
    
    # CRITICAL: Update entropy bottleneck CDFs before compression
    print("Updating entropy model CDFs...")
    model.update(force=True)
    
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
            filename = os.path.splitext(os.path.basename(img_path))[0]
            x = read_image(img_path).to(device)
            
            start = time.time()
            
            # --- CHANGE: Use .compress() instead of .forward() ---
            # Returns dictionary: {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
            out = model.compress(x) 
            
            enc_time = time.time() - start
            total_time += enc_time
            
            # Save the compressed bitstream structure
            save_path = os.path.join(args.output, f"{filename}.bin")
            torch.save(out, save_path)
            
            # Calculate file size in KB for reference
            file_size = os.path.getsize(save_path) / 1024
            print(f"Encoded: {filename} -> {save_path} ({enc_time*1000:.1f}ms) | Size: {file_size:.2f} KB")

    print(f"Finished. Average encoding time: {(total_time/len(images))*1000:.1f}ms")


def run_decompress(args):
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"=== Real Decompression Mode (Device: {device}) ===")
    
    model = DecompressModel(N=args.N, M=args.M).to(device)
    model.eval()
    load_weights(model, args.checkpoint, 'decompress_model')
    
    # CRITICAL: Update entropy bottleneck CDFs (needed for correct decoding)
    print("Updating entropy model CDFs...")
    model.update(force=True)
    
    os.makedirs(args.output, exist_ok=True)
    bin_files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
    
    if not bin_files:
        print(f"No .bin files found in {args.input}")
        return

    print(f"Found {len(bin_files)} compressed files.")
    
    total_time = 0
    psnrs = []
    
    with torch.no_grad():
        for bin_path in bin_files:
            filename = os.path.splitext(os.path.basename(bin_path))[0]
            
            # Load the bitstream dictionary
            compressed_data = torch.load(bin_path, map_location='cpu')
            
            strings = compressed_data['strings']
            shape = compressed_data['shape']
            
            start = time.time()
            
            # --- CHANGE: Use .decompress() instead of .forward() ---
            out = model.decompress(strings, shape)
            
            dec_time = time.time() - start
            total_time += dec_time
            
            x_hat = out['x_hat'].clamp(0, 1)
            save_path = os.path.join(args.output, f"{filename}.png")
            save_image(x_hat, save_path)
            
            # PSNR Calculation (same as before)
            psnr_str = ""
            if args.original:
                orig_path = None
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                    potential_path = os.path.join(args.original, filename + ext)
                    if os.path.exists(potential_path):
                        orig_path = potential_path
                        break
                
                if orig_path:
                    x_original = read_image(orig_path).to(device)
                    if x_original.shape != x_hat.shape:
                        x_hat_eval = x_hat[:, :, :x_original.shape[2], :x_original.shape[3]]
                    else:
                        x_hat_eval = x_hat
                    curr_psnr = compute_psnr(x_hat_eval, x_original).item()
                    psnrs.append(curr_psnr)
                    psnr_str = f" | PSNR: {curr_psnr:.2f}dB"

            print(f"Decoded: {filename} -> {save_path} ({dec_time*1000:.1f}ms){psnr_str}")
            
    print(f"\nFinished.")
    print(f"Average decoding time: {(total_time/len(bin_files))*1000:.1f}ms")
    if psnrs:
        print(f"Average PSNR: {sum(psnrs)/len(psnrs):.2f}dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Split Evaluation Script")
    parser.add_argument("--mode", choices=['compress', 'decompress'], help="Action to perform")
    parser.add_argument("-i", "--input", type=str, default='/home/xie/datasets/dummy/valid', help="Input folder")
    parser.add_argument("-o", "--output", type=str, default='./compressed', help="Output folder")
    parser.add_argument("--original", type=str, default=None, help="Original images path for PSNR")
    parser.add_argument("--checkpoint", type=str, default='./checkpoints_rpc/checkpoint_rpc.pth.tar', help="Weights path")
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--cuda", action="store_true", default=True)
    
    args = parser.parse_args()
    
    if args.mode == 'compress':
        run_compress(args)
    elif args.mode == 'decompress':
        run_decompress(args)