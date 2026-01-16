# import os
# import sys
# import argparse
# import torch
# import glob
# import time
# from torchvision import transforms
# from PIL import Image

# # Add project root to path to find 'models'
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from models.dcae_5 import CompressModel, DecompressModel

# def read_image(filepath):
#     """Reads image and returns tensor (1, 3, H, W)"""
#     img = Image.open(filepath).convert('RGB')
#     return transforms.ToTensor()(img).unsqueeze(0)

# def save_image(tensor, filepath):
#     """Saves tensor (1, 3, H, W) to image file"""
#     img = transforms.ToPILImage()(tensor.squeeze(0).cpu().clamp(0, 1))
#     img.save(filepath)

# def compute_psnr(a, b):
#     """Calculates PSNR between two tensors (1, 3, H, W)"""
#     mse = torch.mean((a - b)**2)
#     if mse == 0:
#         return float('inf')
#     return -10 * torch.log10(mse)

# def load_weights(model, checkpoint_path, key):
#     """Loads specific model weights from the distributed checkpoint"""
#     print(f"Loading {key} from {checkpoint_path}...")
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
#     if key in checkpoint:
#         state_dict = checkpoint[key]
#     else:
#         state_dict = checkpoint
        
#     model.load_state_dict(state_dict)
    
#     if hasattr(model, 'update'):
#         model.update()
#     print("Weights loaded successfully.")

# # def run_compress(args):
# #     device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
# #     print(f"=== Real Compression Mode (Device: {device}) ===")
    
# #     model = CompressModel(N=args.N, M=args.M).to(device)
# #     model.eval()
# #     load_weights(model, args.checkpoint, 'compress_model')
    
# #     # CRITICAL: Update entropy bottleneck CDFs before compression
# #     print("Updating entropy model CDFs...")
# #     model.update(force=True)
    
# #     os.makedirs(args.output, exist_ok=True)
# #     images = sorted(glob.glob(os.path.join(args.input, "*.*")))
# #     images = [x for x in images if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
# #     if not images:
# #         print(f"No images found in {args.input}")
# #         return

# #     print(f"Found {len(images)} images.")
    
# #     total_time = 0
# #     with torch.no_grad():
# #         for img_path in images:
# #             filename = os.path.splitext(os.path.basename(img_path))[0]
# #             x = read_image(img_path).to(device)
            
# #             start = time.time()
            
# #             # --- CHANGE: Use .compress() instead of .forward() ---
# #             # Returns dictionary: {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
# #             out = model.compress(x) 
            
# #             enc_time = time.time() - start
# #             total_time += enc_time
            
# #             # Save the compressed bitstream structure
# #             save_path = os.path.join(args.output, f"{filename}.bin")
# #             torch.save(out, save_path)
            
# #             # Calculate file size in KB for reference
# #             file_size = os.path.getsize(save_path) / 1024
# #             print(f"Encoded: {filename} -> {save_path} ({enc_time*1000:.1f}ms) | Size: {file_size:.2f} KB")

# #     print(f"Finished. Average encoding time: {(total_time/len(images))*1000:.1f}ms")


# # def run_decompress(args):
# #     device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
# #     print(f"=== Real Decompression Mode (Device: {device}) ===")
    
# #     model = DecompressModel(N=args.N, M=args.M).to(device)
# #     model.eval()
# #     load_weights(model, args.checkpoint, 'decompress_model')
    
# #     # CRITICAL: Update entropy bottleneck CDFs (needed for correct decoding)
# #     print("Updating entropy model CDFs...")
# #     model.update(force=True)
    
# #     os.makedirs(args.output, exist_ok=True)
# #     bin_files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
    
# #     if not bin_files:
# #         print(f"No .bin files found in {args.input}")
# #         return

# #     print(f"Found {len(bin_files)} compressed files.")
    
# #     total_time = 0
# #     psnrs = []
    
# #     with torch.no_grad():
# #         for bin_path in bin_files:
# #             filename = os.path.splitext(os.path.basename(bin_path))[0]
            
# #             # Load the bitstream dictionary
# #             compressed_data = torch.load(bin_path, map_location='cpu')
            
# #             strings = compressed_data['strings']
# #             shape = compressed_data['shape']
            
# #             start = time.time()
            
# #             # --- CHANGE: Use .decompress() instead of .forward() ---
# #             out = model.decompress(strings, shape)
            
# #             dec_time = time.time() - start
# #             total_time += dec_time
            
# #             x_hat = out['x_hat'].clamp(0, 1)
# #             save_path = os.path.join(args.output, f"{filename}.png")
# #             save_image(x_hat, save_path)
            
# #             # PSNR Calculation (same as before)
# #             psnr_str = ""
# #             if args.original:
# #                 orig_path = None
# #                 for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
# #                     potential_path = os.path.join(args.original, filename + ext)
# #                     if os.path.exists(potential_path):
# #                         orig_path = potential_path
# #                         break
                
# #                 if orig_path:
# #                     x_original = read_image(orig_path).to(device)
# #                     if x_original.shape != x_hat.shape:
# #                         x_hat_eval = x_hat[:, :, :x_original.shape[2], :x_original.shape[3]]
# #                     else:
# #                         x_hat_eval = x_hat
# #                     curr_psnr = compute_psnr(x_hat_eval, x_original).item()
# #                     psnrs.append(curr_psnr)
# #                     psnr_str = f" | PSNR: {curr_psnr:.2f}dB"

# #             print(f"Decoded: {filename} -> {save_path} ({dec_time*1000:.1f}ms){psnr_str}")
            
# #     print(f"\nFinished.")
# #     print(f"Average decoding time: {(total_time/len(bin_files))*1000:.1f}ms")
# #     if psnrs:
# #         print(f"Average PSNR: {sum(psnrs)/len(psnrs):.2f}dB")



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



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Manual Split Evaluation Script")
#     parser.add_argument("--mode", choices=['compress', 'decompress'], help="Action to perform")
#     parser.add_argument("-i", "--input", type=str, default='/home/xie/datasets/dummy/valid', help="Input folder")
#     parser.add_argument("-o", "--output", type=str, default='./compressed', help="Output folder")
#     parser.add_argument("--original", type=str, default=None, help="Original images path for PSNR")
#     parser.add_argument("--checkpoint", type=str, default='./checkpoints_rpc/checkpoint_rpc_wrong.pth.tar', help="Weights path")
#     parser.add_argument("--N", type=int, default=192)
#     parser.add_argument("--M", type=int, default=320)
#     parser.add_argument("--cuda", action="store_true", default=True)
    
#     args = parser.parse_args()
    
#     if args.mode == 'compress':
#         run_compress(args)
#     elif args.mode == 'decompress':
#         run_decompress(args)


## load tables from master weights when decompressing
import os
import sys
import argparse
import torch
import glob
import time
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Note: Ensure you import the fixed model file where `update_registered_buffers` is defined
from models.dcae_5_fixed import CompressModel, DecompressModel, update_registered_buffers

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

def read_image(filepath):
    img = Image.open(filepath).convert('RGB')
    return transforms.ToTensor()(img).unsqueeze(0)

def save_image(tensor, filepath):
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu().clamp(0, 1))
    img.save(filepath)

def load_weights_smart(model, checkpoint_path, mode='compress'):
    """
    Smart loading that ensures the Decoder gets the correct CDF tables 
    from the Encoder (Master) state, because the Worker state might be desynced.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 1. Determine which state_dict to use as the SOURCE of truth
    # We always prefer 'compress_model' for probability tables because Master is the authority.
    if 'compress_model' in checkpoint:
        master_state = checkpoint['compress_model']
        worker_state = checkpoint.get('decompress_model', checkpoint['compress_model'])
    else:
        # Legacy/Unified checkpoint
        master_state = checkpoint
        worker_state = checkpoint

    # 2. Load Parameters
    if mode == 'compress':
        model.load_state_dict(master_state, strict=False)
    else:
        # For DECOMPRESS mode, we load the Worker's weights (for the decoder layers)
        # BUT we must overwrite the Entropy Tables with the Master's tables.
        model.load_state_dict(worker_state, strict=False)
        
        print("Overwriting Entropy Tables with Master's tables to ensure sync...")
        # Sync Gaussian Conditional tables
        update_registered_buffers(
            model.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            master_state,
        )
        # Sync Entropy Bottleneck tables
        update_registered_buffers(
            model.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            master_state,
        )

    # 3. CRITICAL: Do NOT call model.update() here.
    # We want to use the exactly saved tables, not recalculate them.
    print("Weights and CDF tables loaded successfully.")

def run_compress_real(args):
    """
    Actual Compression generating .bin files (Byte Strings)
    """
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"=== Real Compression Mode (Device: {device}) ===")
    
    model = CompressModel(N=args.N, M=args.M).to(device)
    model.eval()
    
    # Load weights without re-running update()
    load_weights_smart(model, args.checkpoint, mode='compress')
    
    os.makedirs(args.output, exist_ok=True)
    images = sorted(glob.glob(os.path.join(args.input, "*.*")))
    images = [x for x in images if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    total_time = 0
    with torch.no_grad():
        for img_path in images:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            x = read_image(img_path).to(device)
            
            start = time.time()
            
            # --- REAL COMPRESSION CALL ---
            # Returns: {"strings": [[y_strings], [z_strings]], "shape": z.shape}
            out_enc = model.compress(x) 
            
            enc_time = time.time() - start
            total_time += enc_time
            
            # Save the binary dictionary
            save_path = os.path.join(args.output, f"{filename}.bin")
            torch.save(out_enc, save_path)
            
            # Calculate size in KB
            # strings is a list of lists. Flatten it to count bytes.
            total_bytes = 0
            for stream_list in out_enc["strings"]:
                for s in stream_list:
                    total_bytes += len(s)
            
            print(f"Encoded: {filename} -> {save_path} | Size: {total_bytes/1024:.2f} KB | Time: {enc_time*1000:.1f}ms")

def run_decompress_real(args):
    """
    Actual Decompression from .bin files
    """
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"=== Real Decompression Mode (Device: {device}) ===")
    
    model = DecompressModel(N=args.N, M=args.M).to(device)
    model.eval()
    
    # Load weights AND force sync tables from Master
    load_weights_smart(model, args.checkpoint, mode='decompress')
    
    os.makedirs(args.output, exist_ok=True)
    bin_files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
    
    total_time = 0
    with torch.no_grad():
        for bin_path in bin_files:
            filename = os.path.splitext(os.path.basename(bin_path))[0]
            
            # Load the dictionary containing strings and shape
            compressed_data = torch.load(bin_path, map_location='cpu')
            
            start = time.time()
            
            # --- REAL DECOMPRESSION CALL ---
            # Inputs: strings (list), shape (tuple)
            out = model.decompress(compressed_data["strings"], compressed_data["shape"])
            
            dec_time = time.time() - start
            total_time += dec_time
            
            x_hat = out["x_hat"].clamp(0, 1)
            save_path = os.path.join(args.output, f"{filename}.png")
            save_image(x_hat, save_path)
            
            print(f"Decoded: {filename} -> {save_path} ({dec_time*1000:.1f}ms)")
            
    print(f"Finished. Average decoding time: {(total_time/len(bin_files))*1000:.1f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['compress', 'decompress'], required=True)
    parser.add_argument("-i", "--input", type=str, default='./compressed')
    parser.add_argument("-o", "--output", type=str, default='./decompressed')
    parser.add_argument("--checkpoint", type=str, default='./checkpoints_rpc/checkpoint_rpc_wrong.pth.tar')
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--cuda", action="store_true", default=True)
    
    args = parser.parse_args()
    
    if args.mode == 'compress':
        run_compress_real(args)
    elif args.mode == 'decompress':
        run_decompress_real(args)