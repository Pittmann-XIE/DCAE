# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models.g_a_g_s import SimpleAutoencoder
# import warnings
# import torch
# import os
# import sys
# import math
# import argparse
# import time
# import warnings
# from pytorch_msssim import ms_ssim
# from PIL import Image
# from thop import profile
# warnings.filterwarnings("ignore")
# torch.set_num_threads(10)

# print(torch.cuda.is_available())

# def save_image(tensor, filename):
#     img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
#     img.save(filename)

# def save_metrics(filename, psnr, msssim, compression_ratio=None):
#     with open(filename, 'w') as f:
#         f.write(f'PSNR: {psnr:.2f}dB\n')
#         f.write(f'MS-SSIM: {msssim:.4f}\n')
#         if compression_ratio is not None:
#             f.write(f'Compression Ratio: {compression_ratio:.2f}\n')

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2, dim=[1,2,3])  # Compute per image in batch
#     return -10 * torch.log10(mse)

# def compute_msssim(a, b):
#     # Process each image in the batch
#     msssim_values = []
#     for i in range(a.size(0)):
#         msssim_val = ms_ssim(a[i:i+1], b[i:i+1], data_range=1.).item()
#         msssim_values.append(-10 * math.log10(1 - msssim_val))
#     return torch.tensor(msssim_values)

# def compute_compression_ratio(x, latent):
#     """Compute approximate compression ratio based on tensor sizes"""
#     original_size = x.numel() * 8  # Assuming 8-bit integers
#     compressed_size = latent.numel() * 32  # Assuming 32-bit floats
#     return original_size / compressed_size


# def compute_size_analysis(x, latent):
#     """Compute detailed size analysis between original and latent"""
#     # Original image analysis
#     orig_h, orig_w = x.shape[2], x.shape[3]
#     orig_channels = x.shape[1]
#     orig_pixels = orig_h * orig_w * orig_channels
#     orig_size_bits = orig_pixels * 8  # int8 = 8 bits per pixel
#     orig_size_mb = orig_size_bits / (8 * 1024 * 1024)  # Convert to MB
    
#     # Latent analysis
#     latent_h, latent_w = latent.shape[2], latent.shape[3]
#     latent_channels = latent.shape[1]
#     latent_values = latent_h * latent_w * latent_channels
#     latent_size_bits = latent_values * 32  # float32 = 32 bits per value
#     latent_size_mb = latent_size_bits / (8 * 1024 * 1024)  # Convert to MB
    
#     # Spatial compression
#     spatial_reduction = (orig_h * orig_w) / (latent_h * latent_w)
#     channel_expansion = latent_channels / orig_channels
    
#     # Compression ratio
#     compression_ratio = orig_size_bits / latent_size_bits

#     return {
#         'orig_resolution': (orig_h, orig_w, orig_channels),
#         'orig_size_mb': orig_size_mb,
#         'latent_resolution': (latent_h, latent_w, latent_channels),
#         'latent_size_mb': latent_size_mb,
#         'spatial_reduction': spatial_reduction,
#         'channel_expansion': channel_expansion,
#         'compression_ratio': compression_ratio
#     }

# def pad_batch(x, p):
#     batch_size = x.size(0)
#     h, w = x.size(2), x.size(3)
#     new_h = (h + p - 1) // p * p
#     new_w = (w + p - 1) // p * p
#     padding_left = (new_w - w) // 2
#     padding_right = new_w - w - padding_left
#     padding_top = (new_h - h) // 2
#     padding_bottom = new_h - h - padding_top
#     x_padded = F.pad(
#         x,
#         (padding_left, padding_right, padding_top, padding_bottom),
#         mode="constant",
#         value=0,
#     )
#     return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

# def crop_batch(x, padding):
#     return F.pad(
#         x,
#         (-padding[0], -padding[1], -padding[2], -padding[3]),
#     )

# def load_images_batch(img_paths, device):
#     images = []
#     for img_path in img_paths:
#         img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
#         images.append(img)
#     return torch.stack(images).to(device)

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="SimpleAutoencoder split evaluation script (g_a on CPU, g_s on CUDA).")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda for g_s", default=True)
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to SimpleAutoencoder checkpoint", 
#                        default="./checkpoints/train_simple/ms-ssim/checkpoint_best.pth.tar")
#     parser.add_argument("--data", type=str, help="Path to dataset", default='../datasets/dummy/valid')
#     parser.add_argument("--save_path", default=None, type=str, help="Path to save reconstructed images and metrics")
#     parser.add_argument("--N", type=int, default=192, help="Number of channels N")
#     parser.add_argument("--M", type=int, default=320, help="Number of channels M")
#     parser.add_argument("--head_dim", nargs='+', type=int, default=[8, 16, 32, 32, 16, 8], 
#                        help="Head dimensions for attention")
#     parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
#     parser.add_argument("--split_timing", action="store_true", default=True,
#                        help="Measure encoding and decoding times separately")
#     args = parser.parse_args(argv)
#     return args

# def main(argv):
#     torch.backends.cudnn.enabled = False
#     args = parse_args(argv)
#     p = 128  # Padding to multiple of 128
#     path = args.data
#     batch_size = args.batch_size
    
#     img_list = []
#     for file in os.listdir(path):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(os.path.join(path, file))
    
#     # Define devices for split deployment
#     cpu_device = 'cpu'
#     if args.cuda and torch.cuda.is_available():
#         cuda_device = 'cuda:0'
#         print(f"🚀 Split deployment: g_a on CPU, g_s on {cuda_device}")
#     else:
#         cuda_device = 'cpu'
#         print("⚠️  CUDA not available, using CPU for both g_a and g_s")
    
#     # Input data will be on CPU initially
#     input_device = cpu_device
        
#     # Initialize SimpleAutoencoder
#     net = SimpleAutoencoder(head_dim=args.head_dim, N=args.N, M=args.M)
#     net.eval()
    
#     # Move g_a to CPU and g_s to CUDA
#     net.g_a = net.g_a.to(cpu_device)
#     net.g_s = net.g_s.to(cuda_device)
    
#     print(f"✅ g_a deployed on: {cpu_device}")
#     print(f"✅ g_s deployed on: {cuda_device}")
    
#     count = 0
#     total_PSNR = 0
#     total_MS_SSIM = 0
#     total_compression_ratio = 0
#     total_time = 0
#     encode_time = 0
#     decode_time = 0
#     transfer_time = 0
    
#     total_compression_ratio = 0
#     size_analysis_done = False  # To print analysis only once
    
#     # Load checkpoint
#     if args.checkpoint:  
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location='cpu')  # Load to CPU first
        
#         # Handle different checkpoint formats
#         if 'state_dict' in checkpoint:
#             state_dict = checkpoint['state_dict']
#         else:
#             state_dict = checkpoint
            
#         # Remove 'module.' prefix if present (from DataParallel)
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('module.'):
#                 new_state_dict[k[7:]] = v  # Remove 'module.' prefix
#             else:
#                 new_state_dict[k] = v
        
#         net.load_state_dict(new_state_dict)
        
#         # Re-deploy components to their respective devices after loading
#         net.g_a = net.g_a.to(cpu_device)
#         net.g_s = net.g_s.to(cuda_device)
        
#         print("✅ Checkpoint loaded successfully!")
    
#     # Create save directory if specified
#     if args.save_path is not None:
#         os.makedirs(args.save_path, exist_ok=True)
    
#     # Process images in batches
#     num_batches = (len(img_list) + batch_size - 1) // batch_size
    
#     print(f"Processing {len(img_list)} images in {num_batches} batches...")
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(img_list))
#         batch_img_paths = img_list[start_idx:end_idx]
#         current_batch_size = len(batch_img_paths)
        
#         print(f"Processing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
        
#         # Load batch of images to CPU (input device)
#         x_batch = load_images_batch(batch_img_paths, input_device)
#         x_batch_padded, padding = pad_batch(x_batch, p)
        
#         count += current_batch_size
        
#         with torch.no_grad():
#             if args.split_timing:
#                 # Measure encoding time (g_a on CPU)
#                 s = time.time()
#                 latent = net.g_a(x_batch_padded)  # Encoding on CPU
#                 e = time.time()
#                 batch_encode_time = (e - s)
#                 encode_time += batch_encode_time
#                 if not size_analysis_done:
#                     analysis = compute_size_analysis(x_batch_padded[0:1], latent[0:1])
                    
#                     print("\n" + "="*60)
#                     print("📊 SIZE ANALYSIS (Single Image)")
#                     print("="*60)
#                     print(f"Original Image:")
#                     print(f"  Resolution: {analysis['orig_resolution'][0]}×{analysis['orig_resolution'][1]}×{analysis['orig_resolution'][2]}")
#                     print(f"  Data Type: int8 (8 bits/pixel)")
#                     print(f"  Storage Size: {analysis['orig_size_mb']:.3f} MB")
                    
#                     print(f"\nLatent Representation:")
#                     print(f"  Resolution: {analysis['latent_resolution'][0]}×{analysis['latent_resolution'][1]}×{analysis['latent_resolution'][2]}")
#                     print(f"  Data Type: float32 (32 bits/value)")
#                     print(f"  Storage Size: {analysis['latent_size_mb']:.3f} MB")
                    
#                     print(f"\nCompression Analysis:")
#                     print(f"  Spatial Reduction: {analysis['spatial_reduction']:.1f}× (spatial compression)")
#                     print(f"  Channel Expansion: {analysis['channel_expansion']:.1f}× (more channels)")
#                     print(f"  Overall Ratio: {analysis['compression_ratio']:.3f} ({'compression' if analysis['compression_ratio'] > 1 else 'expansion'})")
                    
#                     if analysis['compression_ratio'] < 1:
#                         expansion_percent = (1 / analysis['compression_ratio'] - 1) * 100
#                         print(f"  Data Expansion: {expansion_percent:.1f}% larger than original")
#                     else:
#                         compression_percent = (1 - 1/analysis['compression_ratio']) * 100
#                         print(f"  Data Compression: {compression_percent:.1f}% smaller than original")
                    
#                     print("="*60 + "\n")
#                     size_analysis_done = True
                
#                 # Measure data transfer time (CPU -> CUDA)
#                 s = time.time()
#                 latent_cuda = latent.to(cuda_device)
#                 if cuda_device != 'cpu':
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 batch_transfer_time = (e - s)
#                 transfer_time += batch_transfer_time
                
#                 # Measure decoding time (g_s on CUDA)
#                 if cuda_device != 'cpu':
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 x_hat = net.g_s(latent_cuda)  # Decoding on CUDA
#                 if cuda_device != 'cpu':
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 batch_decode_time = (e - s)
#                 decode_time += batch_decode_time
                
#                 total_time += (batch_encode_time + batch_transfer_time + batch_decode_time)
                
#                 print(f'Batch {batch_idx + 1} - Encoding (CPU): {batch_encode_time*1000:.3f} ms, '
#                       f'Transfer: {batch_transfer_time*1000:.3f} ms, '
#                       f'Decoding (CUDA): {batch_decode_time*1000:.3f} ms')
                
#                 # Move result back to CPU for processing
#                 x_hat = x_hat.to(cpu_device)
                
#             else:
#                 # Measure total forward pass time with split deployment
#                 s = time.time()
                
#                 # Encoding on CPU
#                 latent = net.g_a(x_batch_padded)
                
#                 # Transfer to CUDA
#                 latent_cuda = latent.to(cuda_device)
#                 if cuda_device != 'cpu':
#                     torch.cuda.synchronize()
                
#                 # Decoding on CUDA
#                 x_hat = net.g_s(latent_cuda)
#                 if cuda_device != 'cpu':
#                     torch.cuda.synchronize()
                
#                 # Move result back to CPU
#                 x_hat = x_hat.to(cpu_device)
                
#                 e = time.time()
#                 batch_time = (e - s)
#                 total_time += batch_time
                
#                 print(f'Batch {batch_idx + 1} - Total processing time: {batch_time*1000:.3f} ms')
            
#             # Clamp and crop reconstructed images
#             x_hat.clamp_(0, 1)
#             x_hat = crop_batch(x_hat, padding)
            
#             # Compute metrics for batch (all on CPU)
#             psnr_batch = compute_psnr(x_batch, x_hat)
#             msssim_batch = compute_msssim(x_batch, x_hat)
            
#             # Compute compression ratio for each image
#             compression_ratios = []
#             for i in range(current_batch_size):
#                 ratio = compute_compression_ratio(x_batch[i:i+1], latent[i:i+1])
#                 compression_ratios.append(ratio)
#             compression_ratios = torch.tensor(compression_ratios)
            
#             total_PSNR += psnr_batch.sum().item()
#             total_MS_SSIM += msssim_batch.sum().item()
#             total_compression_ratio += compression_ratios.sum().item()
            
#             # Save individual results if requested
#             if args.save_path is not None:
#                 for i in range(current_batch_size):
#                     img_name = os.path.basename(batch_img_paths[i])
#                     save_metrics(
#                         os.path.join(args.save_path, f"metrics_{img_name.split('.')[0]}.txt"), 
#                         psnr_batch[i].item(), 
#                         msssim_batch[i].item(),
#                         compression_ratios[i].item()
#                     )
#                     save_image(x_hat[i:i+1], os.path.join(args.save_path, f"reconstructed_{img_name}"))
    
#     # Calculate averages
#     avg_PSNR = total_PSNR / count
#     avg_MS_SSIM = total_MS_SSIM / count
#     avg_compression_ratio = total_compression_ratio / count
#     avg_total_time = (total_time / count) * 1000  # Convert to ms
    
#     print("\n" + "="*50)
#     print("SPLIT DEPLOYMENT EVALUATION RESULTS")
#     print("="*50)
#     print(f'Deployment: g_a on {cpu_device}, g_s on {cuda_device}')
#     print(f'Total images processed: {count}')
#     print(f'Average PSNR: {avg_PSNR:.2f} dB')
#     print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
#     print(f'Average Compression Ratio: {avg_compression_ratio:.2f}')
#     print(f'Average total time per image: {avg_total_time:.3f} ms')
    
#     if args.split_timing:
#         avg_encode_time = (encode_time / count) * 1000
#         avg_decode_time = (decode_time / count) * 1000
#         avg_transfer_time = (transfer_time / count) * 1000
#         print(f'Average encode time per image (CPU): {avg_encode_time:.3f} ms')
#         print(f'Average transfer time per image: {avg_transfer_time:.3f} ms')
#         print(f'Average decode time per image (CUDA): {avg_decode_time:.3f} ms')
    
#     # Save summary results
#     if args.save_path is not None:
#         with open(os.path.join(args.save_path, "summary_results_split.txt"), 'w') as f:
#             f.write(f'Split Deployment: g_a on {cpu_device}, g_s on {cuda_device}\n')
#             f.write(f'Total images processed: {count}\n')
#             f.write(f'Average PSNR: {avg_PSNR:.2f} dB\n')
#             f.write(f'Average MS-SSIM: {avg_MS_SSIM:.4f}\n')
#             f.write(f'Average Compression Ratio: {avg_compression_ratio:.2f}\n')
#             f.write(f'Average total time per image: {avg_total_time:.3f} ms\n')
#             if args.split_timing:
#                 f.write(f'Average encode time per image (CPU): {avg_encode_time:.3f} ms\n')
#                 f.write(f'Average transfer time per image: {avg_transfer_time:.3f} ms\n')
#                 f.write(f'Average decode time per image (CUDA): {avg_decode_time:.3f} ms\n')
#         print(f"\nResults saved to: {args.save_path}")

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])




###
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.g_a_g_s import SimpleAutoencoder
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
from thop import profile
warnings.filterwarnings("ignore")
torch.set_num_threads(10)

print(torch.cuda.is_available())

def save_image(tensor, filename):
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    img.save(filename)

def save_metrics(filename, psnr, msssim, compression_ratio=None):
    with open(filename, 'w') as f:
        f.write(f'PSNR: {psnr:.2f}dB\n')
        f.write(f'MS-SSIM: {msssim:.4f}\n')
        if compression_ratio is not None:
            f.write(f'Compression Ratio: {compression_ratio:.2f}\n')

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2, dim=[1,2,3])  # Compute per image in batch
    return -10 * torch.log10(mse)

def compute_msssim(a, b):
    # Process each image in the batch
    msssim_values = []
    for i in range(a.size(0)):
        msssim_val = ms_ssim(a[i:i+1], b[i:i+1], data_range=1.).item()
        msssim_values.append(-10 * math.log10(1 - msssim_val))
    return torch.tensor(msssim_values)

def compute_compression_ratio(x, latent):
    """Compute approximate compression ratio based on tensor sizes"""
    original_size = x.numel() * 8  # Assuming 8-bit integers
    compressed_size = latent.numel() * 32  # Assuming 32-bit floats
    return original_size / compressed_size


def compute_size_analysis(x, latent):
    """Compute detailed size analysis between original and latent"""
    # Original image analysis
    orig_h, orig_w = x.shape[2], x.shape[3]
    orig_channels = x.shape[1]
    orig_pixels = orig_h * orig_w * orig_channels
    orig_size_bits = orig_pixels * 8  # int8 = 8 bits per pixel
    orig_size_mb = orig_size_bits / (8 * 1024 * 1024)  # Convert to MB
    
    # Latent analysis - check if it's float16 or float32
    latent_h, latent_w = latent.shape[2], latent.shape[3]
    latent_channels = latent.shape[1]
    latent_values = latent_h * latent_w * latent_channels
    
    # Determine bits per value based on dtype
    if latent.dtype == torch.float16:
        bits_per_value = 16
        dtype_str = "float16"
    else:
        bits_per_value = 32
        dtype_str = "float32"

    # Latent analysis
    latent_h, latent_w = latent.shape[2], latent.shape[3]
    latent_channels = latent.shape[1]
    latent_values = latent_h * latent_w * latent_channels
    latent_size_bits = latent_values * bits_per_value  # float32 = 32 bits per value
    latent_size_mb = latent_size_bits / (8 * 1024 * 1024)  # Convert to MB
    
    # Spatial compression
    spatial_reduction = (orig_h * orig_w) / (latent_h * latent_w)
    channel_expansion = latent_channels / orig_channels
    
    # Compression ratio
    compression_ratio = orig_size_bits / latent_size_bits

    return {
        'orig_resolution': (orig_h, orig_w, orig_channels),
        'orig_size_mb': orig_size_mb,
        'latent_resolution': (latent_h, latent_w, latent_channels),
        'latent_size_mb': latent_size_mb,
        'data Type': {dtype_str},
        'spatial_reduction': spatial_reduction,
        'channel_expansion': channel_expansion,
        'compression_ratio': compression_ratio
    }

def pad_batch(x, p):
    batch_size = x.size(0)
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop_batch(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def load_images_batch(img_paths, device):
    images = []
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        img = transforms.Resize((256,256))(img)  # Resize to 424x242
        img = transforms.ToTensor()(img)
        images.append(img)
    
    return torch.stack(images).to(device)


def get_model_size_mb(model):
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def parse_args(argv):
    parser = argparse.ArgumentParser(description="SimpleAutoencoder split evaluation script (g_a on CPU, g_s on CUDA).")
    parser.add_argument("--cuda", action="store_true", help="Use cuda for g_s", default=True)
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to SimpleAutoencoder checkpoint", 
                       default="/home/xie/DCAE/checkpoints/train_simple/ms-ssim/checkpoint_best.pth.tar")
    parser.add_argument("--data", type=str, help="Path to dataset", default='../datasets/dummy/valid')
    parser.add_argument("--save_path", default="./dataset/reconstructed", type=str, 
                       help="Path to save reconstructed images and metrics")
    parser.add_argument("--N", type=int, default=192, help="Number of channels N")
    parser.add_argument("--M", type=int, default=160, help="Number of channels M")
    parser.add_argument("--head_dim", nargs='+', type=int, default=[8, 16, 32, 32, 16, 8], 
                       help="Head dimensions for attention")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--split_timing", action="store_true", default=True,
                       help="Measure encoding and decoding times separately")
    args = parser.parse_args(argv)
    return args

def main(argv):
    torch.backends.cudnn.enabled = False
    args = parse_args(argv)
    p = 128  # Padding to multiple of 128
    path = args.data
    batch_size = args.batch_size
    
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(os.path.join(path, file))
    
    # Define devices for split deployment
    cpu_device = 'cpu'
    if args.cuda and torch.cuda.is_available():
        cuda_device = 'cuda:0'
        print(f"🚀 Split deployment: g_a on CPU, g_s on {cuda_device}")
    else:
        cuda_device = 'cpu'
        print("⚠️  CUDA not available, using CPU for both g_a and g_s")
    
    use_amp_for_decoder = cuda_device != 'cpu'
    if use_amp_for_decoder:
        print("🚀 Using AMP for decoder (g_s) on CUDA")
        print("📝 Using float32 for encoder (g_a) on CPU")
    else:
        print("📝 Using float32 for both g_a and g_s")
    
    # Input data will be on CPU initially
    input_device = cpu_device
        
    # Initialize SimpleAutoencoder
    net = SimpleAutoencoder(head_dim=args.head_dim, N=args.N, M=args.M)
    net.eval()

    g_a_size_mb = get_model_size_mb(net.g_a)
    g_s_size_mb = get_model_size_mb(net.g_s)
    total_size_mb = g_a_size_mb + g_s_size_mb

    print(f"📏 Model component sizes:")
    print(f"   g_a (encoder) size: {g_a_size_mb:.2f} MB")
    print(f"   g_s (decoder) size: {g_s_size_mb:.2f} MB")
    print(f"   Total model size: {total_size_mb:.2f} MB")

    
    # Move g_a to CPU and g_s to CUDA
    net.g_a = net.g_a.to(cpu_device)
    net.g_s = net.g_s.to(cuda_device)
    
    print(f"✅ g_a deployed on: {cpu_device}")
    print(f"✅ g_s deployed on: {cuda_device}")
    
    count = 0
    total_PSNR = 0
    total_MS_SSIM = 0
    total_compression_ratio = 0
    total_time = 0
    encode_time = 0
    decode_time = 0
    transfer_time = 0
    
    total_compression_ratio = 0
    size_analysis_done = False  # To print analysis only once

    if args.checkpoint:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')  # Load to CPU first
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        
        # Filter to keep only g_a and g_s parameters
        filtered_state_dict = {k: v for k, v in new_state_dict.items() 
                            if k.startswith('g_a') or k.startswith('g_s')}
        
        net.load_state_dict(filtered_state_dict)
        
        # Re-deploy components to their respective devices after loading
        net.g_a = net.g_a.to(cpu_device)
        net.g_s = net.g_s.to(cuda_device)
        
        print("✅ Checkpoint loaded successfully!")
    
    # Create save directory - now always created since we have a default path
    print(f"📁 Creating output directory: {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    print(f"✅ Reconstructed images will be saved to: {args.save_path}")
    
    # Process images in batches
    num_batches = (len(img_list) + batch_size - 1) // batch_size
    
    print(f"Processing {len(img_list)} images in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(img_list))
        batch_img_paths = img_list[start_idx:end_idx]
        current_batch_size = len(batch_img_paths)
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
        
        # Load batch of images to CPU (input device)
        x_batch = load_images_batch(batch_img_paths, input_device)
        x_batch_padded, padding = pad_batch(x_batch, p)
        
        count += current_batch_size
        
        with torch.no_grad():
            if args.split_timing:
                # Measure encoding time (g_a on CPU)
                s = time.time()
                latent = net.g_a(x_batch_padded)  # Encoding on CPU
                latent_f16 = latent.half()  # Convert to float16
                e = time.time()
                batch_encode_time = (e - s)
                encode_time += batch_encode_time

                if not size_analysis_done:
                    analysis = compute_size_analysis(x_batch_padded[0:1], latent_f16[0:1])
                    
                    print("\n" + "="*60)
                    print("📊 SIZE ANALYSIS (Single Image)")
                    print("="*60)
                    print(f"Original Image:")
                    print(f"  Resolution: {analysis['orig_resolution'][0]}×{analysis['orig_resolution'][1]}×{analysis['orig_resolution'][2]}")
                    print(f"  Data Type: int8 (8 bits/pixel)")
                    print(f"  Storage Size: {analysis['orig_size_mb']:.3f} MB")
                    
                    print(f"\nLatent Representation:")
                    print(f"  Resolution: {analysis['latent_resolution'][0]}×{analysis['latent_resolution'][1]}×{analysis['latent_resolution'][2]}")
                    print(f"  Data Type: {latent_f16.dtype} (16 bits/value)")
                    print(f"  Storage Size: {analysis['latent_size_mb']:.3f} MB")
                    
                    print(f"\nCompression Analysis:")
                    print(f"  Spatial Reduction: {analysis['spatial_reduction']:.1f}× (spatial compression)")
                    print(f"  Channel Expansion: {analysis['channel_expansion']:.1f}× (more channels)")
                    print(f"  Overall Ratio: {analysis['compression_ratio']:.3f} ({'compression' if analysis['compression_ratio'] > 1 else 'expansion'})")
                    
                    if analysis['compression_ratio'] < 1:
                        expansion_percent = (1 / analysis['compression_ratio'] - 1) * 100
                        print(f"  Data Expansion: {expansion_percent:.1f}% larger than original")
                    else:
                        compression_percent = (1 - 1/analysis['compression_ratio']) * 100
                        print(f"  Data Compression: {compression_percent:.1f}% smaller than original")
                    
                    print("="*60 + "\n")
                    size_analysis_done = True
                
                # Measure data transfer time (CPU -> CUDA)
                s = time.time()
                latent_cuda = latent_f16.to(cuda_device)
                if cuda_device != 'cpu':
                    torch.cuda.synchronize()
                e = time.time()
                batch_transfer_time = (e - s)
                transfer_time += batch_transfer_time
                
                # Measure decoding time (g_s on CUDA)
                if cuda_device != 'cpu':
                    torch.cuda.synchronize()
                # Decoding on CUDA (with optional AMP)
                s = time.time()
                if use_amp_for_decoder:
                    with torch.cuda.amp.autocast():
                        x_hat = net.g_s(latent_cuda)  # Mixed precision on CUDA
                else:
                    x_hat = net.g_s(latent_cuda.float())  # Regular float32
                e = time.time()
                batch_decode_time = (e - s)
                decode_time += batch_decode_time
                
                total_time += (batch_encode_time + batch_transfer_time + batch_decode_time)
                
                print(f'Batch {batch_idx + 1} - Encoding (CPU): {batch_encode_time*1000:.3f} ms, '
                      f'Transfer: {batch_transfer_time*1000:.3f} ms, '
                      f'Decoding (CUDA): {batch_decode_time*1000:.3f} ms')
                
                # Move result back to CPU for processing
                x_hat = x_hat.to(cpu_device)
                
            else:
                # Measure total forward pass time with split deployment
                s = time.time()
                
                # Encoding on CPU
                latent = net.g_a(x_batch_padded)
                
                # Transfer to CUDA
                latent_cuda = latent.to(cuda_device)
                if cuda_device != 'cpu':
                    torch.cuda.synchronize()
                
                # Decoding on CUDA
                x_hat = net.g_s(latent_cuda)
                if cuda_device != 'cpu':
                    torch.cuda.synchronize()
                
                # Move result back to CPU
                x_hat = x_hat.to(cpu_device)
                
                e = time.time()
                batch_time = (e - s)
                total_time += batch_time
                
                print(f'Batch {batch_idx + 1} - Total processing time: {batch_time*1000:.3f} ms')
            
            # Clamp and crop reconstructed images
            x_hat.clamp_(0, 1)
            x_hat = crop_batch(x_hat, padding)

            if x_hat.dtype != x_batch.dtype:
                print(f"🔧 Dtype mismatch detected: x_batch={x_batch.dtype}, x_hat={x_hat.dtype}")
                print("   Converting both to float32 for metrics calculation...")
                x_batch_metrics = x_batch.float()
                x_hat_metrics = x_hat.float()
            else:
                x_batch_metrics = x_batch
                x_hat_metrics = x_hat
            
            # Compute metrics for batch (all on CPU)
            psnr_batch = compute_psnr(x_batch_metrics, x_hat_metrics)
            msssim_batch = compute_msssim(x_batch_metrics, x_hat_metrics)
            
            # Compute compression ratio for each image
            compression_ratios = []
            for i in range(current_batch_size):
                ratio = compute_compression_ratio(x_batch[i:i+1], latent[i:i+1])
                compression_ratios.append(ratio)
            compression_ratios = torch.tensor(compression_ratios)
            
            total_PSNR += psnr_batch.sum().item()
            total_MS_SSIM += msssim_batch.sum().item()
            total_compression_ratio += compression_ratios.sum().item()
            
            # Save individual results - now always saves since we have default path
            for i in range(current_batch_size):
                img_name = os.path.basename(batch_img_paths[i])
                
                # Save metrics
                save_metrics(
                    os.path.join(args.save_path, f"metrics_{img_name.split('.')[0]}.txt"), 
                    psnr_batch[i].item(), 
                    msssim_batch[i].item(),
                    compression_ratios[i].item()
                )
                
                # Save reconstructed image
                reconstructed_path = os.path.join(args.save_path, f"reconstructed_{img_name}")
                save_image(x_hat[i:i+1], reconstructed_path)
                
                # Print save confirmation for first few images
                if batch_idx == 0 and i < 3:
                    print(f"💾 Saved: {reconstructed_path}")
    
    # Calculate averages
    avg_PSNR = total_PSNR / count
    avg_MS_SSIM = total_MS_SSIM / count
    avg_compression_ratio = total_compression_ratio / count
    avg_total_time = (total_time / count) * 1000  # Convert to ms
    
    print("\n" + "="*50)
    print("SPLIT DEPLOYMENT EVALUATION RESULTS")
    print("="*50)
    print(f'Deployment: g_a on {cpu_device}, g_s on {cuda_device}')
    print(f'Total images processed: {count}')
    print(f'Average PSNR: {avg_PSNR:.2f} dB')
    print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
    print(f'Average Compression Ratio: {avg_compression_ratio:.2f}')
    print(f'Average total time per image: {avg_total_time:.3f} ms')
    
    if args.split_timing:
        avg_encode_time = (encode_time / count) * 1000
        avg_decode_time = (decode_time / count) * 1000
        avg_transfer_time = (transfer_time / count) * 1000
        print(f'Average encode time per image (CPU): {avg_encode_time:.3f} ms')
        print(f'Average transfer time per image: {avg_transfer_time:.3f} ms')
        print(f'Average decode time per image (CUDA): {avg_decode_time:.3f} ms')
    
    # Save summary results
    with open(os.path.join(args.save_path, "summary_results_split.txt"), 'w') as f:
        f.write(f'Split Deployment: g_a on {cpu_device}, g_s on {cuda_device}\n')
        f.write(f'Total images processed: {count}\n')
        f.write(f'Average PSNR: {avg_PSNR:.2f} dB\n')
        f.write(f'Average MS-SSIM: {avg_MS_SSIM:.4f}\n')
        f.write(f'Average Compression Ratio: {avg_compression_ratio:.2f}\n')
        f.write(f'Average total time per image: {avg_total_time:.3f} ms\n')
        if args.split_timing:
            f.write(f'Average encode time per image (CPU): {avg_encode_time:.3f} ms\n')
            f.write(f'Average transfer time per image: {avg_transfer_time:.3f} ms\n')
            f.write(f'Average decode time per image (CUDA): {avg_decode_time:.3f} ms\n')
    
    print(f"\n✅ All results saved to: {args.save_path}")
    print(f"📊 Summary file: {os.path.join(args.save_path, 'summary_results_split.txt')}")

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])