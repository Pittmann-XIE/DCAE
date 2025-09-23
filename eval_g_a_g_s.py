
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
    original_size = x.numel() * 32  # Assuming 32-bit floats
    compressed_size = latent.numel() * 32  # Assuming 32-bit floats
    return original_size / compressed_size

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
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        images.append(img)
    return torch.stack(images).to(device)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="SimpleAutoencoder evaluation script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda", default=True)
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to SimpleAutoencoder checkpoint", 
                       default="./checkpoints/train_simple/ms-ssim/checkpoint_best.pth.tar")
    parser.add_argument("--data", type=str, help="Path to dataset", default='../datasets/dummy/valid')
    parser.add_argument("--save_path", default=None, type=str, help="Path to save reconstructed images and metrics")
    parser.add_argument("--N", type=int, default=192, help="Number of channels N")
    parser.add_argument("--M", type=int, default=320, help="Number of channels M")
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
    
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
        
    # Initialize SimpleAutoencoder
    net = SimpleAutoencoder(head_dim=args.head_dim, N=args.N, M=args.M)
    net = net.to(device)
    net.eval()
    
    count = 0
    total_PSNR = 0
    total_MS_SSIM = 0
    total_compression_ratio = 0
    total_time = 0
    encode_time = 0
    decode_time = 0
    
    # Load checkpoint
    if args.checkpoint:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
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
        
        net.load_state_dict(new_state_dict)
        print("✅ Checkpoint loaded successfully!")
    
    # Create save directory if specified
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    
    # Process images in batches
    num_batches = (len(img_list) + batch_size - 1) // batch_size
    
    print(f"Processing {len(img_list)} images in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(img_list))
        batch_img_paths = img_list[start_idx:end_idx]
        current_batch_size = len(batch_img_paths)
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
        
        # Load batch of images
        x_batch = load_images_batch(batch_img_paths, device)
        x_batch_padded, padding = pad_batch(x_batch, p)
        
        count += current_batch_size
        
        with torch.no_grad():
            if args.split_timing:
                # Measure encoding time
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                latent = net.g_a(x_batch_padded)  # Encoding
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                batch_encode_time = (e - s)
                encode_time += batch_encode_time
                
                # Measure decoding time
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                x_hat = net.g_s(latent)  # Decoding
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                batch_decode_time = (e - s)
                decode_time += batch_decode_time
                
                total_time += (batch_encode_time + batch_decode_time)
                
                print(f'Batch {batch_idx + 1} - Encoding: {batch_encode_time*1000:.3f} ms, Decoding: {batch_decode_time*1000:.3f} ms')
            else:
                # Measure total forward pass time
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_net = net.forward(x_batch_padded)
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                batch_time = (e - s)
                total_time += batch_time
                
                x_hat = out_net["x_hat"]
                latent = out_net["latent"]
                
                print(f'Batch {batch_idx + 1} - Processing time: {batch_time*1000:.3f} ms')
            
            # Clamp and crop reconstructed images
            x_hat.clamp_(0, 1)
            x_hat = crop_batch(x_hat, padding)
            
            # Compute metrics for batch
            psnr_batch = compute_psnr(x_batch, x_hat)
            msssim_batch = compute_msssim(x_batch, x_hat)
            
            # Compute compression ratio for each image
            compression_ratios = []
            for i in range(current_batch_size):
                ratio = compute_compression_ratio(x_batch[i:i+1], latent[i:i+1])
                compression_ratios.append(ratio)
            compression_ratios = torch.tensor(compression_ratios)
            
            total_PSNR += psnr_batch.sum().item()
            total_MS_SSIM += msssim_batch.sum().item()
            total_compression_ratio += compression_ratios.sum().item()
            
            # Save individual results if requested
            if args.save_path is not None:
                for i in range(current_batch_size):
                    img_name = os.path.basename(batch_img_paths[i])
                    save_metrics(
                        os.path.join(args.save_path, f"metrics_{img_name.split('.')[0]}.txt"), 
                        psnr_batch[i].item(), 
                        msssim_batch[i].item(),
                        compression_ratios[i].item()
                    )
                    save_image(x_hat[i:i+1], os.path.join(args.save_path, f"reconstructed_{img_name}"))
    
    # Calculate averages
    avg_PSNR = total_PSNR / count
    avg_MS_SSIM = total_MS_SSIM / count
    avg_compression_ratio = total_compression_ratio / count
    avg_total_time = (total_time / count) * 1000  # Convert to ms
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f'Total images processed: {count}')
    print(f'Average PSNR: {avg_PSNR:.2f} dB')
    print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
    print(f'Average Compression Ratio: {avg_compression_ratio:.2f}')
    print(f'Average total time per image: {avg_total_time:.3f} ms')
    
    if args.split_timing:
        avg_encode_time = (encode_time / count) * 1000
        avg_decode_time = (decode_time / count) * 1000
        print(f'Average encode time per image: {avg_encode_time:.3f} ms')
        print(f'Average decode time per image: {avg_decode_time:.3f} ms')
    
    # Save summary results
    if args.save_path is not None:
        with open(os.path.join(args.save_path, "summary_results.txt"), 'w') as f:
            f.write(f'Total images processed: {count}\n')
            f.write(f'Average PSNR: {avg_PSNR:.2f} dB\n')
            f.write(f'Average MS-SSIM: {avg_MS_SSIM:.4f}\n')
            f.write(f'Average Compression Ratio: {avg_compression_ratio:.2f}\n')
            f.write(f'Average total time per image: {avg_total_time:.3f} ms\n')
            if args.split_timing:
                f.write(f'Average encode time per image: {avg_encode_time:.3f} ms\n')
                f.write(f'Average decode time per image: {avg_decode_time:.3f} ms\n')
        print(f"\nResults saved to: {args.save_path}")

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])