# # simulated evaluation
# import os
# import argparse
# import math
# import csv
# import time
# from PIL import Image
# import numpy as np

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image

# from compressai.datasets import ImageFolder
# from pytorch_msssim import ms_ssim

# # Import your models
# from models import CompressModel, DecompressModel

# def compute_psnr(a, b):
#     """Calculate PSNR between two tensors"""
#     mse = torch.mean((a - b) ** 2).item()
#     if mse == 0:
#         return float('inf')
#     return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     """Calculate MS-SSIM between two tensors"""
#     return ms_ssim(a, b, data_range=1.).item()

# def compute_bpp(likelihoods, num_pixels):
#     """Calculate bits per pixel"""
#     bpp = sum(
#         (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
#         for likelihood in likelihoods.values()
#     )
#     return bpp.item()

# class ModelEvaluator:
#     def __init__(self, checkpoint_path, N=192, M=320, device='cuda'):
#         self.device = device
#         self.N = N
#         self.M = M
        
#         # Initialize models
#         print("Initializing models...")
#         self.compress_model = CompressModel(N=N, M=M).cpu()  # Keep on CPU as in training
#         self.decompress_model = DecompressModel(N=N, M=M).to(device)  # Move to GPU
        
#         # Load checkpoint
#         print(f"Loading checkpoint: {checkpoint_path}")
#         self.load_checkpoint(checkpoint_path)
        
#         # Update entropy models
#         print("Updating entropy models...")
#         self.compress_model.update()
#         self.decompress_model.update()
        
#         # Set to evaluation mode
#         self.compress_model.eval()
#         self.decompress_model.eval()
        
#         print("Model initialization complete!")
    
#     def load_checkpoint(self, checkpoint_path):
#         """Load checkpoint and restore model states"""
#         checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
#         # Load model states
#         self.compress_model.load_state_dict(checkpoint['compress_model']['state_dict'])
#         self.decompress_model.load_state_dict(checkpoint['decompress_model']['state_dict'])
        
#         # Debug: Check if entropy models have proper buffers
#         print(f"Entropy bottleneck CDF shape: {self.compress_model.entropy_bottleneck._quantized_cdf.shape}")
#         print(f"Gaussian conditional CDF shape: {self.compress_model.gaussian_conditional._quantized_cdf.shape}")
    
#     def evaluate_image(self, image):
#         """Evaluate a single image"""
#         with torch.no_grad():
#             # Forward pass through compression (CPU)
#             image_cpu = image.cpu()
#             compress_out = self.compress_model(image_cpu)
            
#             # Transfer to GPU for decompression
#             y_hat_gpu = compress_out["y_hat"].to(self.device)
#             z_hat_gpu = compress_out["z_hat"].to(self.device)
            
#             # Forward pass through decompression (GPU)
#             decompress_out = self.decompress_model(y_hat_gpu, z_hat_gpu)
            
#             # Calculate metrics
#             image_gpu = image.to(self.device)
#             reconstructed = decompress_out["x_hat"]
            
#             # Ensure reconstructed image is in valid range
#             reconstructed = torch.clamp(reconstructed, 0, 1)
            
#             # Calculate metrics
#             num_pixels = image.numel()
#             psnr = compute_psnr(reconstructed, image_gpu)
#             msssim = compute_msssim(reconstructed, image_gpu)
#             bpp = compute_bpp(compress_out["likelihoods"], num_pixels)
            
#             return {
#                 'reconstructed': reconstructed.cpu(),
#                 'psnr': psnr,
#                 'msssim': msssim,
#                 'bpp': bpp,
#                 'original': image.cpu()
#             }
# def evaluate_dataset(evaluator, test_dataloader, output_dir, save_images=True):
#     """Evaluate the entire test dataset"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     if save_images:
#         os.makedirs(os.path.join(output_dir, 'original'), exist_ok=True)
#         os.makedirs(os.path.join(output_dir, 'reconstructed'), exist_ok=True)
    
#     results = []
#     total_psnr = 0
#     total_msssim = 0
#     total_bpp = 0
    
#     print("Starting evaluation...")
#     start_time = time.time()
    
#     for i, batch in enumerate(test_dataloader):
#         batch_size = batch.size(0)
        
#         for j in range(batch_size):
#             image = batch[j:j+1]  # Keep batch dimension
            
#             # Evaluate single image
#             result = evaluator.evaluate_image(image)
            
#             image_idx = i * test_dataloader.batch_size + j
            
#             # Save results
#             results.append({
#                 'image_idx': image_idx,
#                 'psnr': result['psnr'],
#                 'msssim': result['msssim'],
#                 'bpp': result['bpp']
#             })
            
#             # Save images if requested
#             if save_images:
#                 save_image(
#                     result['original'], 
#                     os.path.join(output_dir, 'original', f'image_{image_idx:04d}.png')
#                 )
#                 save_image(
#                     result['reconstructed'], 
#                     os.path.join(output_dir, 'reconstructed', f'image_{image_idx:04d}.png')
#                 )
            
#             # Update totals
#             total_psnr += result['psnr']
#             total_msssim += result['msssim']
#             total_bpp += result['bpp']
            
#             # Print progress
#             if (image_idx + 1) % 10 == 0:
#                 elapsed = time.time() - start_time
#                 avg_time = elapsed / (image_idx + 1)
#                 print(f"Processed {image_idx + 1} images | "
#                       f"Avg PSNR: {total_psnr/(image_idx + 1):.2f} | "
#                       f"Avg MS-SSIM: {total_msssim/(image_idx + 1):.4f} | "
#                       f"Avg BPP: {total_bpp/(image_idx + 1):.4f} | "
#                       f"Time/image: {avg_time:.2f}s")
    
#     num_images = len(results)
#     avg_psnr = total_psnr / num_images
#     avg_msssim = total_msssim / num_images
#     avg_bpp = total_bpp / num_images
    
#     # Save detailed results to CSV
#     csv_path = os.path.join(output_dir, 'detailed_results.csv')
#     with open(csv_path, 'w', newline='') as csvfile:
#         fieldnames = ['image_idx', 'psnr', 'msssim', 'bpp']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(results)
    
#     # Save summary results
#     summary_path = os.path.join(output_dir, 'summary.txt')
#     with open(summary_path, 'w') as f:
#         f.write(f"Evaluation Summary\n")
#         f.write(f"==================\n")
#         f.write(f"Total images: {num_images}\n")
#         f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
#         f.write(f"Average MS-SSIM: {avg_msssim:.6f}\n")
#         f.write(f"Average BPP: {avg_bpp:.6f}\n")
#         f.write(f"Total evaluation time: {time.time() - start_time:.2f} seconds\n")
    
#     print(f"\nEvaluation Complete!")
#     print(f"==================")
#     print(f"Total images: {num_images}")
#     print(f"Average PSNR: {avg_psnr:.4f} dB")
#     print(f"Average MS-SSIM: {avg_msssim:.6f}")
#     print(f"Average BPP: {avg_bpp:.6f}")
#     print(f"Results saved to: {output_dir}")
    
#     return {
#         'avg_psnr': avg_psnr,
#         'avg_msssim': avg_msssim,
#         'avg_bpp': avg_bpp,
#         'detailed_results': results
#     }

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate DCAE_5 model performance")
    
#     parser.add_argument("--checkpoint", type=str, default='checkpoints/train_5/try_2/60.5checkpoint_best.pth.tar',
#                         help="Path to checkpoint file (e.g., 60.5checkpoint_best.pth.tar)")
#     parser.add_argument("--dataset", type=str, default='dataset',
#                         help="Path to test dataset")
#     parser.add_argument("--output-dir", type=str, default="./evaluation_results",
#                         help="Output directory for results and images")
#     parser.add_argument("--batch-size", type=int, default=1,
#                         help="Batch size for evaluation")
#     parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256),
#                         help="Size of patches for evaluation")
#     parser.add_argument("--num-workers", type=int, default=4,
#                         help="Number of dataloader workers")
#     parser.add_argument("--N", type=int, default=192,
#                         help="N parameter for model")
#     parser.add_argument("--M", type=int, default=320,
#                         help="M parameter for model")
#     parser.add_argument("--no-save-images", action="store_true",
#                         help="Don't save reconstructed images (only compute metrics)")
#     parser.add_argument("--device", type=str, default="cuda",
#                         help="Device to use for evaluation")
    
#     args = parser.parse_args()
    
#     # Check if checkpoint exists
#     if not os.path.exists(args.checkpoint):
#         print(f"Error: Checkpoint file not found: {args.checkpoint}")
#         return
    
#     # Setup transforms
#     test_transforms = transforms.Compose([
#         transforms.CenterCrop(args.patch_size),
#         transforms.ToTensor()
#     ])
    
#     # Load test dataset
#     print(f"Loading test dataset from: {args.dataset}")
#     test_dataset = ImageFolder(args.dataset, split="valid", transform=test_transforms)
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True
#     )
    
#     print(f"Test dataset loaded: {len(test_dataset)} images")
    
#     # Initialize evaluator
#     evaluator = ModelEvaluator(
#         checkpoint_path=args.checkpoint,
#         N=args.N,
#         M=args.M,
#         device=args.device
#     )
    
#     # Run evaluation
#     results = evaluate_dataset(
#         evaluator=evaluator,
#         test_dataloader=test_dataloader,
#         output_dir=args.output_dir,
#         save_images=not args.no_save_images
#     )
    
#     print(f"\nEvaluation completed successfully!")
#     print(f"Check {args.output_dir} for detailed results and reconstructed images.")

# if __name__ == "__main__":
#     main()




# ## real evaluation
# import os
# import argparse
# import math
# import csv
# import time
# from PIL import Image
# import numpy as np

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image

# from compressai.datasets import ImageFolder
# from pytorch_msssim import ms_ssim

# # Import your models
# from models import CompressModel, DecompressModel

# def compute_psnr(a, b):
#     """Calculate PSNR between two tensors"""
#     mse = torch.mean((a - b) ** 2).item()
#     if mse == 0:
#         return float('inf')
#     return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     """Calculate MS-SSIM between two tensors"""
#     return ms_ssim(a, b, data_range=1.).item()

# def compute_bpp(likelihoods, num_pixels):
#     """Calculate bits per pixel"""
#     bpp = sum(
#         (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
#         for likelihood in likelihoods.values()
#     )
#     return bpp.item()

# class ModelEvaluator:
#     def __init__(self, checkpoint_path, N=192, M=320, device='cuda'):
#         self.device = device
#         self.N = N
#         self.M = M
        
#         # Initialize models
#         print("Initializing models...")
#         self.compress_model = CompressModel(N=N, M=M).cpu()  # Keep on CPU as in training
#         self.decompress_model = DecompressModel(N=N, M=M).to(device)  # Move to GPU
        
#         # Load checkpoint
#         print(f"Loading checkpoint: {checkpoint_path}")
#         self.load_checkpoint(checkpoint_path)
        
#         # Update entropy models
#         print("Updating entropy models...")
#         self.compress_model.update()
#         self.decompress_model.update()
        
#         # Set to evaluation mode
#         self.compress_model.eval()
#         self.decompress_model.eval()
        
#         print("Model initialization complete!")
    
#     def load_checkpoint(self, checkpoint_path):
#         """Load checkpoint and restore model states"""
#         checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
#         # Load model states
#         self.compress_model.load_state_dict(checkpoint['compress_model']['state_dict'])
#         self.decompress_model.load_state_dict(checkpoint['decompress_model']['state_dict'])
        
#         # Debug: Check if entropy models have proper buffers
#         print(f"Entropy bottleneck CDF shape: {self.compress_model.entropy_bottleneck._quantized_cdf.shape}")
#         print(f"Gaussian conditional CDF shape: {self.compress_model.gaussian_conditional._quantized_cdf.shape}")
    
#     def evaluate_image(self, image):
#         """Evaluate a single image using actual compress/decompress pipeline"""
#         with torch.no_grad():
#             # STEP 1: Actual compression (CPU) - produces bitstreams
#             image_cpu = image.cpu()
#             compress_result = self.compress_model.compress(image_cpu)
            
#             # compress_result = {
#             #     "strings": [y_strings, z_strings],  # Actual bitstreams
#             #     "shape": z.size()[-2:]
#             # }
            
#             # STEP 2: Transfer bitstreams (much smaller than tensors)
#             strings = compress_result["strings"]
#             shape = compress_result["shape"]
            
#             # STEP 3: Actual decompression (GPU) - reconstructs from bitstreams
#             decompress_result = self.decompress_model.decompress(strings, shape)
            
#             # Calculate metrics
#             image_gpu = image.to(self.device)
#             reconstructed = decompress_result["x_hat"]
#             reconstructed = torch.clamp(reconstructed, 0, 1)
            
#             # Calculate BPP from actual bitstream sizes
#             num_pixels = image.numel()
#             y_string_bits = len(strings[0][0]) * 8  # Convert bytes to bits
#             z_string_bits = len(strings[1]) * 8
#             bpp = (y_string_bits + z_string_bits) / num_pixels
            
#             psnr = compute_psnr(reconstructed, image_gpu)
#             msssim = compute_msssim(reconstructed, image_gpu)
            
#             return {
#                 'reconstructed': reconstructed.cpu(),
#                 'psnr': psnr,
#                 'msssim': msssim,
#                 'bpp': bpp,
#                 'original': image.cpu(),
#                 'bitstream_sizes': {
#                     'y_string_bytes': len(strings[0][0]),
#                     'z_string_bytes': len(strings[1]),
#                     'total_bytes': len(strings[0][0]) + len(strings[1])
#                 }
#             }

# def evaluate_dataset(evaluator, test_dataloader, output_dir, save_images=True):
#     """Evaluate the entire test dataset"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     if save_images:
#         os.makedirs(os.path.join(output_dir, 'original'), exist_ok=True)
#         os.makedirs(os.path.join(output_dir, 'reconstructed'), exist_ok=True)
    
#     results = []
#     total_psnr = 0
#     total_msssim = 0
#     total_bpp = 0
    
#     print("Starting evaluation...")
#     start_time = time.time()
    
#     for i, batch in enumerate(test_dataloader):
#         batch_size = batch.size(0)
        
#         for j in range(batch_size):
#             image = batch[j:j+1]  # Keep batch dimension
            
#             # Evaluate single image
#             result = evaluator.evaluate_image(image)
            
#             image_idx = i * test_dataloader.batch_size + j
            
#             # Save results
#             results.append({
#                 'image_idx': image_idx,
#                 'psnr': result['psnr'],
#                 'msssim': result['msssim'],
#                 'bpp': result['bpp']
#             })
            
#             # Save images if requested
#             if save_images:
#                 save_image(
#                     result['original'], 
#                     os.path.join(output_dir, 'original', f'image_{image_idx:04d}.png')
#                 )
#                 save_image(
#                     result['reconstructed'], 
#                     os.path.join(output_dir, 'reconstructed', f'image_{image_idx:04d}.png')
#                 )
            
#             # Update totals
#             total_psnr += result['psnr']
#             total_msssim += result['msssim']
#             total_bpp += result['bpp']
            
#             # Print progress
#             if (image_idx + 1) % 10 == 0:
#                 elapsed = time.time() - start_time
#                 avg_time = elapsed / (image_idx + 1)
#                 print(f"Processed {image_idx + 1} images | "
#                       f"Avg PSNR: {total_psnr/(image_idx + 1):.2f} | "
#                       f"Avg MS-SSIM: {total_msssim/(image_idx + 1):.4f} | "
#                       f"Avg BPP: {total_bpp/(image_idx + 1):.4f} | "
#                       f"Time/image: {avg_time:.2f}s")
    
#     num_images = len(results)
#     avg_psnr = total_psnr / num_images
#     avg_msssim = total_msssim / num_images
#     avg_bpp = total_bpp / num_images
    
#     # Save detailed results to CSV
#     csv_path = os.path.join(output_dir, 'detailed_results.csv')
#     with open(csv_path, 'w', newline='') as csvfile:
#         fieldnames = ['image_idx', 'psnr', 'msssim', 'bpp']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(results)
    
#     # Save summary results
#     summary_path = os.path.join(output_dir, 'summary.txt')
#     with open(summary_path, 'w') as f:
#         f.write(f"Evaluation Summary\n")
#         f.write(f"==================\n")
#         f.write(f"Total images: {num_images}\n")
#         f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
#         f.write(f"Average MS-SSIM: {avg_msssim:.6f}\n")
#         f.write(f"Average BPP: {avg_bpp:.6f}\n")
#         f.write(f"Total evaluation time: {time.time() - start_time:.2f} seconds\n")
    
#     print(f"\nEvaluation Complete!")
#     print(f"==================")
#     print(f"Total images: {num_images}")
#     print(f"Average PSNR: {avg_psnr:.4f} dB")
#     print(f"Average MS-SSIM: {avg_msssim:.6f}")
#     print(f"Average BPP: {avg_bpp:.6f}")
#     print(f"Results saved to: {output_dir}")
    
#     return {
#         'avg_psnr': avg_psnr,
#         'avg_msssim': avg_msssim,
#         'avg_bpp': avg_bpp,
#         'detailed_results': results
#     }

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate DCAE_5 model performance")
    
#     parser.add_argument("--checkpoint", type=str, default='checkpoints/train_5/try_2/60.5checkpoint_latest.pth.tar',
#                         help="Path to checkpoint file (e.g., 60.5checkpoint_best.pth.tar)")
#     parser.add_argument("--dataset", type=str, default='dataset',
#                         help="Path to test dataset")
#     parser.add_argument("--output-dir", type=str, default="./evaluation_results",
#                         help="Output directory for results and images")
#     parser.add_argument("--batch-size", type=int, default=1,
#                         help="Batch size for evaluation")
#     parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256),
#                         help="Size of patches for evaluation")
#     parser.add_argument("--num-workers", type=int, default=4,
#                         help="Number of dataloader workers")
#     parser.add_argument("--N", type=int, default=192,
#                         help="N parameter for model")
#     parser.add_argument("--M", type=int, default=320,
#                         help="M parameter for model")
#     parser.add_argument("--no-save-images", action="store_true",
#                         help="Don't save reconstructed images (only compute metrics)")
#     parser.add_argument("--device", type=str, default="cuda",
#                         help="Device to use for evaluation")
    
#     args = parser.parse_args()
    
#     # Check if checkpoint exists
#     if not os.path.exists(args.checkpoint):
#         print(f"Error: Checkpoint file not found: {args.checkpoint}")
#         return
    
#     # Setup transforms
#     test_transforms = transforms.Compose([
#         transforms.CenterCrop(args.patch_size),
#         transforms.ToTensor()
#     ])
    
#     # Load test dataset
#     print(f"Loading test dataset from: {args.dataset}")
#     test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True
#     )
    
#     print(f"Test dataset loaded: {len(test_dataset)} images")
    
#     # Initialize evaluator
#     evaluator = ModelEvaluator(
#         checkpoint_path=args.checkpoint,
#         N=args.N,
#         M=args.M,
#         device=args.device
#     )
    
#     # Run evaluation
#     results = evaluate_dataset(
#         evaluator=evaluator,
#         test_dataloader=test_dataloader,
#         output_dir=args.output_dir,
#         save_images=not args.no_save_images
#     )
    
#     print(f"\nEvaluation completed successfully!")
#     print(f"Check {args.output_dir} for detailed results and reconstructed images.")

# if __name__ == "__main__":
#     main()


## real eval, save output of the encoder
import os
import argparse
import math
import csv
import time
import pickle
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from compressai.datasets import ImageFolder
from pytorch_msssim import ms_ssim

# Import your models
from models import CompressModel, DecompressModel

def compute_psnr(a, b):
    """Calculate PSNR between two tensors"""
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float('inf')
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    """Calculate MS-SSIM between two tensors"""
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(likelihoods, num_pixels):
    """Calculate bits per pixel"""
    bpp = sum(
        (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
        for likelihood in likelihoods.values()
    )
    return bpp.item()

class ModelEvaluator:
    def __init__(self, checkpoint_path, N=192, M=320, device='cuda'):
        self.device = device
        self.N = N
        self.M = M
        
        # Initialize models
        print("Initializing models...")
        self.compress_model = CompressModel(N=N, M=M).cpu()  # Keep on CPU as in training
        self.decompress_model = DecompressModel(N=N, M=M).to(device)  # Move to GPU
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        self.load_checkpoint(checkpoint_path)
        
        # Update entropy models
        print("Updating entropy models...")
        self.compress_model.update()
        self.decompress_model.update()
        
        # Set to evaluation mode
        self.compress_model.eval()
        self.decompress_model.eval()
        
        print("Model initialization complete!")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore model states"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model states
        self.compress_model.load_state_dict(checkpoint['compress_model']['state_dict'])
        self.decompress_model.load_state_dict(checkpoint['decompress_model']['state_dict'])
        
        # Debug: Check if entropy models have proper buffers
        print(f"Entropy bottleneck CDF shape: {self.compress_model.entropy_bottleneck._quantized_cdf.shape}")
        print(f"Gaussian conditional CDF shape: {self.compress_model.gaussian_conditional._quantized_cdf.shape}")
    
    def evaluate_image(self, image):
        """Evaluate a single image using actual compress/decompress pipeline"""
        with torch.no_grad():
            # STEP 1: Actual compression (CPU) - produces bitstreams
            image_cpu = image.cpu()
            compress_result = self.compress_model.compress(image_cpu)
            
            # compress_result = {
            #     "strings": [y_strings, z_strings],  # Actual bitstreams
            #     "shape": z.size()[-2:]
            # }
            
            # STEP 2: Transfer bitstreams (much smaller than tensors)
            strings = compress_result["strings"]
            shape = compress_result["shape"]
            
            # STEP 3: Actual decompression (GPU) - reconstructs from bitstreams
            decompress_result = self.decompress_model.decompress(strings, shape)
            
            # Calculate metrics
            image_gpu = image.to(self.device)
            reconstructed = decompress_result["x_hat"]
            reconstructed = torch.clamp(reconstructed, 0, 1)
            
            # Calculate BPP from actual bitstream sizes
            num_pixels = image.numel()
            y_string_bits = len(strings[0][0]) * 8  # Convert bytes to bits
            z_string_bits = len(strings[1]) * 8
            bpp = (y_string_bits + z_string_bits) / num_pixels
            
            psnr = compute_psnr(reconstructed, image_gpu)
            msssim = compute_msssim(reconstructed, image_gpu)
            
            return {
                'reconstructed': reconstructed.cpu(),
                'psnr': psnr,
                'msssim': msssim,
                'bpp': bpp,
                'original': image.cpu(),
                'strings': strings,  # Only the strings
                'shape': shape,      # Only the shape
                'bitstream_sizes': {
                    'y_string_bytes': len(strings[0][0]),
                    'z_string_bytes': len(strings[1]),
                    'total_bytes': len(strings[0][0]) + len(strings[1])
                }
            }

def evaluate_dataset(evaluator, test_dataloader, output_dir, save_images=True, save_compressed=True):
    """Evaluate the entire test dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    if save_images:
        os.makedirs(os.path.join(output_dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reconstructed'), exist_ok=True)
    
    if save_compressed:
        os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
    
    results = []
    total_psnr = 0
    total_msssim = 0
    total_bpp = 0
    
    print("Starting evaluation...")
    start_time = time.time()
    
    for i, batch in enumerate(test_dataloader):
        batch_size = batch.size(0)
        
        for j in range(batch_size):
            image = batch[j:j+1]  # Keep batch dimension
            
            # Evaluate single image
            result = evaluator.evaluate_image(image)
            
            image_idx = i * test_dataloader.batch_size + j
            
            # Save results
            results.append({
                'image_idx': image_idx,
                'psnr': result['psnr'],
                'msssim': result['msssim'],
                'bpp': result['bpp']
            })
            
            # Save images if requested
            if save_images:
                save_image(
                    result['original'], 
                    os.path.join(output_dir, 'original', f'image_{image_idx:04d}.png')
                )
                save_image(
                    result['reconstructed'], 
                    os.path.join(output_dir, 'reconstructed', f'image_{image_idx:04d}.png')
                )
            
            # Save compressed data if requested (only strings and shape)
            if save_compressed:
                compressed_filename = f'image_{image_idx:04d}.pkl'
                compressed_path = os.path.join(output_dir, 'messaged', compressed_filename)
                
                # Save only what's needed for decompression
                compressed_data = {
                    'strings': result['strings'],
                    'shape': result['shape']
                }
                
                with open(compressed_path, 'wb') as f:
                    pickle.dump(compressed_data, f)
            
            # Update totals
            total_psnr += result['psnr']
            total_msssim += result['msssim']
            total_bpp += result['bpp']
            
            # Print progress
            if (image_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (image_idx + 1)
                print(f"Processed {image_idx + 1} images | "
                      f"Avg PSNR: {total_psnr/(image_idx + 1):.2f} | "
                      f"Avg MS-SSIM: {total_msssim/(image_idx + 1):.4f} | "
                      f"Avg BPP: {total_bpp/(image_idx + 1):.4f} | "
                      f"Time/image: {avg_time:.2f}s")
    
    num_images = len(results)
    avg_psnr = total_psnr / num_images
    avg_msssim = total_msssim / num_images
    avg_bpp = total_bpp / num_images
    
    # Save detailed results to CSV
    csv_path = os.path.join(output_dir, 'detailed_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_idx', 'psnr', 'msssim', 'bpp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Save summary results
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"==================\n")
        f.write(f"Total images: {num_images}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average MS-SSIM: {avg_msssim:.6f}\n")
        f.write(f"Average BPP: {avg_bpp:.6f}\n")
        f.write(f"Total evaluation time: {time.time() - start_time:.2f} seconds\n")
        if save_compressed:
            f.write(f"Compressed data saved to: {os.path.join(output_dir, 'messaged')}\n")
    
    print(f"\nEvaluation Complete!")
    print(f"==================")
    print(f"Total images: {num_images}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average MS-SSIM: {avg_msssim:.6f}")
    print(f"Average BPP: {avg_bpp:.6f}")
    print(f"Results saved to: {output_dir}")
    if save_compressed:
        print(f"Compressed data saved to: {os.path.join(output_dir, 'messaged')}")
    
    return {
        'avg_psnr': avg_psnr,
        'avg_msssim': avg_msssim,
        'avg_bpp': avg_bpp,
        'detailed_results': results
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate DCAE_5 model performance")
    
    parser.add_argument("--checkpoint", type=str, default='checkpoints/train_5/try_2/60.5checkpoint_latest.pth.tar',
                        help="Path to checkpoint file (e.g., 60.5checkpoint_best.pth.tar)")
    parser.add_argument("--dataset", type=str, default='dataset',
                        help="Path to test dataset")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                        help="Output directory for results and images")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256),
                        help="Size of patches for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--N", type=int, default=192,
                        help="N parameter for model")
    parser.add_argument("--M", type=int, default=320,
                        help="M parameter for model")
    parser.add_argument("--no-save-images", action="store_true",
                        help="Don't save reconstructed images (only compute metrics)")
    parser.add_argument("--no-save-compressed", action="store_true",
                        help="Don't save compressed data")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # Setup transforms
    test_transforms = transforms.Compose([
        transforms.CenterCrop(args.patch_size),
        transforms.ToTensor()
    ])
    
    # Load test dataset
    print(f"Loading test dataset from: {args.dataset}")
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test dataset loaded: {len(test_dataset)} images")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        N=args.N,
        M=args.M,
        device=args.device
    )
    
    # Run evaluation
    results = evaluate_dataset(
        evaluator=evaluator,
        test_dataloader=test_dataloader,
        output_dir=args.output_dir,
        save_images=not args.no_save_images,
        save_compressed=not args.no_save_compressed
    )
    
    print(f"\nEvaluation completed successfully!")
    print(f"Check {args.output_dir} for detailed results and reconstructed images.")

if __name__ == "__main__":
    main()