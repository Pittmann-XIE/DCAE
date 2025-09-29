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
    
#     parser.add_argument("--checkpoint", type=str, default='checkpoints/train_5/60.5checkpoint_latest.pth.tar',
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


# ## real eval, save output of the encoder
# import os
# import argparse
# import math
# import csv
# import time
# import pickle
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
# from models import (
#     CompressModel,
#     DecompressModel,
#     ParameterSync
# )

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
#                 'strings': strings,  # Only the strings
#                 'shape': shape,      # Only the shape
#                 'bitstream_sizes': {
#                     'y_string_bytes': len(strings[0][0]),
#                     'z_string_bytes': len(strings[1]),
#                     'total_bytes': len(strings[0][0]) + len(strings[1])
#                 }
#             }

# def evaluate_dataset(evaluator, test_dataloader, output_dir, save_images=True, save_compressed=True):
#     """Evaluate the entire test dataset"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     if save_images:
#         os.makedirs(os.path.join(output_dir, 'original'), exist_ok=True)
#         os.makedirs(os.path.join(output_dir, 'reconstructed'), exist_ok=True)
    
#     if save_compressed:
#         os.makedirs(os.path.join(output_dir, 'latent'), exist_ok=True)
    
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
#                 print(f'saving image to {output_dir}')
#                 save_image(
#                     result['original'], 
#                     os.path.join(output_dir, 'original', f'image_{image_idx:04d}.png')
#                 )
#                 save_image(
#                     result['reconstructed'], 
#                     os.path.join(output_dir, 'reconstructed', f'image_{image_idx:04d}.png')
#                 )
            
#             # Save compressed data if requested (only strings and shape)
#             if save_compressed:
#                 compressed_filename = f'image_{image_idx:04d}.pkl'
#                 compressed_path = os.path.join(output_dir, 'latent', compressed_filename)
                
#                 # Save only what's needed for decompression
#                 compressed_data = {
#                     'strings': result['strings'],
#                     'shape': result['shape']
#                 }
                
#                 with open(compressed_path, 'wb') as f:
#                     pickle.dump(compressed_data, f)
            
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
#         if save_compressed:
#             f.write(f"Compressed data saved to: {os.path.join(output_dir, 'latent')}\n")
    
#     print(f"\nEvaluation Complete!")
#     print(f"==================")
#     print(f"Total images: {num_images}")
#     print(f"Average PSNR: {avg_psnr:.4f} dB")
#     print(f"Average MS-SSIM: {avg_msssim:.6f}")
#     print(f"Average BPP: {avg_bpp:.6f}")
#     print(f"Results saved to: {output_dir}")
#     if save_compressed:
#         print(f"Compressed data saved to: {os.path.join(output_dir, 'latent')}")
    
#     return {
#         'avg_psnr': avg_psnr,
#         'avg_msssim': avg_msssim,
#         'avg_bpp': avg_bpp,
#         'detailed_results': results
#     }

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate DCAE_5 model performance")
    
#     parser.add_argument("--checkpoint", type=str, default='checkpoints/train_5/try_6/60.5/checkpoint_best.pth.tar',
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
#     parser.add_argument("--save-images", action="store_true", default= True,
#                         help="Don't save reconstructed images (only compute metrics)")
#     parser.add_argument("--save-compressed", action="store_true", default=True,
#                         help="Don't save compressed data")
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
#         save_images=args.save_images,
#         save_compressed=args.save_compressed
#     )
    
#     print(f"\nEvaluation completed successfully!")
#     print(f"Check {args.output_dir} for detailed results and reconstructed images.")

# if __name__ == "__main__":
#     main()


## eval with pretrained model of dcae.py
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models.dcae_5 import (
#     CompressModel,
#     DecompressModel, 
#     ParameterSync
# )
# from models import DCAE
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

# def save_metrics(filename, psnr, bitrate, msssim):
#     with open(filename, 'w') as f:
#         f.write(f'PSNR: {psnr:.2f}dB\n')
#         f.write(f'Bitrate: {bitrate:.3f}bpp\n')
#         f.write(f'MS-SSIM: {msssim:.4f}\n')

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2).item()
#     return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

# def compute_bpp(out_net):
#     size = out_net['x_hat'].size()
#     num_pixels = size[0] * size[2] * size[3]
#     return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
#               for likelihoods in out_net['likelihoods'].values()).item()

# def pad(x, p):
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

# def crop(x, padding):
#     return F.pad(
#         x,
#         (-padding[0], -padding[1], -padding[2], -padding[3]),
#     )

# def load_pretrained_to_split_models(checkpoint_path, compress_model, decompress_model, device):
#     """Load pretrained unified model weights to split models"""
#     print(f"Loading pretrained weights from {checkpoint_path}")
    
#     # Load unified model checkpoint
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     unified_state_dict = {}
    
#     # Clean up state dict keys
#     for k, v in checkpoint["state_dict"].items():
#         unified_state_dict[k.replace("module.", "")] = v
    
#     # Create temporary unified model to load weights
#     temp_unified_model = DCAE().to(device)
#     temp_unified_model.load_state_dict(unified_state_dict)
    
#     # Transfer encoder components to compress model
#     compress_model.g_a.load_state_dict(temp_unified_model.g_a.state_dict())
#     compress_model.h_a.load_state_dict(temp_unified_model.h_a.state_dict())
    
#     # Transfer decoder components to decompress model
#     decompress_model.g_s.load_state_dict(temp_unified_model.g_s.state_dict())
    
#     # Transfer shared components to both models
#     shared_components = [
#         'dt', 'h_z_s1', 'h_z_s2', 'cc_mean_transforms', 
#         'cc_scale_transforms', 'lrp_transforms', 'dt_cross_attention',
#         'entropy_bottleneck', 'gaussian_conditional'
#     ]
    
#     for component in shared_components:
#         if hasattr(temp_unified_model, component):
#             if component == 'dt':
#                 compress_model.dt.data = temp_unified_model.dt.data.clone()
#                 decompress_model.dt.data = temp_unified_model.dt.data.clone()
#             else:
#                 getattr(compress_model, component).load_state_dict(
#                     getattr(temp_unified_model, component).state_dict()
#                 )
#                 getattr(decompress_model, component).load_state_dict(
#                     getattr(temp_unified_model, component).state_dict()
#                 )
    
#     print("Successfully transferred pretrained weights to split models")
#     del temp_unified_model  # Clean up memory

# def full_pipeline_forward(compress_model, decompress_model, x):
#     """Simulate full pipeline for non-real mode evaluation"""
#     # Compression forward pass
#     compress_out = compress_model.forward(x)
    
#     # Extract the necessary outputs for decompression
#     y_hat = compress_out["y_hat"] 
#     z_hat = compress_out["z_hat"]
    
#     # Decompression forward pass
#     decompress_out = decompress_model.forward(y_hat, z_hat)
    
#     # Combine outputs for evaluation
#     combined_out = {
#         "x_hat": decompress_out["x_hat"],
#         "likelihoods": compress_out["likelihoods"]
#     }
    
#     return combined_out

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Example testing script for split DCAE models.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default='/home/xie/DCAE/60.5checkpoint_best.pth.tar')
#     parser.add_argument("--data", type=str, help="Path to dataset", default='/home/xie/DCAE/dataset/test')
#     parser.add_argument("--save_path", default='eval', type=str, help="Path to save")
#     parser.add_argument(
#         "--real", action="store_true", default=True
#     )
#     parser.set_defaults(real=False)
#     args = parser.parse_args(argv)
#     return args

# def main(argv):
#     torch.backends.cudnn.enabled = False
#     args = parse_args(argv)
#     p = 128
#     path = args.data
#     img_list = []
#     for file in os.listdir(path):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(file)
    
#     if args.cuda:
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
        
#     # Create split models
#     compress_model = CompressModel().to(device)
#     decompress_model = DecompressModel().to(device)
    
#     compress_model.eval()
#     decompress_model.eval()
    
#     count = 0
#     PSNR = 0
#     Bit_rate = 0
#     MS_SSIM = 0
#     total_time = 0
#     encode_time = 0
#     decode_time = 0
#     ave_flops = 0
#     encoder_time = 0
    
#     if args.checkpoint:  
#         load_pretrained_to_split_models(args.checkpoint, compress_model, decompress_model, device)
        
#     if args.real:
#         # Update entropy models for actual compression/decompression
#         compress_model.update()
#         decompress_model.update()
        
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
#             x = img.unsqueeze(0)
#             x_padded, padding = pad(x, p)
#             x_padded = x_padded.to(device)

#             count += 1
#             with torch.no_grad():
#                 # Compression phase
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_enc = compress_model.compress(x_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 encode_time += (e - s)
#                 total_time += (e - s)

#                 # Decompression phase
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_dec = decompress_model.decompress(out_enc["strings"], out_enc["shape"])
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 decode_time += (e - s)
#                 total_time += (e - s)

#                 out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
#                 num_pixels = x.size(0) * x.size(2) * x.size(3)
                
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 psnr = compute_psnr(x, out_dec["x_hat"])
#                 msssim = compute_msssim(x, out_dec["x_hat"])
                
#                 print(f'Bitrate: {bit_rate:.3f}bpp')
#                 print(f'MS-SSIM: {msssim:.2f}dB')
#                 print(f'PSNR: {psnr:.2f}dB')
                
#                 Bit_rate += bit_rate
#                 PSNR += psnr
#                 MS_SSIM += msssim
                
#     else:
#         # Forward pass mode (training-like evaluation)
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = Image.open(img_path).convert('RGB')
#             x = transforms.ToTensor()(img).unsqueeze(0).to(device)
#             x_padded, padding = pad(x, p)

#             count += 1
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
                
#                 # Use full pipeline simulation
#                 out_net = full_pipeline_forward(compress_model, decompress_model, x_padded)
                
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 total_time += (e - s)
                
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop(out_net["x_hat"], padding)
                
#                 psnr = compute_psnr(x, out_net["x_hat"])
#                 msssim = compute_msssim(x, out_net["x_hat"])
#                 bpp = compute_bpp(out_net)
                
#                 print(f'PSNR: {psnr:.2f}dB')
#                 print(f'MS-SSIM: {msssim:.2f}dB')
#                 print(f'Bit-rate: {bpp:.3f}bpp')
                
#                 PSNR += psnr
#                 MS_SSIM += msssim
#                 Bit_rate += bpp
                
#                 if args.save_path is not None:
#                     save_metrics(os.path.join(args.save_path, f"metrics_{img_name.split('.')[0]}.txt"), 
#                                psnr, bpp, msssim)
#                     save_image(out_net["x_hat"], os.path.join(args.save_path, f"decoded_{img_name}"))
    
#     # Calculate averages
#     PSNR = PSNR / count
#     MS_SSIM = MS_SSIM / count
#     Bit_rate = Bit_rate / count
#     total_time = total_time / count
#     encode_time = encode_time / count
#     decode_time = decode_time / count
#     ave_flops = ave_flops / count
#     encoder_time = encoder_time / count
    
#     print(f'average_encoder_time: {encoder_time:.3f} ms')
#     print(f'average_PSNR: {PSNR:.2f} dB')
#     print(f'average_MS-SSIM: {MS_SSIM:.4f}')
#     print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
#     print(f'average_time: {total_time:.3f} ms')
#     print(f'average_encode_time: {encode_time:.3f} ms')
#     print(f'average_decode_time: {decode_time:.3f} ms')
#     print(f'average_flops: {ave_flops:.3f}')

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])


# ## evaluate the pretrained model of dcae.py and save the results
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models.dcae_5 import (
#     CompressModel,
#     DecompressModel, 
#     ParameterSync
# )
# from models import DCAE
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
# import pickle
# warnings.filterwarnings("ignore")
# torch.set_num_threads(10)

# print(torch.cuda.is_available())

# def save_image(tensor, filename):
#     img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
#     img.save(filename)

# def save_compressed_data(strings, shape, filename):
#     """Save compressed data (strings and shape) to file"""
#     compressed_data = {
#         'strings': strings,
#         'shape': shape
#     }
#     with open(filename, 'wb') as f:
#         pickle.dump(compressed_data, f)

# def load_compressed_data(filename):
#     """Load compressed data from file"""
#     with open(filename, 'rb') as f:
#         compressed_data = pickle.load(f)
#     return compressed_data['strings'], compressed_data['shape']

# def save_metrics(filename, psnr, bitrate, msssim):
#     with open(filename, 'w') as f:
#         f.write(f'PSNR: {psnr:.2f}dB\n')
#         f.write(f'Bitrate: {bitrate:.3f}bpp\n')
#         f.write(f'MS-SSIM: {msssim:.4f}\n')

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2).item()
#     return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

# def compute_bpp(out_net):
#     size = out_net['x_hat'].size()
#     num_pixels = size[0] * size[2] * size[3]
#     return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
#               for likelihoods in out_net['likelihoods'].values()).item()

# def pad(x, p):
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

# def crop(x, padding):
#     return F.pad(
#         x,
#         (-padding[0], -padding[1], -padding[2], -padding[3]),
#     )

# def load_pretrained_to_split_models(checkpoint_path, compress_model, decompress_model, device):
#     """Load pretrained unified model weights to split models"""
#     print(f"Loading pretrained weights from {checkpoint_path}")
    
#     # Load unified model checkpoint
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     unified_state_dict = {}
    
#     # Clean up state dict keys
#     for k, v in checkpoint["state_dict"].items():
#         unified_state_dict[k.replace("module.", "")] = v
    
#     # Create temporary unified model to load weights
#     temp_unified_model = DCAE().to(device)
#     temp_unified_model.load_state_dict(unified_state_dict)
    
#     # Transfer encoder components to compress model
#     compress_model.g_a.load_state_dict(temp_unified_model.g_a.state_dict())
#     compress_model.h_a.load_state_dict(temp_unified_model.h_a.state_dict())
    
#     # Transfer decoder components to decompress model
#     decompress_model.g_s.load_state_dict(temp_unified_model.g_s.state_dict())
    
#     # Transfer shared components to both models
#     shared_components = [
#         'dt', 'h_z_s1', 'h_z_s2', 'cc_mean_transforms', 
#         'cc_scale_transforms', 'lrp_transforms', 'dt_cross_attention',
#         'entropy_bottleneck', 'gaussian_conditional'
#     ]
    
#     for component in shared_components:
#         if hasattr(temp_unified_model, component):
#             if component == 'dt':
#                 compress_model.dt.data = temp_unified_model.dt.data.clone()
#                 decompress_model.dt.data = temp_unified_model.dt.data.clone()
#             else:
#                 getattr(compress_model, component).load_state_dict(
#                     getattr(temp_unified_model, component).state_dict()
#                 )
#                 getattr(decompress_model, component).load_state_dict(
#                     getattr(temp_unified_model, component).state_dict()
#                 )
    
#     print("Successfully transferred pretrained weights to split models")
#     del temp_unified_model  # Clean up memory

# def full_pipeline_forward(compress_model, decompress_model, x):
#     """Simulate full pipeline for non-real mode evaluation"""
#     # Compression forward pass
#     compress_out = compress_model.forward(x)
    
#     # Extract the necessary outputs for decompression
#     y_hat = compress_out["y_hat"] 
#     z_hat = compress_out["z_hat"]
    
#     # Decompression forward pass
#     decompress_out = decompress_model.forward(y_hat, z_hat)
    
#     # Combine outputs for evaluation
#     combined_out = {
#         "x_hat": decompress_out["x_hat"],
#         "likelihoods": compress_out["likelihoods"]
#     }
    
#     return combined_out

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Example testing script for split DCAE models.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default='60.5checkpoint_best.pth.tar')
#     parser.add_argument("--data", type=str, help="Path to dataset", default='dataset/test')
#     parser.add_argument("--save_path", default='eval', type=str, help="Path to save")
#     parser.add_argument(
#         "--real", action="store_true"
#     )
#     parser.set_defaults(real=False)
#     args = parser.parse_args(argv)
#     return args

# def main(argv):
#     torch.backends.cudnn.enabled = False
#     args = parse_args(argv)
#     p = 128
#     path = args.data
#     img_list = []
#     for file in os.listdir(path):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(file)
    
#     if args.cuda:
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
        
#     # Create split models
#     compress_model = CompressModel().to(device)
#     decompress_model = DecompressModel().to(device)
    
#     compress_model.eval()
#     decompress_model.eval()
    
#     # Create output directories
#     compressed_dir = os.path.join(args.save_path,"compressed")
#     reconstructed_dir = os.path.join(args.save_path,"reconstructed")
#     os.makedirs(compressed_dir, exist_ok=True)
#     os.makedirs(reconstructed_dir, exist_ok=True)
    
#     count = 0
#     PSNR = 0
#     Bit_rate = 0
#     MS_SSIM = 0
#     total_time = 0
#     encode_time = 0
#     decode_time = 0
#     ave_flops = 0
#     encoder_time = 0
    
#     if args.checkpoint:  
#         load_pretrained_to_split_models(args.checkpoint, compress_model, decompress_model, device)
        
#     if args.real:
#         print("Real compression/decompression mode")
#         # Update entropy models for actual compression/decompression
#         compress_model.update()
#         decompress_model.update()
        
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
#             x = img.unsqueeze(0)
#             x_padded, padding = pad(x, p)
#             x_padded = x_padded.to(device)

#             count += 1
#             with torch.no_grad():
#                 # Compression phase
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_enc = compress_model.compress(x_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 encode_time += (e - s)
#                 total_time += (e - s)

#                 # Save compressed data
#                 img_base_name = img_name.split('.')[0]
#                 compressed_file = os.path.join(compressed_dir, f"{img_base_name}_compressed.pkl")
#                 save_compressed_data(out_enc["strings"], out_enc["shape"], compressed_file)
#                 print(f"Saved compressed data: {compressed_file}")

#                 # Decompression phase
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_dec = decompress_model.decompress(out_enc["strings"], out_enc["shape"])
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 decode_time += (e - s)
#                 total_time += (e - s)

#                 out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                
#                 # Save reconstructed image
#                 reconstructed_file = os.path.join(reconstructed_dir, f"{img_base_name}_reconstructed.png")
#                 save_image(out_dec["x_hat"], reconstructed_file)
#                 print(f"Saved reconstructed image: {reconstructed_file}")
                
#                 num_pixels = x.size(0) * x.size(2) * x.size(3)
                
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 psnr = compute_psnr(x, out_dec["x_hat"])
#                 msssim = compute_msssim(x, out_dec["x_hat"])
                
#                 print(f'Bitrate: {bit_rate:.3f}bpp')
#                 print(f'MS-SSIM: {msssim:.2f}dB')
#                 print(f'PSNR: {psnr:.2f}dB')
                
#                 Bit_rate += bit_rate
#                 PSNR += psnr
#                 MS_SSIM += msssim
                
#     else:
#         print('Forward pass mode')
#         # Forward pass mode (training-like evaluation)
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = Image.open(img_path).convert('RGB')
#             x = transforms.ToTensor()(img).unsqueeze(0).to(device)
#             x_padded, padding = pad(x, p)

#             count += 1
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
                
#                 # Use full pipeline simulation
#                 out_net = full_pipeline_forward(compress_model, decompress_model, x_padded)
                
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 total_time += (e - s)
                
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop(out_net["x_hat"], padding)
                
#                 # Save reconstructed image (forward mode)
#                 img_base_name = img_name.split('.')[0]
#                 reconstructed_file = os.path.join(reconstructed_dir, f"{img_base_name}_forward_reconstructed.png")
#                 save_image(out_net["x_hat"], reconstructed_file)
#                 print(f"Saved forward reconstructed image: {reconstructed_file}")
                
#                 psnr = compute_psnr(x, out_net["x_hat"])
#                 msssim = compute_msssim(x, out_net["x_hat"])
#                 bpp = compute_bpp(out_net)
                
#                 print(f'PSNR: {psnr:.2f}dB')
#                 print(f'MS-SSIM: {msssim:.2f}dB')
#                 print(f'Bit-rate: {bpp:.3f}bpp')
                
#                 PSNR += psnr
#                 MS_SSIM += msssim
#                 Bit_rate += bpp
                
#     # Calculate averages
#     PSNR = PSNR / count
#     MS_SSIM = MS_SSIM / count
#     Bit_rate = Bit_rate / count
#     total_time = total_time / count
#     encode_time = encode_time / count
#     decode_time = decode_time / count
#     ave_flops = ave_flops / count
#     encoder_time = encoder_time / count
    
#     print(f'\n=== EVALUATION RESULTS ===')
#     print(f'average_encoder_time: {encoder_time:.3f} ms')
#     print(f'average_PSNR: {PSNR:.2f} dB')
#     print(f'average_MS-SSIM: {MS_SSIM:.4f}')
#     print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
#     print(f'average_time: {total_time:.3f} ms')
#     print(f'average_encode_time: {encode_time:.3f} ms')
#     print(f'average_decode_time: {decode_time:.3f} ms')
#     print(f'average_flops: {ave_flops:.3f}')
    
#     print(f'\n=== SAVED FILES ===')
#     print(f'Compressed data saved to: {compressed_dir}/')
#     print(f'Reconstructed images saved to: {reconstructed_dir}/')

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])

## eval with pretrained model of dcae.py and save the results and metrics and split the model

# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models.dcae_5 import (
#     CompressModel,
#     DecompressModel, 
#     ParameterSync
# )
# from models import DCAE
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
# import pickle
# warnings.filterwarnings("ignore")
# torch.set_num_threads(10)

# print(torch.cuda.is_available())

# def save_image(tensor, filename):
#     img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
#     img.save(filename)

# def save_compressed_data(strings, shape, filename):
#     """Save compressed data (strings and shape) to file"""
#     compressed_data = {
#         'strings': strings,
#         'shape': shape
#     }
#     with open(filename, 'wb') as f:
#         pickle.dump(compressed_data, f)

# def load_compressed_data(filename):
#     """Load compressed data from file"""
#     with open(filename, 'rb') as f:
#         compressed_data = pickle.load(f)
#     return compressed_data['strings'], compressed_data['shape']

# def save_metrics(filename, psnr, bitrate, msssim):
#     with open(filename, 'w') as f:
#         f.write(f'PSNR: {psnr:.2f}dB\n')
#         f.write(f'Bitrate: {bitrate:.3f}bpp\n')
#         f.write(f'MS-SSIM: {msssim:.4f}\n')

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2).item()
#     return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

# def compute_bpp(out_net):
#     size = out_net['x_hat'].size()
#     num_pixels = size[0] * size[2] * size[3]
#     return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
#               for likelihoods in out_net['likelihoods'].values()).item()

# def pad(x, p):
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

# def crop(x, padding):
#     return F.pad(
#         x,
#         (-padding[0], -padding[1], -padding[2], -padding[3]),
#     )

# def transfer_tensors_to_device(data, target_device):
#     """Transfer tensors in nested structures to target device"""
#     if isinstance(data, torch.Tensor):
#         return data.to(target_device)
#     elif isinstance(data, dict):
#         return {k: transfer_tensors_to_device(v, target_device) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [transfer_tensors_to_device(item, target_device) for item in data]
#     elif isinstance(data, tuple):
#         return tuple(transfer_tensors_to_device(item, target_device) for item in data)
#     else:
#         return data

# def get_device(device_str):
#     """Convert device string to actual device"""
#     if device_str.lower() == 'cpu':
#         return torch.device('cpu')
#     elif device_str.lower().startswith('cuda'):
#         if torch.cuda.is_available():
#             return torch.device(device_str)
#         else:
#             print(f"Warning: CUDA not available, falling back to CPU")
#             return torch.device('cpu')
#     else:
#         raise ValueError(f"Invalid device: {device_str}")

# def load_pretrained_to_split_models(checkpoint_path, compress_model, decompress_model, compress_device, decompress_device):
#     """Load pretrained unified model weights to split models"""
#     print(f"Loading pretrained weights from {checkpoint_path}")
    
#     # Load unified model checkpoint on CPU first to avoid memory issues
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     unified_state_dict = {}
    
#     # Clean up state dict keys
#     for k, v in checkpoint["state_dict"].items():
#         unified_state_dict[k.replace("module.", "")] = v
    
#     # Create temporary unified model on CPU
#     temp_unified_model = DCAE()
#     temp_unified_model.load_state_dict(unified_state_dict)
    
#     # Transfer encoder components to compress model
#     compress_model.g_a.load_state_dict(temp_unified_model.g_a.state_dict())
#     compress_model.h_a.load_state_dict(temp_unified_model.h_a.state_dict())
    
#     # Transfer decoder components to decompress model
#     decompress_model.g_s.load_state_dict(temp_unified_model.g_s.state_dict())
    
#     # Transfer shared components to both models
#     shared_components = [
#         'dt', 'h_z_s1', 'h_z_s2', 'cc_mean_transforms', 
#         'cc_scale_transforms', 'lrp_transforms', 'dt_cross_attention',
#         'entropy_bottleneck', 'gaussian_conditional'
#     ]
    
#     for component in shared_components:
#         if hasattr(temp_unified_model, component):
#             if component == 'dt':
#                 compress_model.dt.data = temp_unified_model.dt.data.clone()
#                 decompress_model.dt.data = temp_unified_model.dt.data.clone()
#             else:
#                 getattr(compress_model, component).load_state_dict(
#                     getattr(temp_unified_model, component).state_dict()
#                 )
#                 getattr(decompress_model, component).load_state_dict(
#                     getattr(temp_unified_model, component).state_dict()
#                 )
    
#     print("Successfully transferred pretrained weights to split models")
#     print(f"Compression model device: {compress_device}")
#     print(f"Decompression model device: {decompress_device}")
#     del temp_unified_model  # Clean up memory

# def full_pipeline_forward(compress_model, decompress_model, x, compress_device, decompress_device):
#     """Simulate full pipeline for non-real mode evaluation with device handling"""
#     # Ensure input is on compression device
#     x = x.to(compress_device)
    
#     # Compression forward pass
#     compress_out = compress_model.forward(x)
    
#     # Extract the necessary outputs for decompression
#     y_hat = compress_out["y_hat"] 
#     z_hat = compress_out["z_hat"]
    
#     # Transfer latents to decompression device if different
#     if compress_device != decompress_device:
#         y_hat = y_hat.to(decompress_device)
#         z_hat = z_hat.to(decompress_device)
    
#     # Decompression forward pass
#     decompress_out = decompress_model.forward(y_hat, z_hat)
    
#     # Transfer output back to original device for evaluation
#     x_hat = decompress_out["x_hat"]
#     if decompress_device != compress_device:
#         x_hat = x_hat.to(compress_device)
    
#     # Combine outputs for evaluation
#     combined_out = {
#         "x_hat": x_hat,
#         "likelihoods": compress_out["likelihoods"]
#     }
    
#     return combined_out

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Example testing script for split DCAE models with device control.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda (deprecated, use --compress_device and --decompress_device)")
#     parser.add_argument("--compress_device", type=str, default="cpu", 
#                        help="Device for compression model (cpu, cuda, cuda:0, cuda:1, etc.)")
#     parser.add_argument("--decompress_device", type=str, default="cuda",
#                        help="Device for decompression model (cpu, cuda, cuda:0, cuda:1, etc.)")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default='/home/xie/DCAE/checkpoints/train_5/try_9/60.5/checkpoint_best.pth.tar')
#     parser.add_argument("--data", type=str, help="Path to dataset", default='/home/xie/datasets/dummy/valid')
#     parser.add_argument("--save_path", default='eval', type=str, help="Path to save")
#     parser.add_argument(
#         "--real", action="store_true"
#     )
#     parser.set_defaults(real=False)
#     args = parser.parse_args(argv)
#     return args

# def main(argv):
#     torch.backends.cudnn.enabled = False
#     args = parse_args(argv)
#     p = 128
#     path = args.data
#     img_list = []
#     for file in os.listdir(path):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(file)
    
#     # Handle device specification
#     if args.cuda:
#         print("Warning: --cuda flag is deprecated. Use --compress_device and --decompress_device instead.")
#         if args.compress_device == "cpu" and args.decompress_device == "cpu":
#             args.compress_device = "cuda"
#             args.decompress_device = "cuda"
    
#     # Get actual devices
#     compress_device = get_device(args.compress_device)
#     decompress_device = get_device(args.decompress_device)
    
#     print(f"Compression device: {compress_device}")
#     print(f"Decompression device: {decompress_device}")
        
#     # Create split models on respective devices
#     compress_model = CompressModel().to(compress_device)
#     decompress_model = DecompressModel().to(decompress_device)
    
#     compress_model.eval()
#     decompress_model.eval()
    
#     # Create output directories
#     compressed_dir = os.path.join(args.save_path,"compressed")
#     reconstructed_dir = os.path.join(args.save_path,"reconstructed")
#     os.makedirs(compressed_dir, exist_ok=True)
#     os.makedirs(reconstructed_dir, exist_ok=True)
    
#     count = 0
#     PSNR = 0
#     Bit_rate = 0
#     MS_SSIM = 0
#     total_time = 0
#     encode_time = 0
#     decode_time = 0
#     transfer_time = 0
#     ave_flops = 0
#     encoder_time = 0
    
#     if args.checkpoint:  
#         load_pretrained_to_split_models(args.checkpoint, compress_model, decompress_model, 
#                                        compress_device, decompress_device)
        
#     if args.real:
#         # Update entropy models for actual compression/decompression
#         compress_model.update()
#         decompress_model.update()
        
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
#             x = img.unsqueeze(0)
#             x_padded, padding = pad(x, p)
            
#             # Move input to compression device
#             x_padded = x_padded.to(compress_device)
#             x = x.to(compress_device)

#             count += 1
#             with torch.no_grad():
#                 # Compression phase
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 s = time.time()
#                 out_enc = compress_model.compress(x_padded)
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 e = time.time()
#                 encode_time += (e - s)
#                 total_time += (e - s)

#                 # Save compressed data
#                 img_base_name = img_name.split('.')[0]
#                 compressed_file = os.path.join(compressed_dir, f"{img_base_name}_compressed.pkl")
#                 save_compressed_data(out_enc["strings"], out_enc["shape"], compressed_file)
#                 print(f"Saved compressed data: {compressed_file}")

#                 # Data transfer time (if devices are different)
#                 transfer_start = time.time()
#                 # Note: strings are already on CPU, so no transfer needed for compressed data
#                 transfer_end = time.time()
#                 transfer_time += (transfer_end - transfer_start)

#                 # Decompression phase
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
#                 s = time.time()
#                 out_dec = decompress_model.decompress(out_enc["strings"], out_enc["shape"])
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
#                 e = time.time()
#                 decode_time += (e - s)
#                 total_time += (e - s)

#                 # Transfer result back to compression device for evaluation
#                 if compress_device != decompress_device:
#                     out_dec["x_hat"] = out_dec["x_hat"].to(compress_device)

#                 out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                
#                 # Save reconstructed image
#                 reconstructed_file = os.path.join(reconstructed_dir, f"{img_base_name}_reconstructed.png")
#                 save_image(out_dec["x_hat"], reconstructed_file)
#                 print(f"Saved reconstructed image: {reconstructed_file}")
                
#                 num_pixels = x.size(0) * x.size(2) * x.size(3)
                
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 psnr = compute_psnr(x, out_dec["x_hat"])
#                 msssim = compute_msssim(x, out_dec["x_hat"])
                
#                 print(f'Bitrate: {bit_rate:.3f}bpp')
#                 print(f'MS-SSIM: {msssim:.2f}dB')
#                 print(f'PSNR: {psnr:.2f}dB')
                
#                 Bit_rate += bit_rate
#                 PSNR += psnr
#                 MS_SSIM += msssim
                
#     else:
#         # Forward pass mode (training-like evaluation)
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = Image.open(img_path).convert('RGB')
#             x = transforms.ToTensor()(img).unsqueeze(0)
#             x_padded, padding = pad(x, p)

#             count += 1
#             with torch.no_grad():
#                 # Synchronize both devices if CUDA
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
                    
#                 s = time.time()
                
#                 # Use full pipeline simulation with device handling
#                 out_net = full_pipeline_forward(compress_model, decompress_model, x_padded, 
#                                               compress_device, decompress_device)
                
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
#                 e = time.time()
#                 total_time += (e - s)
                
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop(out_net["x_hat"], padding)
                
#                 # Ensure x is on the same device as output for comparison
#                 x = x.to(out_net["x_hat"].device)
                
#                 # Save reconstructed image (forward mode)
#                 img_base_name = img_name.split('.')[0]
#                 reconstructed_file = os.path.join(reconstructed_dir, f"{img_base_name}_forward_reconstructed.png")
#                 save_image(out_net["x_hat"], reconstructed_file)
#                 print(f"Saved forward reconstructed image: {reconstructed_file}")
                
#                 psnr = compute_psnr(x, out_net["x_hat"])
#                 msssim = compute_msssim(x, out_net["x_hat"])
#                 bpp = compute_bpp(out_net)
                
#                 print(f'PSNR: {psnr:.2f}dB')
#                 print(f'MS-SSIM: {msssim:.2f}dB')
#                 print(f'Bit-rate: {bpp:.3f}bpp')
                
#                 PSNR += psnr
#                 MS_SSIM += msssim
#                 Bit_rate += bpp
    
#     # Calculate averages
#     PSNR = PSNR / count
#     MS_SSIM = MS_SSIM / count
#     Bit_rate = Bit_rate / count
#     total_time = total_time / count
#     encode_time = encode_time / count
#     decode_time = decode_time / count
#     transfer_time = transfer_time / count
#     ave_flops = ave_flops / count
#     encoder_time = encoder_time / count
    
#     print(f'\n=== EVALUATION RESULTS ===')
#     print(f'Compression Device: {compress_device}')
#     print(f'Decompression Device: {decompress_device}')
#     print(f'average_encoder_time: {encoder_time:.3f} ms')
#     print(f'average_PSNR: {PSNR:.2f} dB')
#     print(f'average_MS-SSIM: {MS_SSIM:.4f}')
#     print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
#     print(f'average_total_time: {total_time:.3f} ms')
#     print(f'average_encode_time: {encode_time:.3f} ms')
#     print(f'average_decode_time: {decode_time:.3f} ms')
#     print(f'average_transfer_time: {transfer_time:.3f} ms')
#     print(f'average_flops: {ave_flops:.3f}')
    
#     print(f'\n=== SAVED FILES ===')
#     print(f'Compressed data saved to: {compressed_dir}/')
#     print(f'Reconstructed images saved to: {reconstructed_dir}/')

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])


# ## load split trained model
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models.dcae_5 import (
#     CompressModel,
#     DecompressModel, 
#     ParameterSync
# )
# from models import DCAE
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
# import pickle
# warnings.filterwarnings("ignore")
# torch.set_num_threads(10)

# print(torch.cuda.is_available())

# def save_image(tensor, filename):
#     img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
#     img.save(filename)

# def save_compressed_data(strings, shape, filename):
#     """Save compressed data (strings and shape) to file"""
#     compressed_data = {
#         'strings': strings,
#         'shape': shape
#     }
#     with open(filename, 'wb') as f:
#         pickle.dump(compressed_data, f)

# def load_compressed_data(filename):
#     """Load compressed data from file"""
#     with open(filename, 'rb') as f:
#         compressed_data = pickle.load(f)
#     return compressed_data['strings'], compressed_data['shape']

# def save_metrics(filename, psnr, bitrate, msssim):
#     with open(filename, 'w') as f:
#         f.write(f'PSNR: {psnr:.2f}dB\n')
#         f.write(f'Bitrate: {bitrate:.3f}bpp\n')
#         f.write(f'MS-SSIM: {msssim:.4f}\n')

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2).item()
#     return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

# def compute_bpp(out_net):
#     size = out_net['x_hat'].size()
#     num_pixels = size[0] * size[2] * size[3]
#     return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
#               for likelihoods in out_net['likelihoods'].values()).item()

# def pad(x, p):
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

# def crop(x, padding):
#     return F.pad(
#         x,
#         (-padding[0], -padding[1], -padding[2], -padding[3]),
#     )

# def transfer_tensors_to_device(data, target_device):
#     """Transfer tensors in nested structures to target device"""
#     if isinstance(data, torch.Tensor):
#         return data.to(target_device)
#     elif isinstance(data, dict):
#         return {k: transfer_tensors_to_device(v, target_device) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [transfer_tensors_to_device(item, target_device) for item in data]
#     elif isinstance(data, tuple):
#         return tuple(transfer_tensors_to_device(item, target_device) for item in data)
#     else:
#         return data

# def get_device(device_str):
#     """Convert device string to actual device"""
#     if device_str.lower() == 'cpu':
#         return torch.device('cpu')
#     elif device_str.lower().startswith('cuda'):
#         if torch.cuda.is_available():
#             return torch.device(device_str)
#         else:
#             print(f"Warning: CUDA not available, falling back to CPU")
#             return torch.device('cpu')
#     else:
#         raise ValueError(f"Invalid device: {device_str}")


# def load_pretrained_to_split_models(checkpoint_path, compress_model, decompress_model, compress_device, decompress_device):
#     """Load pretrained unified model weights to split models"""
#     print(f"Loading pretrained weights from {checkpoint_path}")
    
#     # Load checkpoint on CPU first to avoid memory issues
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
#     # Check if this is a split model checkpoint (from train_5.py) or unified model checkpoint
#     if 'compress_model' in checkpoint and 'decompress_model' in checkpoint:
#         # This is a split model checkpoint from train_5.py
#         print("Loading split model checkpoint...")
        
#         # Load compress model state
#         compress_state_dict = {}
#         for k, v in checkpoint['compress_model']['state_dict'].items():
#             compress_state_dict[k.replace("module.", "")] = v
#         compress_model.load_state_dict(compress_state_dict)
        
#         # Load decompress model state  
#         decompress_state_dict = {}
#         for k, v in checkpoint['decompress_model']['state_dict'].items():
#             decompress_state_dict[k.replace("module.", "")] = v
#         decompress_model.load_state_dict(decompress_state_dict)
        
#         print("Successfully loaded split model checkpoint")
        
#     elif 'state_dict' in checkpoint:
#         # This is a unified model checkpoint - use original logic
#         print("Loading unified model checkpoint...")
#         unified_state_dict = {}
        
#         # Clean up state dict keys
#         for k, v in checkpoint["state_dict"].items():
#             unified_state_dict[k.replace("module.", "")] = v
        
#         # Create temporary unified model on CPU
#         temp_unified_model = DCAE()
#         temp_unified_model.load_state_dict(unified_state_dict)
        
#         # Transfer encoder components to compress model
#         compress_model.g_a.load_state_dict(temp_unified_model.g_a.state_dict())
#         compress_model.h_a.load_state_dict(temp_unified_model.h_a.state_dict())
        
#         # Transfer decoder components to decompress model
#         decompress_model.g_s.load_state_dict(temp_unified_model.g_s.state_dict())
        
#         # Transfer shared components to both models
#         shared_components = [
#             'dt', 'h_z_s1', 'h_z_s2', 'cc_mean_transforms', 
#             'cc_scale_transforms', 'lrp_transforms', 'dt_cross_attention',
#             'entropy_bottleneck', 'gaussian_conditional'
#         ]
        
#         for component in shared_components:
#             if hasattr(temp_unified_model, component):
#                 if component == 'dt':
#                     compress_model.dt.data = temp_unified_model.dt.data.clone()
#                     decompress_model.dt.data = temp_unified_model.dt.data.clone()
#                 else:
#                     getattr(compress_model, component).load_state_dict(
#                         getattr(temp_unified_model, component).state_dict()
#                     )
#                     getattr(decompress_model, component).load_state_dict(
#                         getattr(temp_unified_model, component).state_dict()
#                     )
        
#         del temp_unified_model  # Clean up memory
#         print("Successfully transferred unified model weights to split models")
#     else:
#         raise ValueError("Checkpoint format not recognized. Expected either 'state_dict' key (unified model) or 'compress_model'/'decompress_model' keys (split model).")
    
#     print(f"Compression model device: {compress_device}")
#     print(f"Decompression model device: {decompress_device}")




# def full_pipeline_forward(compress_model, decompress_model, x, compress_device, decompress_device):
#     """Simulate full pipeline for non-real mode evaluation with device handling"""
#     # Ensure input is on compression device
#     x = x.to(compress_device)
    
#     # Compression forward pass
#     compress_out = compress_model.forward(x)
    
#     # Extract the necessary outputs for decompression
#     y_hat = compress_out["y_hat"] 
#     z_hat = compress_out["z_hat"]
    
#     # Transfer latents to decompression device if different
#     if compress_device != decompress_device:
#         y_hat = y_hat.to(decompress_device)
#         z_hat = z_hat.to(decompress_device)
    
#     # Decompression forward pass
#     decompress_out = decompress_model.forward(y_hat, z_hat)
    
#     # Transfer output back to original device for evaluation
#     x_hat = decompress_out["x_hat"]
#     if decompress_device != compress_device:
#         x_hat = x_hat.to(compress_device)
    
#     # Combine outputs for evaluation
#     combined_out = {
#         "x_hat": x_hat,
#         "likelihoods": compress_out["likelihoods"]
#     }
    
#     return combined_out

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Example testing script for split DCAE models with device control.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda (deprecated, use --compress_device and --decompress_device)")
#     parser.add_argument("--compress_device", type=str, default="cpu", 
#                        help="Device for compression model (cpu, cuda, cuda:0, cuda:1, etc.)")
#     parser.add_argument("--decompress_device", type=str, default="cuda",
#                        help="Device for decompression model (cpu, cuda, cuda:0, cuda:1, etc.)")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default='/home/xie/DCAE/checkpoints/train_5/try_9/60.5/checkpoint_best.pth.tar') #, /home/xie/DCAE/60.5checkpoint_best.pth.tar'
#     parser.add_argument("--data", type=str, help="Path to dataset", default='/home/xie/datasets/dummy/valid')
#     parser.add_argument("--save_path", default='eval', type=str, help="Path to save")
#     parser.add_argument(
#         "--real", action="store_true"
#     )
#     parser.set_defaults(real=False)
#     args = parser.parse_args(argv)
#     return args

# def main(argv):
#     torch.backends.cudnn.enabled = False
#     args = parse_args(argv)
#     p = 128
#     path = args.data
#     img_list = []
#     for file in os.listdir(path):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(file)
    
#     # Handle device specification
#     if args.cuda:
#         print("Warning: --cuda flag is deprecated. Use --compress_device and --decompress_device instead.")
#         if args.compress_device == "cpu" and args.decompress_device == "cpu":
#             args.compress_device = "cuda"
#             args.decompress_device = "cuda"
    
#     # Get actual devices
#     compress_device = get_device(args.compress_device)
#     decompress_device = get_device(args.decompress_device)
    
#     print(f"Compression device: {compress_device}")
#     print(f"Decompression device: {decompress_device}")
        
#     # Create split models on respective devices
#     compress_model = CompressModel().to(compress_device)
#     decompress_model = DecompressModel().to(decompress_device)
    
#     compress_model.eval()
#     decompress_model.eval()
    
#     # Create output directories
#     compressed_dir = os.path.join(args.save_path,"compressed")
#     reconstructed_dir = os.path.join(args.save_path,"reconstructed")
#     os.makedirs(compressed_dir, exist_ok=True)
#     os.makedirs(reconstructed_dir, exist_ok=True)
    
#     count = 0
#     PSNR = 0
#     Bit_rate = 0
#     MS_SSIM = 0
#     total_time = 0
#     encode_time = 0
#     decode_time = 0
#     transfer_time = 0
#     ave_flops = 0
    
#     if args.checkpoint:  
#         load_pretrained_to_split_models(args.checkpoint, compress_model, decompress_model, 
#                                        compress_device, decompress_device)
        
#     if args.real:
#         # Update entropy models for actual compression/decompression
#         compress_model.update()
#         decompress_model.update()
        
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
#             x = img.unsqueeze(0)
#             x_padded, padding = pad(x, p)
            
#             # Move input to compression device
#             x_padded = x_padded.to(compress_device)
#             x = x.to(compress_device)

#             count += 1
#             with torch.no_grad():
#                 # Compression phase
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 s = time.time()
#                 out_enc = compress_model.compress(x_padded)
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 e = time.time()
#                 encode_time += (e - s)
#                 total_time += (e - s)

#                 # Save compressed data
#                 img_base_name = img_name.split('.')[0]
#                 compressed_file = os.path.join(compressed_dir, f"{img_base_name}_compressed.pkl")
#                 save_compressed_data(out_enc["strings"], out_enc["shape"], compressed_file)
#                 print(f"Saved compressed data: {compressed_file}")

#                 # Data transfer time (if devices are different)
#                 transfer_start = time.time()
#                 # Note: strings are already on CPU, so no transfer needed for compressed data
#                 transfer_end = time.time()
#                 transfer_time += (transfer_end - transfer_start)

#                 # Decompression phase
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
#                 s = time.time()
#                 out_dec = decompress_model.decompress(out_enc["strings"], out_enc["shape"])
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
#                 e = time.time()
#                 decode_time += (e - s)
#                 total_time += (e - s)

#                 # Transfer result back to compression device for evaluation
#                 if compress_device != decompress_device:
#                     out_dec["x_hat"] = out_dec["x_hat"].to(compress_device)

#                 out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                
#                 # Save reconstructed image
#                 reconstructed_file = os.path.join(reconstructed_dir, f"{img_base_name}_reconstructed.png")
#                 save_image(out_dec["x_hat"], reconstructed_file)
#                 print(f"Saved reconstructed image: {reconstructed_file}")
                
#                 num_pixels = x.size(0) * x.size(2) * x.size(3)
                
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 psnr = compute_psnr(x, out_dec["x_hat"])
#                 msssim = compute_msssim(x, out_dec["x_hat"])
                
#                 print(f'Bitrate: {bit_rate:.3f}bpp')
#                 print(f'MS-SSIM: {msssim:.2f}dB')
#                 print(f'PSNR: {psnr:.2f}dB')
                
#                 Bit_rate += bit_rate
#                 PSNR += psnr
#                 MS_SSIM += msssim
                
#     else:
#         # Forward pass mode (training-like evaluation)
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = Image.open(img_path).convert('RGB')
#             x = transforms.ToTensor()(img).unsqueeze(0)
#             x_padded, padding = pad(x, p)

#             count += 1
#             with torch.no_grad():
#                 # Synchronize both devices if CUDA
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
                    
#                 s = time.time()
                
#                 # Use full pipeline simulation with device handling
#                 out_net = full_pipeline_forward(compress_model, decompress_model, x_padded, 
#                                               compress_device, decompress_device)
                
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
#                 e = time.time()
#                 total_time += (e - s)
                
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop(out_net["x_hat"], padding)
                
#                 # Ensure x is on the same device as output for comparison
#                 x = x.to(out_net["x_hat"].device)
                
#                 # Save reconstructed image (forward mode)
#                 img_base_name = img_name.split('.')[0]
#                 reconstructed_file = os.path.join(reconstructed_dir, f"{img_base_name}_forward_reconstructed.png")
#                 save_image(out_net["x_hat"], reconstructed_file)
#                 print(f"Saved forward reconstructed image: {reconstructed_file}")
                
#                 psnr = compute_psnr(x, out_net["x_hat"])
#                 msssim = compute_msssim(x, out_net["x_hat"])
#                 bpp = compute_bpp(out_net)
                
#                 print(f'PSNR: {psnr:.2f}dB')
#                 print(f'MS-SSIM: {msssim:.2f}dB')
#                 print(f'Bit-rate: {bpp:.3f}bpp')
                
#                 PSNR += psnr
#                 MS_SSIM += msssim
#                 Bit_rate += bpp
    
#     # Calculate averages
#     PSNR = PSNR / count
#     MS_SSIM = MS_SSIM / count
#     Bit_rate = Bit_rate / count
#     total_time = total_time / count
#     encode_time = encode_time / count
#     decode_time = decode_time / count
#     transfer_time = transfer_time / count
#     ave_flops = ave_flops / count
    
#     print(f'\n=== EVALUATION RESULTS ===')
#     print(f'Compression Device: {compress_device}')
#     print(f'Decompression Device: {decompress_device}')
#     print(f'average_PSNR: {PSNR:.2f} dB')
#     print(f'average_MS-SSIM: {MS_SSIM:.4f}')
#     print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
#     print(f'average_total_time: {total_time:.3f} ms')
#     print(f'average_encode_time: {encode_time:.3f} ms')
#     print(f'average_decode_time: {decode_time:.3f} ms')
#     print(f'average_transfer_time: {transfer_time:.3f} ms')
#     print(f'average_flops: {ave_flops:.3f}')
    
#     print(f'\n=== SAVED FILES ===')
#     print(f'Compressed data saved to: {compressed_dir}/')
#     print(f'Reconstructed images saved to: {reconstructed_dir}/')

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])


##

# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models.dcae_5 import (
#     CompressModel,
#     DecompressModel, 
#     ParameterSync
# )
# from models import DCAE
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
# import pickle
# warnings.filterwarnings("ignore")
# torch.set_num_threads(10)

# print(torch.cuda.is_available())

# def save_image(tensor, filename):
#     img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
#     img.save(filename)

# def save_compressed_data(strings, shape, filename):
#     """Save compressed data (strings and shape) to file"""
#     compressed_data = {
#         'strings': strings,
#         'shape': shape
#     }
#     with open(filename, 'wb') as f:
#         pickle.dump(compressed_data, f)

# def load_compressed_data(filename):
#     """Load compressed data from file"""
#     with open(filename, 'rb') as f:
#         compressed_data = pickle.load(f)
#     return compressed_data['strings'], compressed_data['shape']

# def save_metrics(filename, psnr, bitrate, msssim):
#     with open(filename, 'w') as f:
#         f.write(f'PSNR: {psnr:.2f}dB\n')
#         f.write(f'Bitrate: {bitrate:.3f}bpp\n')
#         f.write(f'MS-SSIM: {msssim:.4f}\n')

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2).item()
#     return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

# def compute_bpp(out_net):
#     size = out_net['x_hat'].size()
#     num_pixels = size[0] * size[2] * size[3]
#     return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
#               for likelihoods in out_net['likelihoods'].values()).item()

# def compute_compressed_size(strings, shape):
#     """Compute the size in bytes of compressed strings and shape"""
#     # Calculate size of strings
#     strings_size = 0
#     for string_list in strings:
#         for string in string_list:
#             strings_size += len(string)
    
#     # Calculate size of shape (serialized)
#     shape_size = sys.getsizeof(pickle.dumps(shape))
    
#     return strings_size, shape_size

# def pad(x, p):
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

# def crop(x, padding):
#     return F.pad(
#         x,
#         (-padding[0], -padding[1], -padding[2], -padding[3]),
#     )

# def transfer_tensors_to_device(data, target_device):
#     """Transfer tensors in nested structures to target device"""
#     if isinstance(data, torch.Tensor):
#         return data.to(target_device)
#     elif isinstance(data, dict):
#         return {k: transfer_tensors_to_device(v, target_device) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [transfer_tensors_to_device(item, target_device) for item in data]
#     elif isinstance(data, tuple):
#         return tuple(transfer_tensors_to_device(item, target_device) for item in data)
#     else:
#         return data

# def get_device(device_str):
#     """Convert device string to actual device"""
#     if device_str.lower() == 'cpu':
#         return torch.device('cpu')
#     elif device_str.lower().startswith('cuda'):
#         if torch.cuda.is_available():
#             return torch.device(device_str)
#         else:
#             print(f"Warning: CUDA not available, falling back to CPU")
#             return torch.device('cpu')
#     else:
#         raise ValueError(f"Invalid device: {device_str}")


# def load_pretrained_to_split_models(checkpoint_path, compress_model, decompress_model, compress_device, decompress_device):
#     """Load pretrained unified model weights to split models"""
#     print(f"Loading pretrained weights from {checkpoint_path}")
    
#     # Load checkpoint on CPU first to avoid memory issues
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
#     # Check if this is a split model checkpoint (from train_5.py) or unified model checkpoint
#     if 'compress_model' in checkpoint and 'decompress_model' in checkpoint:
#         # This is a split model checkpoint from train_5.py
#         print("Loading split model checkpoint...")
        
#         # Load compress model state
#         compress_state_dict = {}
#         for k, v in checkpoint['compress_model']['state_dict'].items():
#             compress_state_dict[k.replace("module.", "")] = v
#         compress_model.load_state_dict(compress_state_dict)
        
#         # Load decompress model state  
#         decompress_state_dict = {}
#         for k, v in checkpoint['decompress_model']['state_dict'].items():
#             decompress_state_dict[k.replace("module.", "")] = v
#         decompress_model.load_state_dict(decompress_state_dict)
        
#         print("Successfully loaded split model checkpoint")
        
#     elif 'state_dict' in checkpoint:
#         # This is a unified model checkpoint - use original logic
#         print("Loading unified model checkpoint...")
#         unified_state_dict = {}
        
#         # Clean up state dict keys
#         for k, v in checkpoint["state_dict"].items():
#             unified_state_dict[k.replace("module.", "")] = v
        
#         # Create temporary unified model on CPU
#         temp_unified_model = DCAE()
#         temp_unified_model.load_state_dict(unified_state_dict)
        
#         # Transfer encoder components to compress model
#         compress_model.g_a.load_state_dict(temp_unified_model.g_a.state_dict())
#         compress_model.h_a.load_state_dict(temp_unified_model.h_a.state_dict())
        
#         # Transfer decoder components to decompress model
#         decompress_model.g_s.load_state_dict(temp_unified_model.g_s.state_dict())
        
#         # Transfer shared components to both models
#         shared_components = [
#             'dt', 'h_z_s1', 'h_z_s2', 'cc_mean_transforms', 
#             'cc_scale_transforms', 'lrp_transforms', 'dt_cross_attention',
#             'entropy_bottleneck', 'gaussian_conditional'
#         ]
        
#         for component in shared_components:
#             if hasattr(temp_unified_model, component):
#                 if component == 'dt':
#                     compress_model.dt.data = temp_unified_model.dt.data.clone()
#                     decompress_model.dt.data = temp_unified_model.dt.data.clone()
#                 else:
#                     getattr(compress_model, component).load_state_dict(
#                         getattr(temp_unified_model, component).state_dict()
#                     )
#                     getattr(decompress_model, component).load_state_dict(
#                         getattr(temp_unified_model, component).state_dict()
#                     )
        
#         del temp_unified_model  # Clean up memory
#         print("Successfully transferred unified model weights to split models")
#     else:
#         raise ValueError("Checkpoint format not recognized. Expected either 'state_dict' key (unified model) or 'compress_model'/'decompress_model' keys (split model).")
    
#     print(f"Compression model device: {compress_device}")
#     print(f"Decompression model device: {decompress_device}")




# def full_pipeline_forward(compress_model, decompress_model, x, compress_device, decompress_device):
#     """Simulate full pipeline for non-real mode evaluation with device handling"""
#     # Ensure input is on compression device
#     x = x.to(compress_device)
    
#     # Compression forward pass
#     compress_out = compress_model.forward(x)
    
#     # Extract the necessary outputs for decompression
#     y_hat = compress_out["y_hat"] 
#     z_hat = compress_out["z_hat"]
    
#     # Transfer latents to decompression device if different
#     if compress_device != decompress_device:
#         y_hat = y_hat.to(decompress_device)
#         z_hat = z_hat.to(decompress_device)
    
#     # Decompression forward pass
#     decompress_out = decompress_model.forward(y_hat, z_hat)
    
#     # Transfer output back to original device for evaluation
#     x_hat = decompress_out["x_hat"]
#     if decompress_device != compress_device:
#         x_hat = x_hat.to(compress_device)
    
#     # Combine outputs for evaluation
#     combined_out = {
#         "x_hat": x_hat,
#         "likelihoods": compress_out["likelihoods"]
#     }
    
#     return combined_out

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Example testing script for split DCAE models with device control.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda (deprecated, use --compress_device and --decompress_device)")
#     parser.add_argument("--compress_device", type=str, default="cpu", 
#                        help="Device for compression model (cpu, cuda, cuda:0, cuda:1, etc.)")
#     parser.add_argument("--decompress_device", type=str, default="cuda",
#                        help="Device for decompression model (cpu, cuda, cuda:0, cuda:1, etc.)")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default='/home/xie/DCAE/checkpoints/train_5/try_9/60.5/checkpoint_best.pth.tar') #, /home/xie/DCAE/60.5checkpoint_best.pth.tar'
#     parser.add_argument("--data", type=str, help="Path to dataset", default='/home/xie/datasets/dummy/valid')
#     parser.add_argument("--save_path", default='eval', type=str, help="Path to save")
#     parser.add_argument(
#         "--real", action="store_true"
#     )
#     parser.set_defaults(real=False)
#     args = parser.parse_args(argv)
#     return args

# def main(argv):
#     torch.backends.cudnn.enabled = False
#     args = parse_args(argv)
#     p = 128
#     path = args.data
#     img_list = []
#     for file in os.listdir(path):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(file)
    
#     # Handle device specification
#     if args.cuda:
#         print("Warning: --cuda flag is deprecated. Use --compress_device and --decompress_device instead.")
#         if args.compress_device == "cpu" and args.decompress_device == "cpu":
#             args.compress_device = "cuda"
#             args.decompress_device = "cuda"
    
#     # Get actual devices
#     compress_device = get_device(args.compress_device)
#     decompress_device = get_device(args.decompress_device)
    
#     print(f"Compression device: {compress_device}")
#     print(f"Decompression device: {decompress_device}")
        
#     # Create split models on respective devices
#     compress_model = CompressModel().to(compress_device)
#     decompress_model = DecompressModel().to(decompress_device)
    
#     compress_model.eval()
#     decompress_model.eval()
    
#     # Create output directories
#     compressed_dir = os.path.join(args.save_path,"compressed")
#     reconstructed_dir = os.path.join(args.save_path,"reconstructed")
#     os.makedirs(compressed_dir, exist_ok=True)
#     os.makedirs(reconstructed_dir, exist_ok=True)
    
#     count = 0
#     PSNR = 0
#     Bit_rate = 0
#     MS_SSIM = 0
#     total_time = 0
#     encode_time = 0
#     decode_time = 0
#     transfer_time = 0
#     ave_flops = 0
    
#     # Add variables for compressed size tracking
#     total_strings_size = 0
#     total_shape_size = 0
#     total_compressed_size = 0
    
#     if args.checkpoint:  
#         load_pretrained_to_split_models(args.checkpoint, compress_model, decompress_model, 
#                                        compress_device, decompress_device)
        
#     if args.real:
#         # Update entropy models for actual compression/decompression
#         compress_model.update()
#         decompress_model.update()
        
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
#             x = img.unsqueeze(0)
#             x_padded, padding = pad(x, p)
            
#             # Move input to compression device
#             x_padded = x_padded.to(compress_device)
#             x = x.to(compress_device)

#             count += 1
#             with torch.no_grad():
#                 # Compression phase
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 s = time.time()
#                 out_enc = compress_model.compress(x_padded)
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 e = time.time()
#                 encode_time += (e - s)
#                 total_time += (e - s)

#                 # Calculate compressed sizes
#                 strings_size, shape_size = compute_compressed_size(out_enc["strings"], out_enc["shape"])
#                 print(out_enc["shape"], len(out_enc['strings']), type(out_enc['strings']), len(out_enc["strings"][0]), len(out_enc["strings"][1]))
#                 compressed_size = strings_size + shape_size
                
#                 total_strings_size += strings_size
#                 total_shape_size += shape_size
#                 total_compressed_size += compressed_size
                
#                 print(f"Strings size: {strings_size} bytes")
#                 print(f"Shape size: {shape_size} bytes")
#                 print(f"Total compressed size: {compressed_size} bytes")

#                 # Save compressed data
#                 img_base_name = img_name.split('.')[0]
#                 compressed_file = os.path.join(compressed_dir, f"{img_base_name}_compressed.pkl")
#                 save_compressed_data(out_enc["strings"], out_enc["shape"], compressed_file)
#                 print(f"Saved compressed data: {compressed_file}")

#                 # Data transfer time (if devices are different)
#                 transfer_start = time.time()
#                 # Note: strings are already on CPU, so no transfer needed for compressed data
#                 transfer_end = time.time()
#                 transfer_time += (transfer_end - transfer_start)

#                 # Decompression phase
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
#                 s = time.time()
#                 out_dec = decompress_model.decompress(out_enc["strings"], out_enc["shape"])
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
#                 e = time.time()
#                 decode_time += (e - s)
#                 total_time += (e - s)

#                 # Transfer result back to compression device for evaluation
#                 if compress_device != decompress_device:
#                     out_dec["x_hat"] = out_dec["x_hat"].to(compress_device)

#                 out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                
#                 # Save reconstructed image
#                 reconstructed_file = os.path.join(reconstructed_dir, f"{img_base_name}_reconstructed.png")
#                 save_image(out_dec["x_hat"], reconstructed_file)
#                 print(f"Saved reconstructed image: {reconstructed_file}")
                
#                 num_pixels = x.size(0) * x.size(2) * x.size(3)
                
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 psnr = compute_psnr(x, out_dec["x_hat"])
#                 msssim = compute_msssim(x, out_dec["x_hat"])
                
#                 print(f'Bitrate: {bit_rate:.3f}bpp')
#                 print(f'MS-SSIM: {msssim:.2f}dB')
#                 print(f'PSNR: {psnr:.2f}dB')
                
#                 Bit_rate += bit_rate
#                 PSNR += psnr
#                 MS_SSIM += msssim
                
#     else:
#         # Forward pass mode (training-like evaluation)
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = Image.open(img_path).convert('RGB')
#             x = transforms.ToTensor()(img).unsqueeze(0)
#             x_padded, padding = pad(x, p)

#             count += 1
#             with torch.no_grad():
#                 # Synchronize both devices if CUDA
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
                    
#                 s = time.time()
                
#                 # Use full pipeline simulation with device handling
#                 out_net = full_pipeline_forward(compress_model, decompress_model, x_padded, 
#                                               compress_device, decompress_device)
                
#                 if compress_device.type == 'cuda':
#                     torch.cuda.synchronize(compress_device)
#                 if decompress_device.type == 'cuda':
#                     torch.cuda.synchronize(decompress_device)
#                 e = time.time()
#                 total_time += (e - s)
                
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop(out_net["x_hat"], padding)
                
#                 # Ensure x is on the same device as output for comparison
#                 x = x.to(out_net["x_hat"].device)
                
#                 # Save reconstructed image (forward mode)
#                 img_base_name = img_name.split('.')[0]
#                 reconstructed_file = os.path.join(reconstructed_dir, f"{img_base_name}_forward_reconstructed.png")
#                 save_image(out_net["x_hat"], reconstructed_file)
#                 print(f"Saved forward reconstructed image: {reconstructed_file}")
                
#                 psnr = compute_psnr(x, out_net["x_hat"])
#                 msssim = compute_msssim(x, out_net["x_hat"])
#                 bpp = compute_bpp(out_net)
                
#                 print(f'PSNR: {psnr:.2f}dB')
#                 print(f'MS-SSIM: {msssim:.2f}dB')
#                 print(f'Bit-rate: {bpp:.3f}bpp')
                
#                 PSNR += psnr
#                 MS_SSIM += msssim
#                 Bit_rate += bpp
    
#     # Calculate averages
#     PSNR = PSNR / count
#     MS_SSIM = MS_SSIM / count
#     Bit_rate = Bit_rate / count
#     total_time = total_time / count
#     encode_time = encode_time / count
#     decode_time = decode_time / count
#     transfer_time = transfer_time / count
#     ave_flops = ave_flops / count
    
#     # Calculate average compressed sizes
#     if count > 0:
#         avg_strings_size = total_strings_size / count
#         avg_shape_size = total_shape_size / count
#         avg_compressed_size = total_compressed_size / count
#     else:
#         avg_strings_size = avg_shape_size = avg_compressed_size = 0
    
#     print(f'\n=== EVALUATION RESULTS ===')
#     print(f'Compression Device: {compress_device}')
#     print(f'Decompression Device: {decompress_device}')
#     print(f'average_PSNR: {PSNR:.2f} dB')
#     print(f'average_MS-SSIM: {MS_SSIM:.4f}')
#     print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
#     print(f'average_total_time: {total_time:.3f} ms')
#     print(f'average_encode_time: {encode_time:.3f} ms')
#     print(f'average_decode_time: {decode_time:.3f} ms')
#     print(f'average_transfer_time: {transfer_time:.3f} ms')
#     print(f'average_flops: {ave_flops:.3f}')
    
#     # Print compressed size statistics
#     print(f'\n=== COMPRESSED SIZE STATISTICS ===')
#     print(f'average_strings_size: {avg_strings_size:.2f} bytes')
#     print(f'average_shape_size: {avg_shape_size:.2f} bytes')
#     print(f'average_total_compressed_size: {avg_compressed_size:.2f} bytes')
    
#     print(f'\n=== SAVED FILES ===')
#     print(f'Compressed data saved to: {compressed_dir}/')
#     print(f'Reconstructed images saved to: {reconstructed_dir}/')

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])


## single compress, decompress and both of them

import torch
import torch.nn.functional as F
from torchvision import transforms
from models.dcae_5 import (
    CompressModel,
    DecompressModel, 
    ParameterSync
)
from models import DCAE
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
import pickle
warnings.filterwarnings("ignore")
torch.set_num_threads(10)

print(torch.cuda.is_available())

def save_image(tensor, filename):
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    img.save(filename)

def save_compressed_data(strings, shape, filename):
    """Save compressed data (strings and shape) to file"""
    compressed_data = {
        'strings': strings,
        'shape': shape
    }
    with open(filename, 'wb') as f:
        pickle.dump(compressed_data, f)

def load_compressed_data(filename):
    """Load compressed data from file"""
    with open(filename, 'rb') as f:
        compressed_data = pickle.load(f)
    return compressed_data['strings'], compressed_data['shape']

def save_metrics(filename, psnr, bitrate, msssim):
    with open(filename, 'w') as f:
        f.write(f'PSNR: {psnr:.2f}dB\n')
        f.write(f'Bitrate: {bitrate:.3f}bpp\n')
        f.write(f'MS-SSIM: {msssim:.4f}\n')

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def compute_compressed_size(strings, shape):
    """Compute the size in bytes of compressed strings and shape"""
    # Calculate size of strings
    strings_size = 0
    for string_list in strings:
        for string in string_list:
            strings_size += len(string)
    
    # Calculate size of shape (serialized)
    shape_size = sys.getsizeof(pickle.dumps(shape))
    
    return strings_size, shape_size

def pad(x, p):
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

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def transfer_tensors_to_device(data, target_device):
    """Transfer tensors in nested structures to target device"""
    if isinstance(data, torch.Tensor):
        return data.to(target_device)
    elif isinstance(data, dict):
        return {k: transfer_tensors_to_device(v, target_device) for k, v in data.items()}
    elif isinstance(data, list):
        return [transfer_tensors_to_device(item, target_device) for item in data]
    elif isinstance(data, tuple):
        return tuple(transfer_tensors_to_device(item, target_device) for item in data)
    else:
        return data

def get_device(device_str):
    """Convert device string to actual device"""
    if device_str.lower() == 'cpu':
        return torch.device('cpu')
    elif device_str.lower().startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            print(f"Warning: CUDA not available, falling back to CPU")
            return torch.device('cpu')
    else:
        raise ValueError(f"Invalid device: {device_str}")


def load_pretrained_to_split_models(checkpoint_path, compress_model, decompress_model, compress_device, decompress_device):
    """Load pretrained unified model weights to split models"""
    print(f"Loading pretrained weights from {checkpoint_path}")
    
    # Load checkpoint on CPU first to avoid memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if this is a split model checkpoint (from train_5.py) or unified model checkpoint
    if 'compress_model' in checkpoint and 'decompress_model' in checkpoint:
        # This is a split model checkpoint from train_5.py
        print("Loading split model checkpoint...")
        
        # Load compress model state
        if compress_model is not None:
            compress_state_dict = {}
            for k, v in checkpoint['compress_model']['state_dict'].items():
                compress_state_dict[k.replace("module.", "")] = v
            compress_model.load_state_dict(compress_state_dict)
        
        # Load decompress model state  
        if decompress_model is not None:
            decompress_state_dict = {}
            for k, v in checkpoint['decompress_model']['state_dict'].items():
                decompress_state_dict[k.replace("module.", "")] = v
            decompress_model.load_state_dict(decompress_state_dict)
        
        print("Successfully loaded split model checkpoint")
        
    elif 'state_dict' in checkpoint:
        # This is a unified model checkpoint - use original logic
        print("Loading unified model checkpoint...")
        unified_state_dict = {}
        
        # Clean up state dict keys
        for k, v in checkpoint["state_dict"].items():
            unified_state_dict[k.replace("module.", "")] = v
        
        # Create temporary unified model on CPU
        temp_unified_model = DCAE()
        temp_unified_model.load_state_dict(unified_state_dict)
        
        # Transfer encoder components to compress model
        if compress_model is not None:
            compress_model.g_a.load_state_dict(temp_unified_model.g_a.state_dict())
            compress_model.h_a.load_state_dict(temp_unified_model.h_a.state_dict())
        
        # Transfer decoder components to decompress model
        if decompress_model is not None:
            decompress_model.g_s.load_state_dict(temp_unified_model.g_s.state_dict())
        
        # Transfer shared components to both models
        shared_components = [
            'dt', 'h_z_s1', 'h_z_s2', 'cc_mean_transforms', 
            'cc_scale_transforms', 'lrp_transforms', 'dt_cross_attention',
            'entropy_bottleneck', 'gaussian_conditional'
        ]
        
        for component in shared_components:
            if hasattr(temp_unified_model, component):
                if component == 'dt':
                    if compress_model is not None:
                        compress_model.dt.data = temp_unified_model.dt.data.clone()
                    if decompress_model is not None:
                        decompress_model.dt.data = temp_unified_model.dt.data.clone()
                else:
                    if compress_model is not None and hasattr(compress_model, component):
                        getattr(compress_model, component).load_state_dict(
                            getattr(temp_unified_model, component).state_dict()
                        )
                    if decompress_model is not None and hasattr(decompress_model, component):
                        getattr(decompress_model, component).load_state_dict(
                            getattr(temp_unified_model, component).state_dict()
                        )
        
        del temp_unified_model  # Clean up memory
        print("Successfully transferred unified model weights to split models")
    else:
        raise ValueError("Checkpoint format not recognized. Expected either 'state_dict' key (unified model) or 'compress_model'/'decompress_model' keys (split model).")
    
    if compress_model is not None:
        print(f"Compression model device: {compress_device}")
    if decompress_model is not None:
        print(f"Decompression model device: {decompress_device}")


def full_pipeline_forward(compress_model, decompress_model, x, compress_device, decompress_device):
    """Simulate full pipeline for non-real mode evaluation with device handling"""
    # Ensure input is on compression device
    x = x.to(compress_device)
    
    # Compression forward pass
    compress_out = compress_model.forward(x)
    
    # Extract the necessary outputs for decompression
    y_hat = compress_out["y_hat"] 
    z_hat = compress_out["z_hat"]
    
    # Transfer latents to decompression device if different
    if compress_device != decompress_device:
        y_hat = y_hat.to(decompress_device)
        z_hat = z_hat.to(decompress_device)
    
    # Decompression forward pass
    decompress_out = decompress_model.forward(y_hat, z_hat)
    
    # Transfer output back to original device for evaluation
    x_hat = decompress_out["x_hat"]
    if decompress_device != compress_device:
        x_hat = x_hat.to(compress_device)
    
    # Combine outputs for evaluation
    combined_out = {
        "x_hat": x_hat,
        "likelihoods": compress_out["likelihoods"]
    }
    
    return combined_out

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script for split DCAE models with device control.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda (deprecated, use --compress_device and --decompress_device)")
    parser.add_argument("--compress_device", type=str, default="cpu", 
                       help="Device for compression model (cpu, cuda, cuda:0, cuda:1, etc.)")
    parser.add_argument("--decompress_device", type=str, default="cuda",
                       help="Device for decompression model (cpu, cuda, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default='/home/xie/DCAE/checkpoints/train_5/try_9/60.5/checkpoint_best.pth.tar')
    parser.add_argument("--data", type=str, help="Path to dataset", default='/home/xie/datasets/dummy/valid')
    parser.add_argument("--save_path", default='eval', type=str, help="Path to save")
    parser.add_argument("--compressed_data_path", default='eval/compressed',type=str, help="Path to compressed data for decompress-only mode")
    parser.add_argument(
        "--real", action="store_true"
    )
    parser.add_argument(
        "--mode", type=str, choices=['compress', 'decompress', 'both'], default='both',
        help="Operation mode: 'compress' (compress only), 'decompress' (decompress only), 'both' (compress and decompress)"
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args

def main(argv):
    torch.backends.cudnn.enabled = False
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)
    
    # Handle device specification
    if args.cuda:
        print("Warning: --cuda flag is deprecated. Use --compress_device and --decompress_device instead.")
        if args.compress_device == "cpu" and args.decompress_device == "cpu":
            args.compress_device = "cuda"
            args.decompress_device = "cuda"
    
    # Get actual devices
    compress_device = get_device(args.compress_device)
    decompress_device = get_device(args.decompress_device)
    
    print(f"Operation mode: {args.mode}")
    if args.mode in ['compress', 'both']:
        print(f"Compression device: {compress_device}")
    if args.mode in ['decompress', 'both']:
        print(f"Decompression device: {decompress_device}")
        
    # Create split models based on mode
    compress_model = None
    decompress_model = None
    
    if args.mode in ['compress', 'both']:
        compress_model = CompressModel().to(compress_device)
        compress_model.eval()
    
    if args.mode in ['decompress', 'both']:
        decompress_model = DecompressModel().to(decompress_device)
        decompress_model.eval()
    
    # Create output directories
    compressed_dir = os.path.join(args.save_path, "compressed")
    reconstructed_dir = os.path.join(args.save_path, "reconstructed")
    os.makedirs(compressed_dir, exist_ok=True)
    os.makedirs(reconstructed_dir, exist_ok=True)
    
    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    encode_time = 0
    decode_time = 0
    transfer_time = 0
    ave_flops = 0
    
    # Add variables for compressed size tracking
    total_strings_size = 0
    total_shape_size = 0
    total_compressed_size = 0
    
    if args.checkpoint:  
        load_pretrained_to_split_models(args.checkpoint, compress_model, decompress_model, 
                                       compress_device, decompress_device)
        
    if args.real:
        # Update entropy models for actual compression/decompression
        if compress_model is not None:
            compress_model.update()
        if decompress_model is not None:
            decompress_model.update()
        
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
            x = img.unsqueeze(0)
            x_padded, padding = pad(x, p)
            
            img_base_name = img_name.split('.')[0]
            compressed_file = os.path.join(compressed_dir, f"{img_base_name}_compressed.pkl")
            
            count += 1
            with torch.no_grad():
                # COMPRESS PHASE
                if args.mode in ['compress', 'both']:
                    # Move input to compression device
                    x_padded = x_padded.to(compress_device)
                    x = x.to(compress_device)

                    # Compression phase
                    if compress_device.type == 'cuda':
                        torch.cuda.synchronize(compress_device)
                    s = time.time()
                    out_enc = compress_model.compress(x_padded)
                    if compress_device.type == 'cuda':
                        torch.cuda.synchronize(compress_device)
                    e = time.time()
                    encode_time += (e - s)
                    total_time += (e - s)

                    # Calculate compressed sizes
                    strings_size, shape_size = compute_compressed_size(out_enc["strings"], out_enc["shape"])
                    print(f"Strings size: {strings_size} bytes")
                    print(f"Shape size: {shape_size} bytes")
                    print(f"Total compressed size: {strings_size + shape_size} bytes")

                    total_strings_size += strings_size
                    total_shape_size += shape_size
                    total_compressed_size += (strings_size + shape_size)

                    # Save compressed data
                    save_compressed_data(out_enc["strings"], out_enc["shape"], compressed_file)
                    print(f"Saved compressed data: {compressed_file}")

                    if args.mode == 'compress':
                        print(f"Compression completed for {img_name}")
                        continue  # Skip decompression

                # DECOMPRESS PHASE
                if args.mode in ['decompress', 'both']:
                    # Load compressed data if in decompress-only mode
                    if args.mode == 'decompress':
                        if args.compressed_data_path:
                            # Load from specified directory
                            compressed_file = os.path.join(args.compressed_data_path, f"{img_base_name}_compressed.pkl")
                        else:
                            # Load from default compressed directory
                            compressed_file = os.path.join(compressed_dir, f"{img_base_name}_compressed.pkl")
                        
                        if not os.path.exists(compressed_file):
                            print(f"Warning: Compressed file not found: {compressed_file}")
                            continue
                        
                        strings, shape = load_compressed_data(compressed_file)
                        print(f"Loaded compressed data from: {compressed_file}")
                        
                        # Calculate original image dimensions for padding info
                        # For decompress-only mode, we need to handle padding differently
                        # You might want to save padding info with compressed data
                        # For now, we'll assume standard padding
                        img_temp = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
                        x_temp = img_temp.unsqueeze(0)
                        _, padding = pad(x_temp, p)
                        x = x_temp  # For PSNR calculation later
                    else:
                        # Use data from compression phase
                        strings = out_enc["strings"]
                        shape = out_enc["shape"]

                    # Data transfer time (if devices are different)
                    transfer_start = time.time()
                    # Note: strings are already on CPU, so no transfer needed for compressed data
                    transfer_end = time.time()
                    transfer_time += (transfer_end - transfer_start)

                    # Decompression phase
                    if decompress_device.type == 'cuda':
                        torch.cuda.synchronize(decompress_device)
                    s = time.time()
                    out_dec = decompress_model.decompress(strings, shape)
                    if decompress_device.type == 'cuda':
                        torch.cuda.synchronize(decompress_device)
                    e = time.time()
                    decode_time += (e - s)
                    total_time += (e - s)

                    # Transfer result back to compression device for evaluation
                    if compress_device and compress_device != decompress_device:
                        out_dec["x_hat"] = out_dec["x_hat"].to(compress_device)
                    elif not compress_device:
                        # For decompress-only mode, keep on decompress device
                        pass

                    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                    
                    # Save reconstructed image
                    reconstructed_file = os.path.join(reconstructed_dir, f"{img_base_name}_reconstructed.png")
                    save_image(out_dec["x_hat"], reconstructed_file)
                    print(f"Saved reconstructed image: {reconstructed_file}")
                    
                    # Calculate metrics only if we have both original and reconstructed
                    if args.mode in ['decompress', 'both']:
                        # Ensure x is on the same device as reconstructed image
                        x = x.to(out_dec["x_hat"].device)
                        
                        if args.mode == 'both':
                            # Calculate bitrate from compression
                            num_pixels = x.size(0) * x.size(2) * x.size(3)
                            bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                        else:
                            # For decompress-only, we can't calculate exact bitrate without compression info
                            # You might want to save this info with compressed data
                            bit_rate = 0.0
                        
                        psnr = compute_psnr(x, out_dec["x_hat"])
                        msssim = compute_msssim(x, out_dec["x_hat"])
                        
                        print(f'Bitrate: {bit_rate:.3f}bpp')
                        print(f'MS-SSIM: {msssim:.2f}dB')
                        print(f'PSNR: {psnr:.2f}dB')
                        
                        Bit_rate += bit_rate
                        PSNR += psnr
                        MS_SSIM += msssim
                    
    else:
        # Forward pass mode (training-like evaluation) - only works for 'both' mode
        if args.mode != 'both':
            print("Forward pass mode (--real=False) only supports 'both' mode")
            return
            
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert('RGB')
            x = transforms.ToTensor()(img).unsqueeze(0)
            x_padded, padding = pad(x, p)

            count += 1
            with torch.no_grad():
                # Synchronize both devices if CUDA
                if compress_device.type == 'cuda':
                    torch.cuda.synchronize(compress_device)
                if decompress_device.type == 'cuda':
                    torch.cuda.synchronize(decompress_device)
                    
                s = time.time()
                
                # Use full pipeline simulation with device handling
                out_net = full_pipeline_forward(compress_model, decompress_model, x_padded, 
                                              compress_device, decompress_device)
                
                if compress_device.type == 'cuda':
                    torch.cuda.synchronize(compress_device)
                if decompress_device.type == 'cuda':
                    torch.cuda.synchronize(decompress_device)
                e = time.time()
                total_time += (e - s)
                
                out_net['x_hat'].clamp_(0, 1)
                out_net["x_hat"] = crop(out_net["x_hat"], padding)
                
                # Ensure x is on the same device as output for comparison
                x = x.to(out_net["x_hat"].device)
                
                # Save reconstructed image (forward mode)
                img_base_name = img_name.split('.')[0]
                reconstructed_file = os.path.join(reconstructed_dir, f"{img_base_name}_forward_reconstructed.png")
                save_image(out_net["x_hat"], reconstructed_file)
                print(f"Saved forward reconstructed image: {reconstructed_file}")
                
                psnr = compute_psnr(x, out_net["x_hat"])
                msssim = compute_msssim(x, out_net["x_hat"])
                bpp = compute_bpp(out_net)
                
                print(f'PSNR: {psnr:.2f}dB')
                print(f'MS-SSIM: {msssim:.2f}dB')
                print(f'Bit-rate: {bpp:.3f}bpp')
                
                PSNR += psnr
                MS_SSIM += msssim
                Bit_rate += bpp
    
    # Calculate averages
    if count > 0:
        PSNR = PSNR / count
        MS_SSIM = MS_SSIM / count
        Bit_rate = Bit_rate / count
        total_time = total_time / count
        encode_time = encode_time / count
        decode_time = decode_time / count
        transfer_time = transfer_time / count
        ave_flops = ave_flops / count
        
        # Calculate average compressed sizes
        avg_strings_size = total_strings_size / count
        avg_shape_size = total_shape_size / count
        avg_compressed_size = total_compressed_size / count
    else:
        avg_strings_size = avg_shape_size = avg_compressed_size = 0
    
    print(f'\n=== EVALUATION RESULTS ===')
    print(f'Operation Mode: {args.mode}')
    if compress_model is not None:
        print(f'Compression Device: {compress_device}')
    if decompress_model is not None:
        print(f'Decompression Device: {decompress_device}')
    
    if args.mode in ['decompress', 'both']:
        print(f'average_PSNR: {PSNR:.2f} dB')
        print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    if args.mode in ['compress', 'both']:
        print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    
    print(f'average_total_time: {total_time:.3f} ms')
    if args.mode in ['compress', 'both']:
        print(f'average_encode_time: {encode_time:.3f} ms')
    if args.mode in ['decompress', 'both']:
        print(f'average_decode_time: {decode_time:.3f} ms')
    print(f'average_transfer_time: {transfer_time:.3f} ms')
    print(f'average_flops: {ave_flops:.3f}')
    
    # Print compressed size statistics
    if args.mode in ['compress', 'both']:
        print(f'\n=== COMPRESSED SIZE STATISTICS ===')
        print(f'average_strings_size: {avg_strings_size:.2f} bytes')
        print(f'average_shape_size: {avg_shape_size:.2f} bytes')
        print(f'average_total_compressed_size: {avg_compressed_size:.2f} bytes')
    
    print(f'\n=== SAVED FILES ===')
    if args.mode in ['compress', 'both']:
        print(f'Compressed data saved to: {compressed_dir}/')
    if args.mode in ['decompress', 'both']:
        print(f'Reconstructed images saved to: {reconstructed_dir}/')

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])