import os
import argparse
import glob
import torch
import math

def format_size(size_bytes):
    """Formats bytes into KB or MB"""
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def main(args):
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return

    bin_files = sorted(glob.glob(os.path.join(args.input, "*.bin")))
    
    if not bin_files:
        print(f"No .bin files found in {args.input}")
        return

    print(f"{'Filename':<30} | {'Size':<10} | {'Pixels':<15} | {'BPP':<10}")
    print("-" * 75)

    total_bytes = 0
    total_pixels = 0
    count = 0

    for filepath in bin_files:
        filename = os.path.basename(filepath)
        
        # 1. Get raw file size
        file_size = os.path.getsize(filepath)
        total_bytes += file_size
        
        # 2. Load metadata to calculate BPP
        # We try to load the file to get dimensions. 
        # If it's a raw stream without metadata, we skip BPP.
        pixel_count = 0
        bpp_str = "N/A"
        
        try:
            # Load lightly (map_location cpu) just to read metadata
            data = torch.load(filepath, map_location='cpu')
            
            if 'original_shape' in data:
                h, w = data['original_shape']
                pixel_count = h * w
                total_pixels += pixel_count
                
                # Formula: (Bytes * 8) / (Height * Width)
                bpp = (file_size * 8) / pixel_count
                bpp_str = f"{bpp:.4f}"
            elif 'shape' in data:
                 # Fallback if shape is stored differently
                h, w = data['shape'][-2:]
                pixel_count = h * w
                total_pixels += pixel_count
                bpp = (file_size * 8) / pixel_count
                bpp_str = f"{bpp:.4f}"
                
        except Exception as e:
            # If torch load fails, it might be a raw binary string
            bpp_str = "Err"

        pixel_str = f"{pixel_count} px" if pixel_count > 0 else "Unknown"
        print(f"{filename[:27]:<30} | {format_size(file_size):<10} | {pixel_str:<15} | {bpp_str:<10}")
        count += 1

    print("-" * 75)
    print("SUMMARY")
    print(f"Total Files: {count}")
    print(f"Total Size : {format_size(total_bytes)}")
    
    if total_pixels > 0:
        avg_bpp = (total_bytes * 8) / total_pixels
        print(f"Average BPP: {avg_bpp:.4f}")
        
        # Assuming original is RGB (24 bits), calculate compression ratio
        # CR = Original Size / Compressed Size
        # Original Size approx = Total Pixels * 3 bytes (24 bits)
        original_size_bytes = total_pixels * 3
        compression_ratio = original_size_bytes / total_bytes
        print(f"Compression Ratio: {compression_ratio:.2f}x (Original is ~{compression_ratio:.0f} times larger)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate File Sizes and BPP")
    parser.add_argument("-i", "--input", type=str, default='./compressed', help="Folder containing .bin files")
    
    args = parser.parse_args()
    main(args)