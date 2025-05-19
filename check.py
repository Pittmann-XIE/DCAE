## Check the average size of the images and the bin files
# import os

# def average_file_size(folder_path, file_extension=None):
#     total_size = 0
#     file_count = 0
    
#     for filename in os.listdir(folder_path):
#         if file_extension is None or filename.endswith(file_extension):
#             file_path = os.path.join(folder_path, filename)
#             if os.path.isfile(file_path):
#                 total_size += os.path.getsize(file_path)
#                 file_count += 1
                
#     if file_count == 0:
#         return 0
#     return total_size / file_count / 1024  # Convert bytes to KB


# import cv2

# def average_image_size(folder_path):
#     total_size = 0
#     file_count = 0
#     image_sizes = []

#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Common image formats
#             file_path = os.path.join(folder_path, filename)
#             if os.path.isfile(file_path):
#                 # Get image size using cv2
#                 img = cv2.imread(file_path)
#                 if img is not None:
#                     height, width, _ = img.shape
#                     image_sizes.append((width, height))  # Append (width, height)
                
#                 total_size += os.path.getsize(file_path)
#                 file_count += 1

#     if file_count == 0:
#         return 0, []

#     average_size_kb = total_size / file_count / 1024  # Convert bytes to KB
#     return average_size_kb, image_sizes



# image_size_kB, image_size = average_image_size("../datasets/dummy/valid")
# print(f"Average size of images: {image_size_kB:.2f} KB, the size of the images: {image_size[0][0]}x{image_size[0][1]}")

# image_size_kB, image_size = average_image_size("./output/reconstructed")
# print(f"Average size of reconstructed images: {image_size_kB:.2f} KB, the size of the images: {image_size[0][0]}x{image_size[0][1]}")


# average_bin_size = average_file_size("./output/binary/bin", file_extension=".bin")
# print(f"Average size of .bin files in 'bins': {average_bin_size:.2f} KB")


## Run on cpu and gpu separately
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from models import (
#     DCAE,
# )
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
# import pandas as pd
# import numpy as np
# import torch.nn as nn
# import struct
# warnings.filterwarnings("ignore")

# print(torch.cuda.is_available())

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2).item()
#     return -10 * math.log10(mse)

# def compute_msssim(a, b):
#     return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

# def compute_bpp(out_net):
#     size = out_net['x_hat'].size()
#     num_pixels = size[0] * size[2] * size[3]
#     bpp_dict = {}
#     ans = 0
#     for likelihoods in out_net['likelihoods']:
#         fsize = out_net['likelihoods'][likelihoods].size()
#         num = 1
#         for s in fsize:
#             num = num * s
#         bpp_dict[likelihoods] = torch.log(out_net['likelihoods'][likelihoods]).sum() / (-math.log(2) * num_pixels)
#         ans = ans + bpp_dict[likelihoods]
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
#     padding = (padding_left, padding_right, padding_top, padding_bottom)
#     # pad_layer = nn.ReflectionPad2d(padding)
#     # x_padded = pad_layer(x)
#     x_padded = F.pad(
#         x,
#         (padding_left, padding_right, padding_top, padding_bottom),
#         mode="constant",
#         value=0,
#     )
#     return x_padded, padding

# def crop(x, padding):
#     return F.pad(
#         x,
#         (-padding[0], -padding[1], -padding[2], -padding[3]),
#     )



# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Example testing script.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument(
#         "--N", type=int, default=128,
#     )
#     parser.add_argument(
#         "--M", type=int, default=320,
#     )
#     parser.add_argument("--checkpoint", type=str, default="./60.5checkpoint_best.pth.tar", help="Path to a checkpoint")
#     parser.add_argument("--save_path_en", type=str, default="./output/binary", help="Path to save")
#     parser.add_argument("--data_en", type=str, default="../datasets/dummy/valid", help="Path to dataset")
#     parser.add_argument("--save_path_de", type=str, default="./output/reconstructed", help="Path to save")
#     parser.add_argument("--data_de", type=str, default="./output/binary/bin", help="Path to save")

#     parser.add_argument("--mode", type=str, choices=['compress', "decompress"])
#     parser.add_argument(
#         "--real", action="store_true", default=True
#     )
#     parser.set_defaults(real=False)
#     args = parser.parse_args(argv)
#     return args

# def save_png_image(img, img_name, png_path):
#     img = img.squeeze(0).permute(1,2,0).cpu().numpy()
#     file_name, ext = os.path.splitext(img_name)
#     new_img_name = file_name + '.png'
#     decompressed_path = os.path.join(png_path, new_img_name)
#     os.makedirs(os.path.dirname(decompressed_path), exist_ok=True)
#     print(decompressed_path)
#     plt.imsave(decompressed_path, img)
    
# def save_bin(string, size, img_name, bin_path):
#     file_name, ext = os.path.splitext(img_name)
#     bin_name = file_name + '.bin'
#     compress_path = os.path.join(bin_path,'bin/', bin_name)
#     os.makedirs(os.path.dirname(compress_path), exist_ok=True)
#     # print(type(string[0][0]))
#     with open(compress_path, 'wb') as f:
#         f.write(struct.pack(">H", size[0]))
#         f.write(struct.pack(">H", size[1]))
#         f.write(struct.pack(">I", len(string[0][0]))) 
#         f.write(string[0][0])
#         f.write(struct.pack(">I", len(string[1][0])))  
#         f.write(string[1][0])
    
# def calculate_padding(h, w, p=128):
#     new_h = (h + p - 1) // p * p
#     new_w = (w + p - 1) // p * p
#     padding_left = (new_w - w) // 2
#     padding_right = new_w - w - padding_left
#     padding_top = (new_h - h) // 2
#     padding_bottom = new_h - h - padding_top
#     padding_size = (new_h, new_w)
#     padding = (padding_left, padding_right, padding_top, padding_bottom)
#     return padding_size, padding

# def read_bin(bin_path):
#     with open(bin_path, 'rb') as f:
#         h = struct.unpack(">H", f.read(2))[0]
#         w = struct.unpack(">H", f.read(2))[0]
#         length_y = struct.unpack(">I", f.read(4))[0]
#         string_y = f.read(length_y)
#         length_z = struct.unpack(">I", f.read(4))[0]
#         string_z = f.read(length_z)
        
#     padding_size, padding = calculate_padding(h, w)
#     z_shape = [padding_size[0]//64, padding_size[1]//64]
    
#     string = [[string_y], [string_z]]
#     return string, z_shape, padding
    

# def main(argv):
#     torch.backends.cudnn.enabled = False
#     args = parse_args(argv)
#     p = 128
#     path_en = args.data_en
#     path_de = args.data_de
#     img_list = []
#     bin_list = []
#     for file in os.listdir(path_en):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(file)
#     for file in os.listdir(path_de):
#         if file[-3:] in ["bin"]:
#             bin_list.append(file)
        
#     if args.cuda:
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
#     device_en = "cuda"
#     device_de = "cuda"
    
#     base_path_en = args.save_path_en
#     base_path_de = args.save_path_de

#     net = DCAE()
#     net = net.to(device_en)
#     net.eval()
#     dictory = {}
#     if args.checkpoint:  
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location=device_en)
#         for k, v in checkpoint["state_dict"].items():
#             dictory[k.replace("module.", "")] = v
#         net.load_state_dict(dictory)
#     net.update()


#     for img_name in img_list:    
#         with torch.no_grad():
#             img_path = os.path.join(path_en, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device_en)
#             x = img.unsqueeze(0)
#             x_size = x.size()[-2:]
#             x_padded, padding = pad(x, p)
#             x_padded.to(device_en)
#             out_enc = net.compress(x_padded)
#             print(type(out_enc["strings"]), len(out_enc["strings"]))
#             print(out_enc["strings"])
#             # out_enc = out_enc.to(device_de)
#             save_bin(out_enc["strings"], x_size, img_name, base_path_en)
#     print("compression done")


#     net = DCAE()
#     net = net.to(device_de)
#     net.eval()
#     dictory = {}
#     if args.checkpoint:  
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location=device_de)
#         for k, v in checkpoint["state_dict"].items():
#             dictory[k.replace("module.", "")] = v
#         net.load_state_dict(dictory)
#     net.update()

#     for img_name in bin_list:  
#         with torch.no_grad():
#             bin_path = os.path.join(path_de, img_name)
#             string, shape, padding = read_bin(bin_path)
#             out_dec = net.decompress(string, shape)
#             out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
#             out_dec["x_hat"] = out_dec["x_hat"].clamp(0, 1)
#             to_save_img = out_dec["x_hat"]
#             save_png_image(to_save_img, img_name, base_path_de)


    # device_en = "cuda"
    # device_de = "cuda"
    
    # net_en = DCAE()
    # net_en = net_en.to(device_en)
    # net_en.eval()
    # dictory = {}
    # if args.checkpoint:  
    #     print("Loading", args.checkpoint)
    #     checkpoint = torch.load(args.checkpoint, map_location=device_en)
    #     for k, v in checkpoint["state_dict"].items():
    #         dictory[k.replace("module.", "")] = v
    #     net_en.load_state_dict(dictory)
    # net_en.update()
    # compressor = net_en.compress()



    # net_de = DCAE()
    # net_de = net_de.to(device_de)
    # net_de.eval()
    # dictory = {}
    # if args.checkpoint:  
    #     print("Loading", args.checkpoint)
    #     checkpoint = torch.load(args.checkpoint, map_location=device_de)
    #     for k, v in checkpoint["state_dict"].items():
    #         dictory[k.replace("module.", "")] = v
    #     net_de.load_state_dict(dictory)
    # net_de.update()
    # decompressor = net_de.decompress()


# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])



# import numpy as np
# import matplotlib.pyplot as plt
# import struct

# def read_bin(file_path:str):
#     with open(file_path, "rb") as f:
#         h = struct.unpack(">H", f.read(2))[0]
#         w = struct.unpack(">H", f.read(2))[0]
#         # Read string_y
#         length_y = struct.unpack(">I", f.read(4))[0]
#         string_y = f.read(length_y)
#         # Read string_z
#         length_z = struct.unpack(">I", f.read(4))[0]
#         string_z = f.read(length_z)
#     data_y = np.frombuffer(string_y, dtype=np.uint8)
#     data_z = np.frombuffer(string_z, dtype=np.uint8)

#     return data_y, data_z


# def plot_all(data_y_1, data_z_1, data_y_2, data_z_2, data_y_3, data_z_3, data_y_4, data_z_4)->None:
#     # Plot histograms
#     plt.figure(figsize=(12, 6))
#     bins = np.arange(257)

#     plt.subplot(1, 2, 1)
#     plt.hist(data_y_1, bins=bins, range=(0, 255), alpha=0.5, label='GPU1', color='blue', density=True)
#     plt.hist(data_y_2, bins=bins, range=(0, 255), alpha=0.5, label='GPU2', color='green', density=True)
#     plt.hist(data_y_3, bins=bins, range=(0, 255), alpha=0.5, label='CPU1', color='red', density=True)
#     plt.hist(data_y_4, bins=bins, range=(0, 255), alpha=0.5, label='CPU2', color='orange', density=True)


#     plt.title('Histogram of string_y')
#     plt.xlabel('Byte Value')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.tight_layout()

#     plt.subplot(1, 2, 2)
#     plt.hist(data_z_1, bins=bins, range=(0, 255), alpha=0.5, label='GPU1', color='blue', density=True)
#     plt.hist(data_z_2, bins=bins, range=(0, 255), alpha=0.5, label='GPU2', color='green', density=True)
#     plt.hist(data_z_3, bins=bins, range=(0, 255), alpha=0.5, label='CPU1', color='red', density=True)
#     plt.hist(data_z_4, bins=bins, range=(0, 255), alpha=0.5, label='CPU2', color='orange', density=True)


#     plt.title('Histogram of string_z')
#     plt.xlabel('Byte Value')
#     plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.legend()

#     plt.savefig("histogram.png")  # Save the current figure as a PNG file
#     plt.show()


# def compare(a, b):
#     if np.array_equal(a, b):
#         return True
#     else:
#         return False


# file_1 = '/home/xie/DCAE/output/binary/bin_gpu/image_1743411007.664348877.bin'
# data_y_1, data_z_1 = read_bin(file_1)

# file_2 = '/home/xie/DCAE/output/binary/bin_gpu2/image_1743411007.664348877.bin'
# data_y_2, data_z_2 = read_bin(file_2)

# file_3 = '/home/xie/DCAE/output/binary/bin_cpu/image_1743411007.664348877.bin'
# data_y_3, data_z_3 = read_bin(file_3)

# file_4 = '/home/xie/DCAE/output/binary/bin_cpu2/image_1743411007.664348877.bin'
# data_y_4, data_z_4 = read_bin(file_4)

# plot_all(data_y_1, data_z_1, data_y_2, data_z_2, data_y_3, data_z_3, data_y_4, data_z_4)

# result_y_1_2 = compare(data_y_1, data_y_2)
# result_y_3_4 = compare(data_y_3, data_y_4) 
# result_y_2_3 = compare(data_y_2, data_y_3)

# result_z_1_2 = compare(data_z_1, data_z_2)
# result_z_3_4 = compare(data_z_3, data_z_4)
# result_z_2_3 = compare(data_z_2, data_z_3)

# print(f'y: 1=2: {result_y_1_2}, 3=4: {result_y_3_4}, 2=3: {result_y_2_3}')
# print(f'z: 1=2: {result_z_1_2}, 3=4: {result_z_3_4}, 2=3: {result_z_2_3}')

##
# import pandas as pd

# # Replace these paths with your actual CSV file paths
# csv_file1 = '/home/xie/DCAE/output/debug/debug/cpu_2/image_1743411007.664348877_debug_z.csv'
# csv_file2 = '/home/xie/DCAE/output/debug/debug/cpu_1/image_1743411007.664348877_debug_z.csv'

# df1 = pd.read_csv(csv_file1)
# df2 = pd.read_csv(csv_file2)

# if df1.equals(df2):
#     print("The CSV files are the same.")
# else:
#     print("The CSV files are different.")


# import struct


# def read_bin(file_path):
#     with open(file_path, 'rb') as f:
#         # Read sizes (if needed)
#         size0 = struct.unpack('>H', f.read(2))[0]
#         size1 = struct.unpack('>H', f.read(2))[0]

#         # Read the first string
#         len_str0 = struct.unpack('>I', f.read(4))[0]
#         string_y = f.read(len_str0)

#         # Read the second string
#         len_str1 = struct.unpack('>I', f.read(4))[0]
#         string_z = f.read(len_str1)

#     return string_y, string_z

# def compare_strings(string1, string2):
#     if string1 == string2:
#         print("Equal")
#     else:
#         print("Not equal")


# file_1 = "/home/xie/DCAE/output/debug/bin_cpu_zstrings/image_1743411007.664348877.bin"
# file_2 = "/home/xie/DCAE/output/debug/bin_gpu/image_1743411007.664348877.bin"

# string_y_1, string_z_1 = read_bin(file_1)
# string_y_2, string_z_2 = read_bin(file_2)    
# compare_strings(string_y_1, string_y_2)
# compare_strings(string_z_1, string_z_2)

# import inspect
# from compressai.ans import BufferedRansEncoder
# from compressai.entropy_models import GaussianConditional

# gauss = GaussianConditional(None)
# cdf = gauss.quantized_cdf.tolist()

# # This will print the source code if CompressAI is installed from source
# print(cdf)

# from compressai.ans import BufferedRansEncoder

# help(BufferedRansEncoder)

# import torch

# # Load the checkpoint
# checkpoint = torch.load('0.0018checkpoint_best.pth.tar')
# state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# # Print parameter dtypes
# for param_name, weights in state_dict.items():
#     print(param_name, weights.dtype)
#     break  # remove or adjust this break to view all parameters

# check model structure
from models import DCAE
from torchsummary import summary
import torch

device_1 = "cuda"
device_2 = "cpu"
checkpoint_path = "./60.5checkpoint_best.pth.tar"

# torch.set_default_dtype(torch.float64)

net_1 = DCAE()
net_1 = net_1.to(device_1)
net_1.eval()
dictory_1 = {}
checkpoint_1 = torch.load(checkpoint_path, map_location=device_1)
trainable_blocks = []
for k, v in checkpoint_1["state_dict"].items():
    k_parts = k.split(".")
    k_name = k_parts[0] + "_" + k_parts[1]
    if k_name not in trainable_blocks:
        trainable_blocks.append(k_name)
    dictory_1[k.replace("module.", "")] = v
net_1.load_state_dict(dictory_1)


summary(net_1.g_a, input_size=(3, 256,256), device="cuda")
print(net_1.g_a)
# print("trainable blocks: ", trainable_blocks)
# print(net_1.dt)



# net_2 = DCAE()
# net_2 = net_2.to(device_2)
# net_2.eval()
# dictory_2 = {}
# checkpoint_2 = torch.load(checkpoint_path, map_location=device_2)
# trainable_blocks = []
# for k, v in checkpoint_2["state_dict"].items():
#     k_parts = k.split(".")
#     k_name = k_parts[0] + "_" + k_parts[1]
#     if k_name not in trainable_blocks:
#         trainable_blocks.append(k_name)
#     dictory_2[k.replace("module.", "")] = v
# net_2.load_state_dict(dictory_2)


# checkpoint = torch.load(checkpoint_1, map_location=device_1)
# trainable_blocks = []
# for k, v in checkpoint["state_dict"].items():
#     k_parts = k.split(".")
#     k_name = k_parts[0] + "_" + k_parts[1]
#     if k_name not in trainable_blocks:
#         trainable_blocks.append(k_name)
#     dictory_1[k.replace("module.", "")] = v
# net.load_state_dict(dictory)



# # compare variables under different conditions
# import torch

# def compare(tensor1, tensor2):
#     if isinstance(tensor1, bytes):
#         tensor1 = list(tensor1)
#         tensor1 = torch.tensor(tensor1, dtype=torch.float32)
#     if isinstance(tensor2, bytes):
#         tensor2 = list(tensor2)
#         tensor2 = torch.tensor(tensor2, dtype=torch.float32)
#     if tensor1.device != tensor2.device:
#         tensor1=tensor1.to("cpu")
#         tensor2=tensor2.to("cpu")
#     are_identical = torch.equal(tensor1, tensor2)
#     if are_identical:
#         print("The tensors are identical. \n")
#     else:
#         print("The tensors are not identical.")
#         # Calculate absolute difference
#         absolute_difference = torch.abs(tensor1 - tensor2)/tensor2
#         print("Relative difference:", torch.mean(torch.abs(absolute_difference)))
#         # Calculate mean squared error
#         mse = torch.mean((tensor1 - tensor2) ** 2)
#         print("Mean squared error:", mse, "\n")

# z_cuda_32 = torch.load("./output/debug/z_cuda_32.pt")
# z_cpu_32 = torch.load("./output/debug/z_cpu_32.pt") # diff

# z_hat_cuda_32 = torch.load("./output/debug/z_hat_cuda_32.pt")
# z_hat_cpu_32 = torch.load("./output/debug/z_hat_cpu_32.pt") # same

# y_string_cpu_32 = torch.load("./output/debug/y_string_cpu_32.pt")
# y_string_cuda_32 = torch.load("./output/debug/y_string_cuda_32.pt") # same

# z_strings_cpu_32 = torch.load("./output/debug/z_strings_cpu_32.pt") # diff
# z_strings_cuda_32 = torch.load("./output/debug/z_strings_cuda_32.pt")

# y_hat_cpu_32 = torch.load("./output/debug/y_hat_cpu_32.pt") # diff
# y_hat_cuda_32 = torch.load("./output/debug/y_hat_cuda_32.pt")

# rv_cpu_32 = torch.load("./output/debug/rv_cpu_32.pt")
# rv_cuda_32 = torch.load("./output/debug/rv_cuda_32.pt")

# compare(z_cuda_32, z_cpu_32)

# compare(z_hat_cuda_32, z_hat_cpu_32)

# compare(y_string_cpu_32, y_string_cuda_32)

# compare(z_strings_cpu_32, z_strings_cuda_32)

# compare(y_hat_cpu_32, y_hat_cuda_32)

# compare(rv_cpu_32, rv_cuda_32)


## check the last epoch of the checkpoints
# import torch
# from models import DCAE_3
# import torch.optim as optim

# device = "cuda"
# print("Loading", "./60.5checkpoint_best.pth.tar")
# checkpoint = torch.load("./60.5checkpoint_best.pth.tar", map_location=device)
# print(f'the last epoch of this checkpoint is: {checkpoint["epoch"]}')
