# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models import (
#     DCAE    
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

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Example testing script.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default="./checkpoints/60.5checkpoint_best_30k_wi_dummy.pth.tar")
#     parser.add_argument("--data", type=str, help="Path to dataset", default='../datasets/dummy/test')
#     parser.add_argument("--save_path", default=None, type=str, help="Path to save")
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
        
#     net = DCAE()
#     net = net.to(device)
#     net.eval()
    
#     count = 0
#     PSNR = 0
#     Bit_rate = 0
#     MS_SSIM = 0
#     total_time = 0
#     encode_time = 0
#     decode_time = 0
#     ave_flops = 0
#     encoder_time = 0
#     dictory = {}
    
#     if args.checkpoint:  
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location=device)
#         for k, v in checkpoint["state_dict"].items():
#             dictory[k.replace("module.", "")] = v
#         net.load_state_dict(dictory)
        
#     if args.real:
#         net.update()
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
#             x = img.unsqueeze(0)
#             x_padded, padding = pad(x, p)
#             x_padded.to(device)

#             count += 1
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_enc = net.compress(x_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 encode_time += (e - s)
#                 total_time += (e - s)
#                 # flops, params = profile(net, inputs=(x_padded,))

#                 # flops, params = profile(net, inputs=(out_enc["strings"], out_enc["shape"]))
#                 # print('flops:{}'.format(flops))
#                 # print('params:{}'.format(params))
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 decode_time += (e - s)
#                 total_time += (e - s)

#                 out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
#                 num_pixels = x.size(0) * x.size(2) * x.size(3)
#                 # print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
#                 # print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
#                 # print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
#                 print(f'encoding time for {img_name}: {(e - s)*1000:.3f} ms, decoding time for {img_name}: {(e - s)*1000:.3f} ms')
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 PSNR += compute_psnr(x, out_dec["x_hat"])
#                 MS_SSIM += compute_msssim(x, out_dec["x_hat"])
                
#     else:
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = Image.open(img_path).convert('RGB')
#             x = transforms.ToTensor()(img).unsqueeze(0).to(device)
#             x_padded, padding = pad(x, p)

#             # flops, params = profile(net, inputs=(x_padded, ))
#             # ave_flops += flops
#             # print('flops:{}'.format(flops))
#             # print('params:{}'.format(params))

#             count += 1
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_net = net.forward(x_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 total_time += (e - s)
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop(out_net["x_hat"], padding)
#                 print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
#                 print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB')
#                 print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
#                 PSNR += compute_psnr(x, out_net["x_hat"])
#                 MS_SSIM += compute_msssim(x, out_net["x_hat"])
#                 Bit_rate += compute_bpp(out_net)
#                 if args.save_path is not None:
#                     save_metrics(os.path.join(args.save_path, f"metrics_{img_name.split('.')[0]}.txt"), compute_psnr(x, out_net["x_hat"]), compute_bpp(out_net), compute_msssim(x, out_net["x_hat"]))
#                     save_image(out_net["x_hat"], os.path.join(args.save_path, f"decoded_{img_name}"))
#     PSNR = PSNR / count
#     MS_SSIM = MS_SSIM / count
#     Bit_rate = Bit_rate / count
#     total_time = total_time / count
#     encode_time = encode_time / count
#     decode_time = decode_time /count
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
    


# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models import (
#     DCAE    
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
# from thop import profile
# warnings.filterwarnings("ignore")
# torch.set_num_threads(10)
# torch.set_default_dtype(torch.float64)
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

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Example testing script.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, default="./checkpoints/60.5checkpoint_best_30k_wi_dummy.pth.tar", help="Path to a checkpoint")
#     parser.add_argument("--data", type=str, default="../datasets/dummy/valid", help="Path to dataset")
#     parser.add_argument("--save_path", default=None, type=str, help="Path to save")
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
    
#     print("device been used:", device)

#     net = DCAE()
#     net = net.to(device)
#     net.eval()
    
#     count = 0
#     PSNR = 0
#     Bit_rate = 0
#     MS_SSIM = 0
#     total_time = 0
#     encode_time = 0
#     decode_time = 0
#     ave_flops = 0
#     encoder_time = 0
#     dictory = {}
    
#     if args.checkpoint:  
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location=device)
#         for k, v in checkpoint["state_dict"].items():
#             dictory[k.replace("module.", "")] = v
#         net.load_state_dict(dictory)
#         # print details of the model
#         # print("Model loaded successfully: ", net)
#         # for k, v in net.named_parameters():
#         #     print(f'{k}: {v.size()}')
        
#     if args.real:
#         net.update()
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
#             x = img.unsqueeze(0)
#             x_padded, padding = pad(x, p)
#             x_padded.to(device)

#             count += 1
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_enc = net.compress(x_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 encode_time += (e - s)
#                 total_time += (e - s)
#                 # flops, params = profile(net, inputs=(x_padded,))

#                 # flops, params = profile(net, inputs=(out_enc["strings"], out_enc["shape"]))
#                 # print('flops:{}'.format(flops))
#                 # print('params:{}'.format(params))
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 decode_time += (e - s)
#                 total_time += (e - s)

#                 out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
#                 num_pixels = x.size(0) * x.size(2) * x.size(3)
#                 # print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
#                 # print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
#                 # print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 PSNR += compute_psnr(x, out_dec["x_hat"])
#                 MS_SSIM += compute_msssim(x, out_dec["x_hat"])
                
#     else:
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = Image.open(img_path).convert('RGB')
#             x = transforms.ToTensor()(img).unsqueeze(0).to(device)
#             x_padded, padding = pad(x, p)

#             flops, params = profile(net, inputs=(x_padded, ))
#             ave_flops += flops
#             print('flops:{}'.format(flops))
#             print('params:{}'.format(params))

#             count += 1
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_net = net.forward(x_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 total_time += (e - s)
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop(out_net["x_hat"], padding)
#                 print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
#                 print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB')
#                 print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
#                 PSNR += compute_psnr(x, out_net["x_hat"])
#                 MS_SSIM += compute_msssim(x, out_net["x_hat"])
#                 Bit_rate += compute_bpp(out_net)
#                 if args.save_path is not None:
#                     save_metrics(os.path.join(args.save_path, f"metrics_{img_name.split('.')[0]}.txt"), compute_psnr(x, out_net["x_hat"]), compute_bpp(out_net), compute_msssim(x, out_net["x_hat"]))
#                     save_image(out_net["x_hat"], os.path.join(args.save_path, f"decoded_{img_name}"))
#     PSNR = PSNR / count
#     MS_SSIM = MS_SSIM / count
#     Bit_rate = Bit_rate / count
#     total_time = total_time / count
#     encode_time = encode_time / count
#     decode_time = decode_time /count
#     ave_flops = ave_flops / count
#     encoder_time = encoder_time / count
#     print(f'average_encoder_time: {encoder_time:.3f} s')
#     print(f'average_PSNR: {PSNR:.2f} dB')
#     print(f'average_MS-SSIM: {MS_SSIM:.4f}')
#     print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
#     print(f'average_time: {total_time:.3f} s')
#     print(f'average_encode_time: {encode_time:.6f} s')
#     print(f'average_decode_time: {decode_time:.6f} s')
#     print(f'average_flops: {ave_flops:.3f}')

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])
    


# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models import (
#     DCAE    
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

# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Example testing script.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, default="./checkpoints/60.5checkpoint_best_30k_wi_dummy.pth.tar", help="Path to a checkpoint")
#     parser.add_argument("--data", type=str, default="../datasets/dummy/valid", help="Path to dataset")
#     parser.add_argument("--save_path", default=None, type=str, help="Path to save")
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
        
#     net = DCAE()
#     net = net.to(device)
#     net.eval()
    
#     count = 0
#     PSNR = 0
#     Bit_rate = 0
#     MS_SSIM = 0
#     total_time = 0
#     encode_time = 0
#     decode_time = 0
#     ave_flops = 0
#     encoder_time = 0
#     dictory = {}
    
#     if args.checkpoint:  
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location=device)
#         for k, v in checkpoint["state_dict"].items():
#             dictory[k.replace("module.", "")] = v
#         net.load_state_dict(dictory)
        
#     if args.real:
#         net.update()
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
#             x = img.unsqueeze(0)
#             x_padded, padding = pad(x, p)
#             x_padded.to(device)

#             count += 1
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_enc = net.compress(x_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 encode_time += (e - s)
#                 total_time += (e - s)
#                 print(f'Encoding time for {img_name}: {(e - s)*1000:.3f} ms')
#                 # flops, params = profile(net, inputs=(x_padded,))

#                 # flops, params = profile(net, inputs=(out_enc["strings"], out_enc["shape"]))
#                 # print('flops:{}'.format(flops))
#                 # print('params:{}'.format(params))
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 decode_time += (e - s)
#                 total_time += (e - s)
#                 print(f'Decoding time for {img_name}: {(e - s)*1000:.3f} ms')

#                 out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
#                 num_pixels = x.size(0) * x.size(2) * x.size(3)
#                 print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
#                 print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
#                 print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#                 PSNR += compute_psnr(x, out_dec["x_hat"])
#                 MS_SSIM += compute_msssim(x, out_dec["x_hat"])
                
#     else:
#         for img_name in img_list:
#             img_path = os.path.join(path, img_name)
#             img = Image.open(img_path).convert('RGB')
#             x = transforms.ToTensor()(img).unsqueeze(0).to(device)
#             x_padded, padding = pad(x, p)

#             # flops, params = profile(net, inputs=(x_padded, ))
#             # ave_flops += flops
#             # print('flops:{}'.format(flops))
#             # print('params:{}'.format(params))

#             count += 1
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_net = net.forward(x_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 total_time += (e - s)
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop(out_net["x_hat"], padding)
#                 print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
#                 print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB')
#                 print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
#                 PSNR += compute_psnr(x, out_net["x_hat"])
#                 MS_SSIM += compute_msssim(x, out_net["x_hat"])
#                 Bit_rate += compute_bpp(out_net)
#                 if args.save_path is not None:
#                     save_metrics(os.path.join(args.save_path, f"metrics_{img_name.split('.')[0]}.txt"), compute_psnr(x, out_net["x_hat"]), compute_bpp(out_net), compute_msssim(x, out_net["x_hat"]))
#                     save_image(out_net["x_hat"], os.path.join(args.save_path, f"decoded_{img_name}"))
#     PSNR = PSNR / count
#     MS_SSIM = MS_SSIM / count
#     Bit_rate = Bit_rate / count
#     total_time = total_time / count
#     encode_time = encode_time / count
#     decode_time = decode_time /count
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


import torch
import torch.nn.functional as F
from torchvision import transforms
from models import (
    DCAE    
)
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

def save_metrics(filename, psnr, bitrate, msssim):
    with open(filename, 'w') as f:
        f.write(f'PSNR: {psnr:.2f}dB\n')
        f.write(f'Bitrate: {bitrate:.3f}bpp\n')
        f.write(f'MS-SSIM: {msssim:.4f}\n')

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

def compute_bpp_batch(out_net):
    # Compute BPP for each image in the batch
    batch_size = out_net['x_hat'].size(0)
    height, width = out_net['x_hat'].size(2), out_net['x_hat'].size(3)
    num_pixels = height * width
    
    bpp_values = []
    for b in range(batch_size):
        total_bits = 0
        for key, likelihoods in out_net['likelihoods'].items():
            total_bits += torch.log(likelihoods[b]).sum() / (-math.log(2))
        bpp_values.append(total_bits / num_pixels)
    
    return torch.tensor(bpp_values)

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
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default="./checkpoints/60.5checkpoint_best_30k_wi_dummy.pth.tar")
    parser.add_argument("--data", type=str, help="Path to dataset", default='../datasets/dummy/valid')
    parser.add_argument("--save_path", default=None, type=str, help="Path to save")
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args

def main(argv):
    torch.backends.cudnn.enabled = False
    args = parse_args(argv)
    p = 128
    path = args.data
    batch_size = 1
    
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(os.path.join(path, file))
    
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
        
    net = DCAE()
    net = net.to(device)
    net.eval()
    
    count = 0
    total_PSNR = 0
    total_Bit_rate = 0
    total_MS_SSIM = 0
    total_time = 0
    encode_time = 0
    decode_time = 0
    ave_flops = 0
    encoder_time = 0
    dictory = {}
    
    if args.checkpoint:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
    
    # Process images in batches
    num_batches = (len(img_list) + batch_size - 1) // batch_size
    
    if args.real:
        net.update()
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
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_enc = net.compress(x_batch_padded)
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                batch_encode_time = (e - s)
                encode_time += batch_encode_time
                total_time += batch_encode_time
                
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                batch_decode_time = (e - s)
                decode_time += batch_decode_time
                total_time += batch_decode_time
                
                out_dec["x_hat"] = crop_batch(out_dec["x_hat"], padding)
                
                # Compute metrics for each image in the batch
                for i in range(current_batch_size):
                    x_single = x_batch[i:i+1]
                    x_hat_single = out_dec["x_hat"][i:i+1]
                    num_pixels = x_single.size(2) * x_single.size(3)
                    
                    # Calculate bitrate for single image (simplified)
                    bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / (num_pixels * current_batch_size)
                    
                    total_Bit_rate += bit_rate
                    total_PSNR += compute_psnr(x_single, x_hat_single).item()
                    total_MS_SSIM += compute_msssim(x_single, x_hat_single).item()
                
                print(f'Batch {batch_idx + 1} - Encoding time: {batch_encode_time*1000:.3f} ms, Decoding time: {batch_decode_time*1000:.3f} ms')
                
    else:
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
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_net = net.forward(x_batch_padded)
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                batch_time = (e - s)
                total_time += batch_time
                
                out_net['x_hat'].clamp_(0, 1)
                out_net["x_hat"] = crop_batch(out_net["x_hat"], padding)
                
                # Compute metrics for batch
                psnr_batch = compute_psnr(x_batch, out_net["x_hat"])
                msssim_batch = compute_msssim(x_batch, out_net["x_hat"])
                bpp_batch = compute_bpp_batch(out_net)
                
                total_PSNR += psnr_batch.sum().item()
                total_MS_SSIM += msssim_batch.sum().item()
                total_Bit_rate += bpp_batch.sum().item()
                
                # Save individual results if requested
                if args.save_path is not None:
                    for i in range(current_batch_size):
                        img_name = os.path.basename(batch_img_paths[i])
                        save_metrics(
                            os.path.join(args.save_path, f"metrics_{img_name.split('.')[0]}.txt"), 
                            psnr_batch[i].item(), 
                            bpp_batch[i].item(), 
                            msssim_batch[i].item()
                        )
                        save_image(out_net["x_hat"][i:i+1], os.path.join(args.save_path, f"decoded_{img_name}"))
                
                print(f'Batch {batch_idx + 1} - Processing time: {batch_time*1000:.3f} ms')
    
    # Calculate averages
    avg_PSNR = total_PSNR / count
    avg_MS_SSIM = total_MS_SSIM / count
    avg_Bit_rate = total_Bit_rate / count
    avg_total_time = (total_time / count) * 1000  # Convert to ms
    avg_encode_time = (encode_time / count) * 1000
    avg_decode_time = (decode_time / count) * 1000
    
    print(f'Total images processed: {count}')
    print(f'Average PSNR: {avg_PSNR:.2f} dB')
    print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
    print(f'Average Bit-rate: {avg_Bit_rate:.3f} bpp')
    print(f'Average total time per image: {avg_total_time:.3f} ms')
    print(f'Average encode time per image: {avg_encode_time:.3f} ms')
    print(f'Average decode time per image: {avg_decode_time:.3f} ms')

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])


