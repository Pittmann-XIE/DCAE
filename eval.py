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

# ## original code before modification
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
#     mse = torch.mean((a - b)**2, dim=[1,2,3])  # Compute per image in batch
#     return -10 * torch.log10(mse)

# def compute_msssim(a, b):
#     # Process each image in the batch
#     msssim_values = []
#     for i in range(a.size(0)):
#         msssim_val = ms_ssim(a[i:i+1], b[i:i+1], data_range=1.).item()
#         msssim_values.append(-10 * math.log10(1 - msssim_val))
#     return torch.tensor(msssim_values)

# def compute_bpp_batch(out_net):
#     # Compute BPP for each image in the batch
#     batch_size = out_net['x_hat'].size(0)
#     height, width = out_net['x_hat'].size(2), out_net['x_hat'].size(3)
#     num_pixels = height * width
    
#     bpp_values = []
#     for b in range(batch_size):
#         total_bits = 0
#         for key, likelihoods in out_net['likelihoods'].items():
#             total_bits += torch.log(likelihoods[b]).sum() / (-math.log(2))
#         bpp_values.append(total_bits / num_pixels)
    
#     return torch.tensor(bpp_values)

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
#     parser = argparse.ArgumentParser(description="Example testing script.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default="./60.5checkpoint_best.pth.tar")
#     parser.add_argument("--data", type=str, help="Path to dataset", default='../datasets/dummy/valid')
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
#     batch_size = 1
    
#     img_list = []
#     for file in os.listdir(path):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(os.path.join(path, file))
    
#     if args.cuda:
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
        
#     net = DCAE()
#     net = net.to(device)
#     net.eval()
    
#     count = 0
#     total_PSNR = 0
#     total_Bit_rate = 0
#     total_MS_SSIM = 0
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
    
#     # Process images in batches
#     num_batches = (len(img_list) + batch_size - 1) // batch_size
    
#     if args.real:
#         net.update()
#         for batch_idx in range(num_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = min((batch_idx + 1) * batch_size, len(img_list))
#             batch_img_paths = img_list[start_idx:end_idx]
#             current_batch_size = len(batch_img_paths)
            
#             print(f"Processing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
            
#             # Load batch of images
#             x_batch = load_images_batch(batch_img_paths, device)
#             x_batch_padded, padding = pad_batch(x_batch, p)
            
#             count += current_batch_size
            
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_enc = net.compress(x_batch_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 batch_encode_time = (e - s)
#                 encode_time += batch_encode_time
#                 total_time += batch_encode_time
                
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 batch_decode_time = (e - s)
#                 decode_time += batch_decode_time
#                 total_time += batch_decode_time
                
#                 out_dec["x_hat"] = crop_batch(out_dec["x_hat"], padding)
                
#                 # Compute metrics for each image in the batch
#                 for i in range(current_batch_size):
#                     x_single = x_batch[i:i+1]
#                     x_hat_single = out_dec["x_hat"][i:i+1]
#                     num_pixels = x_single.size(2) * x_single.size(3)
                    
#                     # Calculate bitrate for single image (simplified)
#                     bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / (num_pixels * current_batch_size)
                    
#                     total_Bit_rate += bit_rate
#                     total_PSNR += compute_psnr(x_single, x_hat_single).item()
#                     total_MS_SSIM += compute_msssim(x_single, x_hat_single).item()
                
#                 print(f'Batch {batch_idx + 1} - Encoding time: {batch_encode_time*1000:.3f} ms, Decoding time: {batch_decode_time*1000:.3f} ms')
                
#     else:
#         for batch_idx in range(num_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = min((batch_idx + 1) * batch_size, len(img_list))
#             batch_img_paths = img_list[start_idx:end_idx]
#             current_batch_size = len(batch_img_paths)
            
#             print(f"Processing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
            
#             # Load batch of images
#             x_batch = load_images_batch(batch_img_paths, device)
#             x_batch_padded, padding = pad_batch(x_batch, p)
            
#             count += current_batch_size
            
#             with torch.no_grad():
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 s = time.time()
#                 out_net = net.forward(x_batch_padded)
#                 if args.cuda:
#                     torch.cuda.synchronize()
#                 e = time.time()
#                 batch_time = (e - s)
#                 total_time += batch_time
                
#                 out_net['x_hat'].clamp_(0, 1)
#                 out_net["x_hat"] = crop_batch(out_net["x_hat"], padding)
                
#                 # Compute metrics for batch
#                 psnr_batch = compute_psnr(x_batch, out_net["x_hat"])
#                 msssim_batch = compute_msssim(x_batch, out_net["x_hat"])
#                 bpp_batch = compute_bpp_batch(out_net)
                
#                 total_PSNR += psnr_batch.sum().item()
#                 total_MS_SSIM += msssim_batch.sum().item()
#                 total_Bit_rate += bpp_batch.sum().item()
                
#                 # Save individual results if requested
#                 if args.save_path is not None:
#                     for i in range(current_batch_size):
#                         img_name = os.path.basename(batch_img_paths[i])
#                         save_metrics(
#                             os.path.join(args.save_path, f"metrics_{img_name.split('.')[0]}.txt"), 
#                             psnr_batch[i].item(), 
#                             bpp_batch[i].item(), 
#                             msssim_batch[i].item()
#                         )
#                         save_image(out_net["x_hat"][i:i+1], os.path.join(args.save_path, f"decoded_{img_name}"))
                
#                 print(f'Batch {batch_idx + 1} - Processing time: {batch_time*1000:.3f} ms')
    
#     # Calculate averages
#     avg_PSNR = total_PSNR / count
#     avg_MS_SSIM = total_MS_SSIM / count
#     avg_Bit_rate = total_Bit_rate / count
#     avg_total_time = (total_time / count) * 1000  # Convert to ms
#     avg_encode_time = (encode_time / count) * 1000
#     avg_decode_time = (decode_time / count) * 1000
    
#     print(f'Total images processed: {count}')
#     print(f'Average PSNR: {avg_PSNR:.2f} dB')
#     print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
#     print(f'Average Bit-rate: {avg_Bit_rate:.3f} bpp')
#     print(f'Average total time per image: {avg_total_time:.3f} ms')
#     print(f'Average encode time per image: {avg_encode_time:.3f} ms')
#     print(f'Average decode time per image: {avg_decode_time:.3f} ms')

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
# import pickle
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

# def save_compressed_data(compressed_data, filename):
#     """Save compressed data to a file"""
#     with open(filename, 'wb') as f:
#         pickle.dump(compressed_data, f)
#     print(f"Compressed data saved to: {filename}")

# def load_compressed_data(filename):
#     """Load compressed data from a file"""
#     with open(filename, 'rb') as f:
#         compressed_data = pickle.load(f)
#     print(f"Compressed data loaded from: {filename}")
#     return compressed_data

# def get_size_in_bytes(obj):
#     """Calculate size of an object in bytes"""
#     if isinstance(obj, torch.Tensor):
#         return obj.numel() * obj.element_size()
#     elif isinstance(obj, (list, tuple)):
#         total_size = 0
#         for item in obj:
#             total_size += get_size_in_bytes(item)
#         return total_size
#     elif isinstance(obj, str):
#         return len(obj.encode('utf-8'))
#     elif isinstance(obj, bytes):
#         return len(obj)
#     elif isinstance(obj, dict):
#         total_size = 0
#         for key, value in obj.items():
#             total_size += get_size_in_bytes(key) + get_size_in_bytes(value)
#         return total_size
#     else:
#         return sys.getsizeof(obj)

# def analyze_data_size(device, args):
#     """Analyze and display sizes of compressed data and original images"""
#     print("=== SIZE ANALYSIS MODE ===")
    
#     if not os.path.exists(args.compressed_path):
#         print(f"Error: Compressed data path {args.compressed_path} does not exist!")
#         return
    
#     if not os.path.exists(args.data):
#         print(f"Error: Original images path {args.data} does not exist!")
#         return
    
#     # Find compressed files
#     compressed_files = [f for f in os.listdir(args.compressed_path) if f.endswith('_compressed.pkl')]
    
#     if not compressed_files:
#         print(f"Error: No compressed files found in {args.compressed_path}")
#         return
    
#     # Get original image files
#     img_list = []
#     for file in os.listdir(args.data):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(os.path.join(args.data, file))
    
#     print(f"\nFound {len(compressed_files)} compressed files and {len(img_list)} original images")
#     print("-" * 80)
    
#     total_strings_size = 0
#     total_shape_size = 0
#     total_original_size = 0
#     total_compressed_file_size = 0
    
#     analysis_results = []
    
#     for i, compressed_file in enumerate(compressed_files):
#         img_name = compressed_file.replace('_compressed.pkl', '')
#         compressed_path = os.path.join(args.compressed_path, compressed_file)
        
#         # Find corresponding original image
#         original_img_path = None
#         for img_path in img_list:
#             if img_name in os.path.basename(img_path):
#                 original_img_path = img_path
#                 break
        
#         if original_img_path is None:
#             print(f"Warning: No original image found for {img_name}")
#             continue
        
#         # Load compressed data
#         compressed_data = load_compressed_data(compressed_path)
        
#         # Load original image
#         original_img = transforms.ToTensor()(Image.open(original_img_path).convert('RGB'))
        
#         # Calculate sizes
#         strings_size = get_size_in_bytes(compressed_data["strings"])
#         shape_size = get_size_in_bytes(compressed_data["shape"])
#         original_size = get_size_in_bytes(original_img)
#         compressed_file_size = os.path.getsize(compressed_path)
        
#         # Get data types and shapes
#         strings_info = {
#             'type': type(compressed_data["strings"]).__name__,
#             'length': len(compressed_data["strings"]),
#             'structure': f"List of {len(compressed_data['strings'])} elements"
#         }
        
#         shape_info = {
#             'type': type(compressed_data["shape"]).__name__,
#             'value': compressed_data["shape"],
#             'structure': f"Shape: {compressed_data['shape']}"
#         }
        
#         original_info = {
#             'type': type(original_img).__name__,
#             'shape': list(original_img.shape),
#             'dtype': str(original_img.dtype)
#         }
        
#         # Store results
#         result = {
#             'img_name': img_name,
#             'strings_size': strings_size,
#             'shape_size': shape_size,
#             'original_size': original_size,
#             'compressed_file_size': compressed_file_size,
#             'strings_info': strings_info,
#             'shape_info': shape_info,
#             'original_info': original_info
#         }
#         analysis_results.append(result)
        
#         # Update totals
#         total_strings_size += strings_size
#         total_shape_size += shape_size
#         total_original_size += original_size
#         total_compressed_file_size += compressed_file_size
        
#         # Print individual results
#         print(f"\n--- Image {i+1}: {img_name} ---")
#         print(f"Original Image:")
#         print(f"  • Type: {original_info['type']}")
#         print(f"  • Shape: {original_info['shape']}")
#         print(f"  • Data Type: {original_info['dtype']}")
#         print(f"  • Size: {original_size:,} bytes ({original_size/1024:.2f} KB)")
        
#         print(f"\nCompressed Data:")
#         print(f"  • Strings:")
#         print(f"    - Type: {strings_info['type']}")
#         print(f"    - Structure: {strings_info['structure']}")
#         print(f"    - Size: {strings_size:,} bytes ({strings_size/1024:.2f} KB)")
        
#         print(f"  • Shape:")
#         print(f"    - Type: {shape_info['type']}")
#         print(f"    - Value: {shape_info['value']}")
#         print(f"    - Size: {shape_size:,} bytes")
        
#         print(f"  • Total compressed file size: {compressed_file_size:,} bytes ({compressed_file_size/1024:.2f} KB)")
        
#         # Compression ratio
#         compression_ratio = original_size / compressed_file_size if compressed_file_size > 0 else float('inf')
#         print(f"  • Compression ratio: {compression_ratio:.2f}:1")
        
#         print("-" * 40)
    
#     # Print summary
#     print(f"\n=== SUMMARY ===")
#     print(f"Total files analyzed: {len(analysis_results)}")
#     print(f"\nTotal sizes:")
#     print(f"  • Original images: {total_original_size:,} bytes ({total_original_size/1024/1024:.2f} MB)")
#     print(f"  • Compressed strings: {total_strings_size:,} bytes ({total_strings_size/1024:.2f} KB)")
#     print(f"  • Compressed shapes: {total_shape_size:,} bytes")
#     print(f"  • Total compressed files: {total_compressed_file_size:,} bytes ({total_compressed_file_size/1024:.2f} KB)")
    
#     # Average sizes
#     num_files = len(analysis_results)
#     if num_files > 0:
#         print(f"\nAverage sizes per image:")
#         print(f"  • Original: {total_original_size/num_files:,.0f} bytes")
#         print(f"  • Compressed strings: {total_strings_size/num_files:,.0f} bytes")
#         print(f"  • Compressed shapes: {total_shape_size/num_files:,.0f} bytes")
#         print(f"  • Total compressed: {total_compressed_file_size/num_files:,.0f} bytes")
        
#         overall_compression_ratio = total_original_size / total_compressed_file_size if total_compressed_file_size > 0 else float('inf')
#         print(f"  • Overall compression ratio: {overall_compression_ratio:.2f}:1")
    
#     # Save detailed analysis to file if save_path is provided
#     if args.save_path:
#         os.makedirs(args.save_path, exist_ok=True)
#         analysis_file = os.path.join(args.save_path, "size_analysis_report.txt")
        
#         with open(analysis_file, 'w') as f:
#             f.write("=== SIZE ANALYSIS REPORT ===\n\n")
            
#             for result in analysis_results:
#                 f.write(f"Image: {result['img_name']}\n")
#                 f.write(f"Original size: {result['original_size']:,} bytes\n")
#                 f.write(f"Compressed strings size: {result['strings_size']:,} bytes\n")
#                 f.write(f"Compressed shape size: {result['shape_size']:,} bytes\n")
#                 f.write(f"Total compressed file size: {result['compressed_file_size']:,} bytes\n")
#                 f.write(f"Compression ratio: {result['original_size']/result['compressed_file_size']:.2f}:1\n")
#                 f.write("-" * 40 + "\n")
            
#             f.write(f"\nSUMMARY:\n")
#             f.write(f"Total files: {len(analysis_results)}\n")
#             f.write(f"Total original size: {total_original_size:,} bytes\n")
#             f.write(f"Total compressed size: {total_compressed_file_size:,} bytes\n")
#             f.write(f"Overall compression ratio: {total_original_size/total_compressed_file_size:.2f}:1\n")
        
#         print(f"\nDetailed analysis saved to: {analysis_file}")

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2, dim=[1,2,3])
#     return -10 * torch.log10(mse)

# def compute_msssim(a, b):
#     msssim_values = []
#     for i in range(a.size(0)):
#         msssim_val = ms_ssim(a[i:i+1], b[i:i+1], data_range=1.).item()
#         msssim_values.append(-10 * math.log10(1 - msssim_val))
#     return torch.tensor(msssim_values)

# def compute_bpp_batch(out_net):
#     batch_size = out_net['x_hat'].size(0)
#     height, width = out_net['x_hat'].size(2), out_net['x_hat'].size(3)
#     num_pixels = height * width
    
#     bpp_values = []
#     for b in range(batch_size):
#         total_bits = 0
#         for key, likelihoods in out_net['likelihoods'].items():
#             total_bits += torch.log(likelihoods[b]).sum() / (-math.log(2))
#         bpp_values.append(total_bits / num_pixels)
    
#     return torch.tensor(bpp_values)

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
#     parser = argparse.ArgumentParser(description="Image compression/decompression script.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default="./60.5checkpoint_best.pth.tar")
#     parser.add_argument("--data", type=str, help="Path to dataset", default='../datasets/dummy/valid')
#     parser.add_argument("--save_path", default='dataset/compressed_results', type=str, help="Path to save results")
#     parser.add_argument("--compressed_path", default="./dataset/compressed_data", type=str, help="Path to save/load compressed data")
#     parser.add_argument(
#         "--mode", 
#         choices=["compress", "decompress", "both", "size_analysis"], 
#         default="both",
#         help="Choose operation mode: compress, decompress, both, or size_analysis (default: both)"
#     )
#     parser.add_argument(
#         "--real", action="store_true", default=True
#     )
#     parser.set_defaults(real=False)
#     args = parser.parse_args(argv)
#     return args

# def compress_images(net, img_list, device, args, p=128, batch_size=1):
#     """Compress images and save compressed data"""
#     print("=== COMPRESSION MODE ===")
    
#     # Create compressed data directory
#     os.makedirs(args.compressed_path, exist_ok=True)
    
#     net.update()
#     num_batches = (len(img_list) + batch_size - 1) // batch_size
    
#     total_time = 0
#     count = 0
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(img_list))
#         batch_img_paths = img_list[start_idx:end_idx]
#         current_batch_size = len(batch_img_paths)
        
#         print(f"Compressing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
        
#         # Load batch of images
#         x_batch = load_images_batch(batch_img_paths, device)
#         x_batch_padded, padding = pad_batch(x_batch, p)
        
#         count += current_batch_size
        
#         with torch.no_grad():
#             if args.cuda:
#                 torch.cuda.synchronize()
#             s = time.time()
#             out_enc = net.compress(x_batch_padded)
#             if args.cuda:
#                 torch.cuda.synchronize()
#             e = time.time()
            
#             batch_time = (e - s)
#             total_time += batch_time
            
#             # Save compressed data for each image
#             for i in range(current_batch_size):
#                 img_name = os.path.basename(batch_img_paths[i]).split('.')[0]
#                 compressed_filename = os.path.join(args.compressed_path, f"{img_name}_compressed.pkl")
                
#                 # Extract data for single image
#                 single_compressed = {
#                     "strings": [[s[0]] for s in out_enc["strings"]],  # Take first element of each string list
#                     "shape": out_enc["shape"],
#                     "padding": padding,
#                     "original_shape": (x_batch[i].size(1), x_batch[i].size(2))  # H, W of original image
#                 }
                
#                 save_compressed_data(single_compressed, compressed_filename)
                
#                 # Calculate and save compression stats
#                 num_pixels = x_batch[i].size(1) * x_batch[i].size(2)
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                
#                 if args.save_path:
#                     os.makedirs(args.save_path, exist_ok=True)
#                     with open(os.path.join(args.save_path, f"{img_name}_compression_stats.txt"), 'w') as f:
#                         f.write(f'Image: {img_name}\n')
#                         f.write(f'Bitrate: {bit_rate:.3f} bpp\n')
#                         f.write(f'Compression time: {batch_time/current_batch_size*1000:.3f} ms\n')
        
#         print(f'Batch {batch_idx + 1} - Compression time: {batch_time*1000:.3f} ms')
    
#     avg_time = (total_time / count) * 1000
#     print(f'\nCompression Summary:')
#     print(f'Total images compressed: {count}')
#     print(f'Average compression time per image: {avg_time:.3f} ms')
#     print(f'Compressed data saved to: {args.compressed_path}')

# def decompress_images(net, device, args):
#     """Load compressed data and decompress images"""
#     print("=== DECOMPRESSION MODE ===")
#     net.update()
#     if not os.path.exists(args.compressed_path):
#         print(f"Error: Compressed data path {args.compressed_path} does not exist!")
#         return
    
#     # Find all compressed files
#     compressed_files = [f for f in os.listdir(args.compressed_path) if f.endswith('_compressed.pkl')]
    
#     if not compressed_files:
#         print(f"Error: No compressed files found in {args.compressed_path}")
#         return
    
#     if args.save_path:
#         os.makedirs(args.save_path, exist_ok=True)
    
#     total_time = 0
#     count = 0
    
#     for compressed_file in compressed_files:
#         print(f"Decompressing: {compressed_file}")
        
#         # Load compressed data
#         compressed_data = load_compressed_data(os.path.join(args.compressed_path, compressed_file))
        
#         with torch.no_grad():
#             if args.cuda:
#                 torch.cuda.synchronize()
#             s = time.time()
#             out_dec = net.decompress(compressed_data["strings"], compressed_data["shape"])
#             if args.cuda:
#                 torch.cuda.synchronize()
#             e = time.time()
            
#             decode_time = (e - s)
#             total_time += decode_time
#             count += 1
            
#             # Crop the padded image
#             out_dec["x_hat"] = crop_batch(out_dec["x_hat"], compressed_data["padding"])
#             out_dec["x_hat"].clamp_(0, 1)
            
#             # Save decompressed image
#             if args.save_path:
#                 img_name = compressed_file.replace('_compressed.pkl', '')
#                 save_image(out_dec["x_hat"], os.path.join(args.save_path, f"{img_name}_decompressed.png"))
                
#                 # Save decompression stats
#                 with open(os.path.join(args.save_path, f"{img_name}_decompression_stats.txt"), 'w') as f:
#                     f.write(f'Image: {img_name}\n')
#                     f.write(f'Decompression time: {decode_time*1000:.3f} ms\n')
#                     f.write(f'Output shape: {out_dec["x_hat"].shape}\n')
        
#         print(f'Decompression time: {decode_time*1000:.3f} ms')
    
#     avg_time = (total_time / count) * 1000
#     print(f'\nDecompression Summary:')
#     print(f'Total images decompressed: {count}')
#     print(f'Average decompression time per image: {avg_time:.3f} ms')
#     if args.save_path:
#         print(f'Decompressed images saved to: {args.save_path}')

# def main(argv):
#     torch.backends.cudnn.enabled = False
#     args = parse_args(argv)
    
#     if args.cuda:
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
        
#     # For size_analysis mode, we don't need to load the model
#     if args.mode == "size_analysis":
#         analyze_data_size(device, args)
#         return
    
#     net = DCAE()
#     net = net.to(device)
#     net.eval()
    
#     dictory = {}
    
#     if args.checkpoint:  
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location=device)
#         for k, v in checkpoint["state_dict"].items():
#             dictory[k.replace("module.", "")] = v
#         net.load_state_dict(dictory)
    
#     if args.mode == "compress":
#         # Get image list for compression
#         path = args.data
#         img_list = []
#         for file in os.listdir(path):
#             if file[-3:] in ["jpg", "png", "peg"]:
#                 img_list.append(os.path.join(path, file))
        
#         if not img_list:
#             print(f"No images found in {path}")
#             return
            
#         compress_images(net, img_list, device, args)
        
#     elif args.mode == "decompress":
#         decompress_images(net, device, args)
        
#     elif args.mode == "both":
#         # Original functionality - compress and decompress for evaluation
#         print("=== BOTH MODE (Evaluation) ===")
        
#         p = 128
#         path = args.data
#         batch_size = 1
        
#         img_list = []
#         for file in os.listdir(path):
#             if file[-3:] in ["jpg", "png", "peg"]:
#                 img_list.append(os.path.join(path, file))
        
#         count = 0
#         total_PSNR = 0
#         total_Bit_rate = 0
#         total_MS_SSIM = 0
#         total_time = 0
#         encode_time = 0
#         decode_time = 0
        
#         num_batches = (len(img_list) + batch_size - 1) // batch_size
        
#         if args.real:
#             net.update()
#             for batch_idx in range(num_batches):
#                 start_idx = batch_idx * batch_size
#                 end_idx = min((batch_idx + 1) * batch_size, len(img_list))
#                 batch_img_paths = img_list[start_idx:end_idx]
#                 current_batch_size = len(batch_img_paths)
                
#                 print(f"Processing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
                
#                 x_batch = load_images_batch(batch_img_paths, device)
#                 x_batch_padded, padding = pad_batch(x_batch, p)
                
#                 count += current_batch_size
                
#                 with torch.no_grad():
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     s = time.time()
#                     out_enc = net.compress(x_batch_padded)
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     e = time.time()
#                     batch_encode_time = (e - s)
#                     encode_time += batch_encode_time
#                     total_time += batch_encode_time
                    
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     s = time.time()
#                     out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     e = time.time()
#                     batch_decode_time = (e - s)
#                     decode_time += batch_decode_time
#                     total_time += batch_decode_time
                    
#                     out_dec["x_hat"] = crop_batch(out_dec["x_hat"], padding)
                    
#                     for i in range(current_batch_size):
#                         x_single = x_batch[i:i+1]
#                         x_hat_single = out_dec["x_hat"][i:i+1]
#                         num_pixels = x_single.size(2) * x_single.size(3)
                        
#                         bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / (num_pixels * current_batch_size)
                        
#                         total_Bit_rate += bit_rate
#                         total_PSNR += compute_psnr(x_single, x_hat_single).item()
#                         total_MS_SSIM += compute_msssim(x_single, x_hat_single).item()
                    
#                     print(f'Batch {batch_idx + 1} - Encoding time: {batch_encode_time*1000:.3f} ms, Decoding time: {batch_decode_time*1000:.3f} ms')
        
#         # Calculate averages
#         avg_PSNR = total_PSNR / count
#         avg_MS_SSIM = total_MS_SSIM / count
#         avg_Bit_rate = total_Bit_rate / count
#         avg_total_time = (total_time / count) * 1000
#         avg_encode_time = (encode_time / count) * 1000
#         avg_decode_time = (decode_time / count) * 1000
        
#         print(f'Total images processed: {count}')
#         print(f'Average PSNR: {avg_PSNR:.2f} dB')
#         print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
#         print(f'Average Bit-rate: {avg_Bit_rate:.3f} bpp')
#         print(f'Average total time per image: {avg_total_time:.3f} ms')
#         print(f'Average encode time per image: {avg_encode_time:.3f} ms')
#         print(f'Average decode time per image: {avg_decode_time:.3f} ms')

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])


# ### 
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
# import pickle
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

# def save_compressed_data(compressed_data, filename):
#     """Save compressed data to a file"""
#     with open(filename, 'wb') as f:
#         pickle.dump(compressed_data, f)
#     print(f"Compressed data saved to: {filename}")

# def load_compressed_data(filename):
#     """Load compressed data from a file"""
#     with open(filename, 'rb') as f:
#         compressed_data = pickle.load(f)
#     print(f"Compressed data loaded from: {filename}")
#     return compressed_data

# def get_size_in_bytes(obj):
#     """Calculate size of an object in bytes"""
#     if isinstance(obj, torch.Tensor):
#         return obj.numel() * obj.element_size()
#     elif isinstance(obj, (list, tuple)):
#         total_size = 0
#         for item in obj:
#             total_size += get_size_in_bytes(item)
#         return total_size
#     elif isinstance(obj, str):
#         return len(obj.encode('utf-8'))
#     elif isinstance(obj, bytes):
#         return len(obj)
#     elif isinstance(obj, dict):
#         total_size = 0
#         for key, value in obj.items():
#             total_size += get_size_in_bytes(key) + get_size_in_bytes(value)
#         return total_size
#     else:
#         return sys.getsizeof(obj)

# def get_shape_info(obj, name="object"):
#     """Get detailed shape information for various data types"""
#     shape_info = {
#         'type': type(obj).__name__,
#         'shape': None,
#         'detailed_shape': None,
#         'description': None
#     }
    
#     if isinstance(obj, torch.Tensor):
#         shape_info['shape'] = list(obj.shape)
#         shape_info['detailed_shape'] = f"torch.Size({list(obj.shape)})"
#         shape_info['description'] = f"Tensor shape: {list(obj.shape)}"
#         if hasattr(obj, 'dtype'):
#             shape_info['dtype'] = str(obj.dtype)
    
#     elif isinstance(obj, (list, tuple)):
#         shape_info['shape'] = [len(obj)]
#         shape_info['detailed_shape'] = f"Length: {len(obj)}"
        
#         # For nested structures, get more details
#         if len(obj) > 0:
#             first_elem = obj[0]
#             if isinstance(first_elem, (list, tuple)):
#                 nested_lens = [len(item) if isinstance(item, (list, tuple)) else 1 for item in obj]
#                 shape_info['detailed_shape'] = f"Outer length: {len(obj)}, Inner lengths: {nested_lens[:5]}{'...' if len(nested_lens) > 5 else ''}"
                
#                 # Check if it's a list of byte strings
#                 if len(obj) > 0 and len(obj[0]) > 0 and isinstance(obj[0][0], bytes):
#                     byte_lens = []
#                     for sublist in obj:
#                         if isinstance(sublist, list) and len(sublist) > 0:
#                             byte_lens.extend([len(item) for item in sublist if isinstance(item, bytes)])
#                     if byte_lens:
#                         shape_info['description'] = f"List of {len(obj)} sublists, containing byte strings of lengths: {byte_lens[:10]}{'...' if len(byte_lens) > 10 else ''}"
#                         shape_info['detailed_shape'] += f", Total byte strings: {len(byte_lens)}"
#             else:
#                 elem_types = [type(item).__name__ for item in obj[:5]]
#                 shape_info['description'] = f"List of {len(obj)} elements, types: {elem_types}{'...' if len(obj) > 5 else ''}"
#         else:
#             shape_info['description'] = "Empty list"
    
#     elif isinstance(obj, dict):
#         shape_info['shape'] = [len(obj)]
#         shape_info['detailed_shape'] = f"Dict with {len(obj)} keys: {list(obj.keys())}"
#         shape_info['description'] = f"Dictionary with keys: {list(obj.keys())}"
    
#     elif isinstance(obj, (str, bytes)):
#         shape_info['shape'] = [len(obj)]
#         shape_info['detailed_shape'] = f"Length: {len(obj)}"
#         shape_info['description'] = f"{'String' if isinstance(obj, str) else 'Bytes'} of length {len(obj)}"
    
#     elif isinstance(obj, (int, float)):
#         shape_info['shape'] = []  # Scalar
#         shape_info['detailed_shape'] = "Scalar"
#         shape_info['description'] = f"Scalar {type(obj).__name__}: {obj}"
    
#     else:
#         # Try to get shape for other types
#         if hasattr(obj, 'shape'):
#             shape_info['shape'] = list(obj.shape) if hasattr(obj.shape, '__iter__') else [obj.shape]
#             shape_info['detailed_shape'] = f"Shape: {obj.shape}"
#         elif hasattr(obj, '__len__'):
#             try:
#                 shape_info['shape'] = [len(obj)]
#                 shape_info['detailed_shape'] = f"Length: {len(obj)}"
#             except:
#                 shape_info['detailed_shape'] = "Unknown shape"
#         else:
#             shape_info['detailed_shape'] = "No shape information"
        
#         shape_info['description'] = f"Object of type {type(obj).__name__}"
    
#     return shape_info

# def analyze_data_size(device, args):
#     """Analyze and display sizes, shapes, and types of compressed data and original images"""
#     print("=== SIZE ANALYSIS MODE ===")
    
#     if not os.path.exists(args.compressed_path):
#         print(f"Error: Compressed data path {args.compressed_path} does not exist!")
#         return
    
#     if not os.path.exists(args.data):
#         print(f"Error: Original images path {args.data} does not exist!")
#         return
    
#     # Find compressed files
#     compressed_files = [f for f in os.listdir(args.compressed_path) if f.endswith('_compressed.pkl')]
    
#     if not compressed_files:
#         print(f"Error: No compressed files found in {args.compressed_path}")
#         return
    
#     # Get original image files
#     img_list = []
#     for file in os.listdir(args.data):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(os.path.join(args.data, file))
    
#     print(f"\nFound {len(compressed_files)} compressed files and {len(img_list)} original images")
#     print("=" * 100)
    
#     total_strings_size = 0
#     total_shape_size = 0
#     total_original_size = 0
#     total_compressed_file_size = 0
    
#     analysis_results = []
    
#     for i, compressed_file in enumerate(compressed_files):
#         img_name = compressed_file.replace('_compressed.pkl', '')
#         compressed_path = os.path.join(args.compressed_path, compressed_file)
        
#         # Find corresponding original image
#         original_img_path = None
#         for img_path in img_list:
#             if img_name in os.path.basename(img_path):
#                 original_img_path = img_path
#                 break
        
#         if original_img_path is None:
#             print(f"Warning: No original image found for {img_name}")
#             continue
        
#         # Load compressed data
#         compressed_data = load_compressed_data(compressed_path)
        
#         # Load original image
#         original_img = transforms.ToTensor()(Image.open(original_img_path).convert('RGB'))
        
#         # Calculate sizes
#         strings_size = get_size_in_bytes(compressed_data["strings"])
#         shape_size = get_size_in_bytes(compressed_data["shape"])
#         original_size = get_size_in_bytes(original_img)
#         compressed_file_size = os.path.getsize(compressed_path)
        
#         # Get detailed shape information
#         strings_shape_info = get_shape_info(compressed_data["strings"], "strings")
#         shape_shape_info = get_shape_info(compressed_data["shape"], "shape")
#         original_shape_info = get_shape_info(original_img, "original")
        
#         # Store results
#         result = {
#             'img_name': img_name,
#             'strings_size': strings_size,
#             'shape_size': shape_size,
#             'original_size': original_size,
#             'compressed_file_size': compressed_file_size,
#             'strings_shape_info': strings_shape_info,
#             'shape_shape_info': shape_shape_info,
#             'original_shape_info': original_shape_info
#         }
#         analysis_results.append(result)
        
#         # Update totals
#         total_strings_size += strings_size
#         total_shape_size += shape_size
#         total_original_size += original_size
#         total_compressed_file_size += compressed_file_size
        
#         # Print individual results
#         print(f"\n--- Image {i+1}: {img_name} ---")
        
#         # Original Image Analysis
#         print(f"📷 Original Image:")
#         print(f"  • Type: {original_shape_info['type']}")
#         print(f"  • Shape: {original_shape_info['shape']}")
#         print(f"  • Detailed Shape: {original_shape_info['detailed_shape']}")
#         if 'dtype' in original_shape_info:
#             print(f"  • Data Type: {original_shape_info['dtype']}")
#         print(f"  • Size: {original_size:,} bytes ({original_size/1024:.2f} KB)")
#         print(f"  • Description: {original_shape_info['description']}")
        
#         # Compressed Strings Analysis
#         print(f"\n🗜️ Compressed Strings:")
#         print(f"  • Type: {strings_shape_info['type']}")
#         print(f"  • Shape: {strings_shape_info['shape']}")
#         print(f"  • Detailed Shape: {strings_shape_info['detailed_shape']}")
#         print(f"  • Size: {strings_size:,} bytes ({strings_size/1024:.2f} KB)")
#         print(f"  • Description: {strings_shape_info['description']}")
        
#         # Compressed Shape Analysis  
#         print(f"\n📐 Compressed Shape:")
#         print(f"  • Type: {shape_shape_info['type']}")
#         print(f"  • Shape: {shape_shape_info['shape']}")
#         print(f"  • Detailed Shape: {shape_shape_info['detailed_shape']}")
#         print(f"  • Size: {shape_size:,} bytes")
#         print(f"  • Description: {shape_shape_info['description']}")
#         if hasattr(compressed_data["shape"], '__iter__'):
#             try:
#                 print(f"  • Value: {list(compressed_data['shape']) if hasattr(compressed_data['shape'], '__iter__') else compressed_data['shape']}")
#             except:
#                 print(f"  • Value: {compressed_data['shape']}")
        
#         # File size and compression info
#         print(f"\n💾 File Information:")
#         print(f"  • Total compressed file size: {compressed_file_size:,} bytes ({compressed_file_size/1024:.2f} KB)")
        
#         # Compression ratio
#         compression_ratio = original_size / compressed_file_size if compressed_file_size > 0 else float('inf')
#         print(f"  • Compression ratio: {compression_ratio:.2f}:1")
        
#         # Component size breakdown
#         strings_percentage = (strings_size / compressed_file_size * 100) if compressed_file_size > 0 else 0
#         shape_percentage = (shape_size / compressed_file_size * 100) if compressed_file_size > 0 else 0
#         print(f"  • Strings component: {strings_percentage:.1f}% of total")
#         print(f"  • Shape component: {shape_percentage:.1f}% of total")
        
#         print("-" * 60)
    
#     # Print summary
#     print(f"\n{'='*20} SUMMARY {'='*20}")
#     print(f"Total files analyzed: {len(analysis_results)}")
    
#     print(f"\n📊 Total sizes:")
#     print(f"  • Original images: {total_original_size:,} bytes ({total_original_size/1024/1024:.2f} MB)")
#     print(f"  • Compressed strings: {total_strings_size:,} bytes ({total_strings_size/1024:.2f} KB)")
#     print(f"  • Compressed shapes: {total_shape_size:,} bytes ({total_shape_size/1024:.2f} KB)")
#     print(f"  • Total compressed files: {total_compressed_file_size:,} bytes ({total_compressed_file_size/1024:.2f} KB)")
    
#     # Average sizes and shapes
#     num_files = len(analysis_results)
#     if num_files > 0:
#         print(f"\n📈 Average per image:")
#         print(f"  • Original size: {total_original_size/num_files:,.0f} bytes")
#         print(f"  • Compressed strings size: {total_strings_size/num_files:,.0f} bytes")
#         print(f"  • Compressed shapes size: {total_shape_size/num_files:,.0f} bytes")
#         print(f"  • Total compressed size: {total_compressed_file_size/num_files:,.0f} bytes")
        
#         overall_compression_ratio = total_original_size / total_compressed_file_size if total_compressed_file_size > 0 else float('inf')
#         print(f"  • Overall compression ratio: {overall_compression_ratio:.2f}:1")
        
#         # Average component percentages
#         avg_strings_percentage = (total_strings_size / total_compressed_file_size * 100) if total_compressed_file_size > 0 else 0
#         avg_shape_percentage = (total_shape_size / total_compressed_file_size * 100) if total_compressed_file_size > 0 else 0
#         print(f"  • Average strings component: {avg_strings_percentage:.1f}% of compressed size")
#         print(f"  • Average shape component: {avg_shape_percentage:.1f}% of compressed size")
    
#     # Save detailed analysis to file if save_path is provided
#     if args.save_path:
#         os.makedirs(args.save_path, exist_ok=True)
#         analysis_file = os.path.join(args.save_path, "detailed_size_analysis_report.txt")
        
#         with open(analysis_file, 'w') as f:
#             f.write("=== DETAILED SIZE AND SHAPE ANALYSIS REPORT ===\n\n")
            
#             for result in analysis_results:
#                 f.write(f"Image: {result['img_name']}\n")
#                 f.write("-" * 40 + "\n")
                
#                 # Original image info
#                 orig = result['original_shape_info']
#                 f.write(f"Original Image:\n")
#                 f.write(f"  Type: {orig['type']}\n")
#                 f.write(f"  Shape: {orig['shape']}\n")
#                 f.write(f"  Detailed Shape: {orig['detailed_shape']}\n")
#                 f.write(f"  Size: {result['original_size']:,} bytes\n")
#                 f.write(f"  Description: {orig['description']}\n\n")
                
#                 # Compressed strings info
#                 strings = result['strings_shape_info']
#                 f.write(f"Compressed Strings:\n")
#                 f.write(f"  Type: {strings['type']}\n")
#                 f.write(f"  Shape: {strings['shape']}\n")
#                 f.write(f"  Detailed Shape: {strings['detailed_shape']}\n")
#                 f.write(f"  Size: {result['strings_size']:,} bytes\n")
#                 f.write(f"  Description: {strings['description']}\n\n")
                
#                 # Compressed shape info
#                 shape = result['shape_shape_info']
#                 f.write(f"Compressed Shape:\n")
#                 f.write(f"  Type: {shape['type']}\n")
#                 f.write(f"  Shape: {shape['shape']}\n")
#                 f.write(f"  Detailed Shape: {shape['detailed_shape']}\n")
#                 f.write(f"  Size: {result['shape_size']:,} bytes\n")
#                 f.write(f"  Description: {shape['description']}\n\n")
                
#                 f.write(f"Total compressed file size: {result['compressed_file_size']:,} bytes\n")
#                 f.write(f"Compression ratio: {result['original_size']/result['compressed_file_size']:.2f}:1\n")
#                 f.write("-" * 60 + "\n\n")
            
#             f.write(f"SUMMARY:\n")
#             f.write(f"Total files: {len(analysis_results)}\n")
#             f.write(f"Total original size: {total_original_size:,} bytes\n")
#             f.write(f"Total compressed size: {total_compressed_file_size:,} bytes\n")
#             f.write(f"Overall compression ratio: {total_original_size/total_compressed_file_size:.2f}:1\n")
        
#         print(f"\n📄 Detailed analysis saved to: {analysis_file}")

# # ... (rest of the functions remain the same as in the previous version)

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2, dim=[1,2,3])
#     return -10 * torch.log10(mse)

# def compute_msssim(a, b):
#     msssim_values = []
#     for i in range(a.size(0)):
#         msssim_val = ms_ssim(a[i:i+1], b[i:i+1], data_range=1.).item()
#         msssim_values.append(-10 * math.log10(1 - msssim_val))
#     return torch.tensor(msssim_values)

# def compute_bpp_batch(out_net):
#     batch_size = out_net['x_hat'].size(0)
#     height, width = out_net['x_hat'].size(2), out_net['x_hat'].size(3)
#     num_pixels = height * width
    
#     bpp_values = []
#     for b in range(batch_size):
#         total_bits = 0
#         for key, likelihoods in out_net['likelihoods'].items():
#             total_bits += torch.log(likelihoods[b]).sum() / (-math.log(2))
#         bpp_values.append(total_bits / num_pixels)
    
#     return torch.tensor(bpp_values)

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
#     parser = argparse.ArgumentParser(description="Image compression/decompression script.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default="./60.5checkpoint_best.pth.tar")
#     parser.add_argument("--data", type=str, help="Path to dataset", default='../datasets/dummy/valid')
#     parser.add_argument("--save_path", default='eval', type=str, help="Path to save results")
#     parser.add_argument("--compressed_path", default="eval/compressed_medical_pills", type=str, help="Path to save/load compressed data")
#     parser.add_argument(
#         "--mode", 
#         choices=["compress", "decompress", "both", "size_analysis"], 
#         default="both",
#         help="Choose operation mode: compress, decompress, both, or size_analysis (default: both)"
#     )
#     parser.add_argument(
#         "--real", action="store_true", default=True
#     )
#     parser.set_defaults(real=False)
#     args = parser.parse_args(argv)
#     return args

# def compress_images(net, img_list, device, args, p=128, batch_size=1):
#     """Compress images and save compressed data"""
#     print("=== COMPRESSION MODE ===")
    
#     # Create compressed data directory
#     os.makedirs(args.compressed_path, exist_ok=True)
    
#     net.update()
#     num_batches = (len(img_list) + batch_size - 1) // batch_size
    
#     total_time = 0
#     count = 0
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(img_list))
#         batch_img_paths = img_list[start_idx:end_idx]
#         current_batch_size = len(batch_img_paths)
        
#         print(f"Compressing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
        
#         # Load batch of images
#         x_batch = load_images_batch(batch_img_paths, device)
#         x_batch_padded, padding = pad_batch(x_batch, p)
        
#         count += current_batch_size
        
#         with torch.no_grad():
#             if args.cuda:
#                 torch.cuda.synchronize()
#             s = time.time()
#             out_enc = net.compress(x_batch_padded)
#             if args.cuda:
#                 torch.cuda.synchronize()
#             e = time.time()
            
#             batch_time = (e - s)
#             total_time += batch_time
            
#             # Save compressed data for each image
#             for i in range(current_batch_size):
#                 img_name = os.path.basename(batch_img_paths[i]).split('.')[0]
#                 compressed_filename = os.path.join(args.compressed_path, f"{img_name}_compressed.pkl")
                
#                 # Extract data for single image
#                 single_compressed = {
#                     "strings": [[s[0]] for s in out_enc["strings"]],  # Take first element of each string list
#                     "shape": out_enc["shape"],
#                     "padding": padding,
#                     "original_shape": (x_batch[i].size(1), x_batch[i].size(2))  # H, W of original image
#                 }
                
#                 save_compressed_data(single_compressed, compressed_filename)
                
#                 # Calculate and save compression stats
#                 num_pixels = x_batch[i].size(1) * x_batch[i].size(2)
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                
#                 if args.save_path:
#                     os.makedirs(args.save_path, exist_ok=True)
#                     with open(os.path.join(args.save_path, f"{img_name}_compression_stats.txt"), 'w') as f:
#                         f.write(f'Image: {img_name}\n')
#                         f.write(f'Bitrate: {bit_rate:.3f} bpp\n')
#                         f.write(f'Compression time: {batch_time/current_batch_size*1000:.3f} ms\n')
        
#         print(f'Batch {batch_idx + 1} - Compression time: {batch_time*1000:.3f} ms')
    
#     avg_time = (total_time / count) * 1000
#     print(f'\nCompression Summary:')
#     print(f'Total images compressed: {count}')
#     print(f'Average compression time per image: {avg_time:.3f} ms')
#     print(f'Compressed data saved to: {args.compressed_path}')

# def decompress_images(net, device, args):
#     """Load compressed data and decompress images"""
#     print("=== DECOMPRESSION MODE ===")
#     net.update()
#     if not os.path.exists(args.compressed_path):
#         print(f"Error: Compressed data path {args.compressed_path} does not exist!")
#         return
    
#     # Find all compressed files
#     compressed_files = [f for f in os.listdir(args.compressed_path) if f.endswith('_compressed.pkl')]
    
#     if not compressed_files:
#         print(f"Error: No compressed files found in {args.compressed_path}")
#         return
    
#     if args.save_path:
#         os.makedirs(args.save_path, exist_ok=True)
    
#     total_time = 0
#     count = 0
    
#     for compressed_file in compressed_files:
#         print(f"Decompressing: {compressed_file}")
        
#         # Load compressed data
#         compressed_data = load_compressed_data(os.path.join(args.compressed_path, compressed_file))
        
#         with torch.no_grad():
#             if args.cuda:
#                 torch.cuda.synchronize()
#             s = time.time()
#             out_dec = net.decompress(compressed_data["strings"], compressed_data["shape"])
#             if args.cuda:
#                 torch.cuda.synchronize()
#             e = time.time()
            
#             decode_time = (e - s)
#             total_time += decode_time
#             count += 1
            
#             # Crop the padded image
#             out_dec["x_hat"] = crop_batch(out_dec["x_hat"], compressed_data["padding"])
#             out_dec["x_hat"].clamp_(0, 1)
            
#             # Save decompressed image
#             if args.save_path:
#                 img_name = compressed_file.replace('_compressed.pkl', '')
#                 save_image(out_dec["x_hat"], os.path.join(args.save_path, f"{img_name}_decompressed.png"))
                
#                 # Save decompression stats
#                 with open(os.path.join(args.save_path, f"{img_name}_decompression_stats.txt"), 'w') as f:
#                     f.write(f'Image: {img_name}\n')
#                     f.write(f'Decompression time: {decode_time*1000:.3f} ms\n')
#                     f.write(f'Output shape: {out_dec["x_hat"].shape}\n')
        
#         print(f'Decompression time: {decode_time*1000:.3f} ms')
    
#     avg_time = (total_time / count) * 1000
#     print(f'\nDecompression Summary:')
#     print(f'Total images decompressed: {count}')
#     print(f'Average decompression time per image: {avg_time:.3f} ms')
#     if args.save_path:
#         print(f'Decompressed images saved to: {args.save_path}')

# def main(argv):
#     torch.backends.cudnn.enabled = False
#     args = parse_args(argv)
    
#     if args.cuda:
#         device = 'cuda:0'
#     else:
#         device = 'cpu'
        
#     # For size_analysis mode, we don't need to load the model
#     if args.mode == "size_analysis":
#         analyze_data_size(device, args)
#         return
    
#     net = DCAE()
#     net = net.to(device)

#     param_size = 0
#     buffer_size = 0

#     for param in net.parameters():
#         param_size += param.nelement() * param.element_size()

#     for buffer in net.buffers():
#         buffer_size += buffer.nelement() * buffer.element_size()

#     size_all_mb = (param_size + buffer_size) / 1024**2
    
    
#     net.eval()

    
#     dictory = {}
    
#     if args.checkpoint:  
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location=device)
#         for k, v in checkpoint["state_dict"].items():
#             dictory[k.replace("module.", "")] = v
#         net.load_state_dict(dictory)
    
#     if args.mode == "compress":
#         # Get image list for compression
#         path = args.data
#         img_list = []
#         for file in os.listdir(path):
#             if file[-3:] in ["jpg", "png", "peg"]:
#                 img_list.append(os.path.join(path, file))
        
#         if not img_list:
#             print(f"No images found in {path}")
#             return
            
#         compress_images(net, img_list, device, args)
#         print(f'Model size: {size_all_mb:.2f} MB')
        
#     elif args.mode == "decompress":
#         decompress_images(net, device, args)
#         print(f'Model size: {size_all_mb:.2f} MB')
#         print(f'Model size: {size_all_mb:.2f} MB')
        
#     elif args.mode == "both":
#         # Original functionality - compress and decompress for evaluation
#         print("=== BOTH MODE (Evaluation) ===")
        
#         p = 128
#         path = args.data
#         batch_size = 1
        
#         img_list = []
#         for file in os.listdir(path):
#             if file[-3:] in ["jpg", "png", "peg"]:
#                 img_list.append(os.path.join(path, file))
        
#         count = 0
#         total_PSNR = 0
#         total_Bit_rate = 0
#         total_MS_SSIM = 0
#         total_time = 0
#         encode_time = 0
#         decode_time = 0
        
#         num_batches = (len(img_list) + batch_size - 1) // batch_size
        
#         if args.real:
#             net.update()
#             for batch_idx in range(num_batches):
#                 start_idx = batch_idx * batch_size
#                 end_idx = min((batch_idx + 1) * batch_size, len(img_list))
#                 batch_img_paths = img_list[start_idx:end_idx]
#                 current_batch_size = len(batch_img_paths)
                
#                 print(f"Processing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
                
#                 x_batch = load_images_batch(batch_img_paths, device)
#                 x_batch_padded, padding = pad_batch(x_batch, p)
                
#                 count += current_batch_size
                
#                 with torch.no_grad():
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     s = time.time()
#                     out_enc = net.compress(x_batch_padded)
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     e = time.time()
#                     batch_encode_time = (e - s)
#                     encode_time += batch_encode_time
#                     total_time += batch_encode_time
                    
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     s = time.time()
#                     out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     e = time.time()
#                     batch_decode_time = (e - s)
#                     decode_time += batch_decode_time
#                     total_time += batch_decode_time
                    
#                     out_dec["x_hat"] = crop_batch(out_dec["x_hat"], padding)
                    
#                     for i in range(current_batch_size):
#                         x_single = x_batch[i:i+1]
#                         x_hat_single = out_dec["x_hat"][i:i+1]
#                         num_pixels = x_single.size(2) * x_single.size(3)
                        
#                         bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / (num_pixels * current_batch_size)
                        
#                         total_Bit_rate += bit_rate
#                         total_PSNR += compute_psnr(x_single, x_hat_single).item()
#                         total_MS_SSIM += compute_msssim(x_single, x_hat_single).item()
                    
#                     print(f'Batch {batch_idx + 1} - Encoding time: {batch_encode_time*1000:.3f} ms, Decoding time: {batch_decode_time*1000:.3f} ms')
        
#         # Calculate averages
#         avg_PSNR = total_PSNR / count
#         avg_MS_SSIM = total_MS_SSIM / count
#         avg_Bit_rate = total_Bit_rate / count
#         avg_total_time = (total_time / count) * 1000
#         avg_encode_time = (encode_time / count) * 1000
#         avg_decode_time = (decode_time / count) * 1000
        
#         print(f'Total images processed: {count}')
#         print(f'Average PSNR: {avg_PSNR:.2f} dB')
#         print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
#         print(f'Average Bit-rate: {avg_Bit_rate:.3f} bpp')
#         print(f'Average total time per image: {avg_total_time:.3f} ms')
#         print(f'Average encode time per image: {avg_encode_time:.3f} ms')
#         print(f'Average decode time per image: {avg_decode_time:.3f} ms')
#         print(f'Model size: {size_all_mb:.2f} MB')

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])




#  ## duoble, skip first image, 
# ### 
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
# import pickle
# from pytorch_msssim import ms_ssim
# from PIL import Image
# from thop import profile
# warnings.filterwarnings("ignore")
# torch.set_num_threads(10)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.cuda.manual_seed(42)
# import numpy as np
# np.random.seed(42)

# print(torch.cuda.is_available())

# def save_image(tensor, filename):
#     img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
#     img.save(filename)

# def save_metrics(filename, psnr, bitrate, msssim):
#     with open(filename, 'w') as f:
#         f.write(f'PSNR: {psnr:.2f}dB\n')
#         f.write(f'Bitrate: {bitrate:.3f}bpp\n')
#         f.write(f'MS-SSIM: {msssim:.4f}\n')

# def save_compressed_data(compressed_data, filename):
#     """Save compressed data to a file"""
#     with open(filename, 'wb') as f:
#         pickle.dump(compressed_data, f)
#     print(f"Compressed data saved to: {filename}")

# def load_compressed_data(filename):
#     """Load compressed data from a file"""
#     with open(filename, 'rb') as f:
#         compressed_data = pickle.load(f)
#     print(f"Compressed data loaded from: {filename}")
#     return compressed_data

# def get_size_in_bytes(obj):
#     """Calculate size of an object in bytes"""
#     if isinstance(obj, torch.Tensor):
#         return obj.numel() * obj.element_size()
#     elif isinstance(obj, (list, tuple)):
#         total_size = 0
#         for item in obj:
#             total_size += get_size_in_bytes(item)
#         return total_size
#     elif isinstance(obj, str):
#         return len(obj.encode('utf-8'))
#     elif isinstance(obj, bytes):
#         return len(obj)
#     elif isinstance(obj, dict):
#         total_size = 0
#         for key, value in obj.items():
#             total_size += get_size_in_bytes(key) + get_size_in_bytes(value)
#         return total_size
#     else:
#         return sys.getsizeof(obj)

# def get_shape_info(obj, name="object"):
#     """Get detailed shape information for various data types"""
#     shape_info = {
#         'type': type(obj).__name__,
#         'shape': None,
#         'detailed_shape': None,
#         'description': None
#     }
    
#     if isinstance(obj, torch.Tensor):
#         shape_info['shape'] = list(obj.shape)
#         shape_info['detailed_shape'] = f"torch.Size({list(obj.shape)})"
#         shape_info['description'] = f"Tensor shape: {list(obj.shape)}"
#         if hasattr(obj, 'dtype'):
#             shape_info['dtype'] = str(obj.dtype)
    
#     elif isinstance(obj, (list, tuple)):
#         shape_info['shape'] = [len(obj)]
#         shape_info['detailed_shape'] = f"Length: {len(obj)}"
        
#         # For nested structures, get more details
#         if len(obj) > 0:
#             first_elem = obj[0]
#             if isinstance(first_elem, (list, tuple)):
#                 nested_lens = [len(item) if isinstance(item, (list, tuple)) else 1 for item in obj]
#                 shape_info['detailed_shape'] = f"Outer length: {len(obj)}, Inner lengths: {nested_lens[:5]}{'...' if len(nested_lens) > 5 else ''}"
                
#                 # Check if it's a list of byte strings
#                 if len(obj) > 0 and len(obj[0]) > 0 and isinstance(obj[0][0], bytes):
#                     byte_lens = []
#                     for sublist in obj:
#                         if isinstance(sublist, list) and len(sublist) > 0:
#                             byte_lens.extend([len(item) for item in sublist if isinstance(item, bytes)])
#                     if byte_lens:
#                         shape_info['description'] = f"List of {len(obj)} sublists, containing byte strings of lengths: {byte_lens[:10]}{'...' if len(byte_lens) > 10 else ''}"
#                         shape_info['detailed_shape'] += f", Total byte strings: {len(byte_lens)}"
#             else:
#                 elem_types = [type(item).__name__ for item in obj[:5]]
#                 shape_info['description'] = f"List of {len(obj)} elements, types: {elem_types}{'...' if len(obj) > 5 else ''}"
#         else:
#             shape_info['description'] = "Empty list"
    
#     elif isinstance(obj, dict):
#         shape_info['shape'] = [len(obj)]
#         shape_info['detailed_shape'] = f"Dict with {len(obj)} keys: {list(obj.keys())}"
#         shape_info['description'] = f"Dictionary with keys: {list(obj.keys())}"
    
#     elif isinstance(obj, (str, bytes)):
#         shape_info['shape'] = [len(obj)]
#         shape_info['detailed_shape'] = f"Length: {len(obj)}"
#         shape_info['description'] = f"{'String' if isinstance(obj, str) else 'Bytes'} of length {len(obj)}"
    
#     elif isinstance(obj, (int, float)):
#         shape_info['shape'] = []  # Scalar
#         shape_info['detailed_shape'] = "Scalar"
#         shape_info['description'] = f"Scalar {type(obj).__name__}: {obj}"
    
#     else:
#         # Try to get shape for other types
#         if hasattr(obj, 'shape'):
#             shape_info['shape'] = list(obj.shape) if hasattr(obj.shape, '__iter__') else [obj.shape]
#             shape_info['detailed_shape'] = f"Shape: {obj.shape}"
#         elif hasattr(obj, '__len__'):
#             try:
#                 shape_info['shape'] = [len(obj)]
#                 shape_info['detailed_shape'] = f"Length: {len(obj)}"
#             except:
#                 shape_info['detailed_shape'] = "Unknown shape"
#         else:
#             shape_info['detailed_shape'] = "No shape information"
        
#         shape_info['description'] = f"Object of type {type(obj).__name__}"
    
#     return shape_info

# def analyze_data_size(device, args):
#     """Analyze and display sizes, shapes, and types of compressed data and original images"""
#     print("=== SIZE ANALYSIS MODE ===")
    
#     if not os.path.exists(args.compressed_path):
#         print(f"Error: Compressed data path {args.compressed_path} does not exist!")
#         return
    
#     if not os.path.exists(args.data):
#         print(f"Error: Original images path {args.data} does not exist!")
#         return
    
#     # Find compressed files
#     compressed_files = [f for f in os.listdir(args.compressed_path) if f.endswith('_compressed.pkl')]
    
#     if not compressed_files:
#         print(f"Error: No compressed files found in {args.compressed_path}")
#         return
    
#     # Get original image files
#     img_list = []
#     for file in os.listdir(args.data):
#         if file[-3:] in ["jpg", "png", "peg"]:
#             img_list.append(os.path.join(args.data, file))
    
#     print(f"\nFound {len(compressed_files)} compressed files and {len(img_list)} original images")
#     print("=" * 100)
    
#     total_strings_size = 0
#     total_shape_size = 0
#     total_original_size = 0
#     total_compressed_file_size = 0
    
#     analysis_results = []
    
#     for i, compressed_file in enumerate(compressed_files):
#         img_name = compressed_file.replace('_compressed.pkl', '')
#         compressed_path = os.path.join(args.compressed_path, compressed_file)
        
#         # Find corresponding original image
#         original_img_path = None
#         for img_path in img_list:
#             if img_name in os.path.basename(img_path):
#                 original_img_path = img_path
#                 break
        
#         if original_img_path is None:
#             print(f"Warning: No original image found for {img_name}")
#             continue
        
#         # Load compressed data
#         compressed_data = load_compressed_data(compressed_path)
        
#         # Load original image
#         original_img = transforms.ToTensor()(Image.open(original_img_path).convert('RGB'))
        
#         # Calculate sizes
#         strings_size = get_size_in_bytes(compressed_data["strings"])
#         shape_size = get_size_in_bytes(compressed_data["shape"])
#         original_size = get_size_in_bytes(original_img)
#         compressed_file_size = os.path.getsize(compressed_path)
        
#         # Get detailed shape information
#         strings_shape_info = get_shape_info(compressed_data["strings"], "strings")
#         shape_shape_info = get_shape_info(compressed_data["shape"], "shape")
#         original_shape_info = get_shape_info(original_img, "original")
        
#         # Store results
#         result = {
#             'img_name': img_name,
#             'strings_size': strings_size,
#             'shape_size': shape_size,
#             'original_size': original_size,
#             'compressed_file_size': compressed_file_size,
#             'strings_shape_info': strings_shape_info,
#             'shape_shape_info': shape_shape_info,
#             'original_shape_info': original_shape_info
#         }
#         analysis_results.append(result)
        
#         # Update totals
#         total_strings_size += strings_size
#         total_shape_size += shape_size
#         total_original_size += original_size
#         total_compressed_file_size += compressed_file_size
        
#         # Print individual results
#         print(f"\n--- Image {i+1}: {img_name} ---")
        
#         # Original Image Analysis
#         print(f"📷 Original Image:")
#         print(f"  • Type: {original_shape_info['type']}")
#         print(f"  • Shape: {original_shape_info['shape']}")
#         print(f"  • Detailed Shape: {original_shape_info['detailed_shape']}")
#         if 'dtype' in original_shape_info:
#             print(f"  • Data Type: {original_shape_info['dtype']}")
#         print(f"  • Size: {original_size:,} bytes ({original_size/1024:.2f} KB)")
#         print(f"  • Description: {original_shape_info['description']}")
        
#         # Compressed Strings Analysis
#         print(f"\n🗜️ Compressed Strings:")
#         print(f"  • Type: {strings_shape_info['type']}")
#         print(f"  • Shape: {strings_shape_info['shape']}")
#         print(f"  • Detailed Shape: {strings_shape_info['detailed_shape']}")
#         print(f"  • Size: {strings_size:,} bytes ({strings_size/1024:.2f} KB)")
#         print(f"  • Description: {strings_shape_info['description']}")
        
#         # Compressed Shape Analysis  
#         print(f"\n📐 Compressed Shape:")
#         print(f"  • Type: {shape_shape_info['type']}")
#         print(f"  • Shape: {shape_shape_info['shape']}")
#         print(f"  • Detailed Shape: {shape_shape_info['detailed_shape']}")
#         print(f"  • Size: {shape_size:,} bytes")
#         print(f"  • Description: {shape_shape_info['description']}")
#         if hasattr(compressed_data["shape"], '__iter__'):
#             try:
#                 print(f"  • Value: {list(compressed_data['shape']) if hasattr(compressed_data['shape'], '__iter__') else compressed_data['shape']}")
#             except:
#                 print(f"  • Value: {compressed_data['shape']}")
        
#         # File size and compression info
#         print(f"\n💾 File Information:")
#         print(f"  • Total compressed file size: {compressed_file_size:,} bytes ({compressed_file_size/1024:.2f} KB)")
        
#         # Compression ratio
#         compression_ratio = original_size / compressed_file_size if compressed_file_size > 0 else float('inf')
#         print(f"  • Compression ratio: {compression_ratio:.2f}:1")
        
#         # Component size breakdown
#         strings_percentage = (strings_size / compressed_file_size * 100) if compressed_file_size > 0 else 0
#         shape_percentage = (shape_size / compressed_file_size * 100) if compressed_file_size > 0 else 0
#         print(f"  • Strings component: {strings_percentage:.1f}% of total")
#         print(f"  • Shape component: {shape_percentage:.1f}% of total")
        
#         print("-" * 60)
    
#     # Print summary
#     print(f"\n{'='*20} SUMMARY {'='*20}")
#     print(f"Total files analyzed: {len(analysis_results)}")
    
#     print(f"\n📊 Total sizes:")
#     print(f"  • Original images: {total_original_size:,} bytes ({total_original_size/1024/1024:.2f} MB)")
#     print(f"  • Compressed strings: {total_strings_size:,} bytes ({total_strings_size/1024:.2f} KB)")
#     print(f"  • Compressed shapes: {total_shape_size:,} bytes ({total_shape_size/1024:.2f} KB)")
#     print(f"  • Total compressed files: {total_compressed_file_size:,} bytes ({total_compressed_file_size/1024:.2f} KB)")
    
#     # Average sizes and shapes
#     num_files = len(analysis_results)
#     if num_files > 0:
#         print(f"\n📈 Average per image:")
#         print(f"  • Original size: {total_original_size/num_files:,.0f} bytes")
#         print(f"  • Compressed strings size: {total_strings_size/num_files:,.0f} bytes")
#         print(f"  • Compressed shapes size: {total_shape_size/num_files:,.0f} bytes")
#         print(f"  • Total compressed size: {total_compressed_file_size/num_files:,.0f} bytes")
        
#         overall_compression_ratio = total_original_size / total_compressed_file_size if total_compressed_file_size > 0 else float('inf')
#         print(f"  • Overall compression ratio: {overall_compression_ratio:.2f}:1")
        
#         # Average component percentages
#         avg_strings_percentage = (total_strings_size / total_compressed_file_size * 100) if total_compressed_file_size > 0 else 0
#         avg_shape_percentage = (total_shape_size / total_compressed_file_size * 100) if total_compressed_file_size > 0 else 0
#         print(f"  • Average strings component: {avg_strings_percentage:.1f}% of compressed size")
#         print(f"  • Average shape component: {avg_shape_percentage:.1f}% of compressed size")
    
#     # Save detailed analysis to file if save_path is provided
#     if args.save_path:
#         os.makedirs(args.save_path, exist_ok=True)
#         analysis_file = os.path.join(args.save_path, "detailed_size_analysis_report.txt")
        
#         with open(analysis_file, 'w') as f:
#             f.write("=== DETAILED SIZE AND SHAPE ANALYSIS REPORT ===\n\n")
            
#             for result in analysis_results:
#                 f.write(f"Image: {result['img_name']}\n")
#                 f.write("-" * 40 + "\n")
                
#                 # Original image info
#                 orig = result['original_shape_info']
#                 f.write(f"Original Image:\n")
#                 f.write(f"  Type: {orig['type']}\n")
#                 f.write(f"  Shape: {orig['shape']}\n")
#                 f.write(f"  Detailed Shape: {orig['detailed_shape']}\n")
#                 f.write(f"  Size: {result['original_size']:,} bytes\n")
#                 f.write(f"  Description: {orig['description']}\n\n")
                
#                 # Compressed strings info
#                 strings = result['strings_shape_info']
#                 f.write(f"Compressed Strings:\n")
#                 f.write(f"  Type: {strings['type']}\n")
#                 f.write(f"  Shape: {strings['shape']}\n")
#                 f.write(f"  Detailed Shape: {strings['detailed_shape']}\n")
#                 f.write(f"  Size: {result['strings_size']:,} bytes\n")
#                 f.write(f"  Description: {strings['description']}\n\n")
                
#                 # Compressed shape info
#                 shape = result['shape_shape_info']
#                 f.write(f"Compressed Shape:\n")
#                 f.write(f"  Type: {shape['type']}\n")
#                 f.write(f"  Shape: {shape['shape']}\n")
#                 f.write(f"  Detailed Shape: {shape['detailed_shape']}\n")
#                 f.write(f"  Size: {result['shape_size']:,} bytes\n")
#                 f.write(f"  Description: {shape['description']}\n\n")
                
#                 f.write(f"Total compressed file size: {result['compressed_file_size']:,} bytes\n")
#                 f.write(f"Compression ratio: {result['original_size']/result['compressed_file_size']:.2f}:1\n")
#                 f.write("-" * 60 + "\n\n")
            
#             f.write(f"SUMMARY:\n")
#             f.write(f"Total files: {len(analysis_results)}\n")
#             f.write(f"Total original size: {total_original_size:,} bytes\n")
#             f.write(f"Total compressed size: {total_compressed_file_size:,} bytes\n")
#             f.write(f"Overall compression ratio: {total_original_size/total_compressed_file_size:.2f}:1\n")
        
#         print(f"\n📄 Detailed analysis saved to: {analysis_file}")

# # ... (rest of the functions remain the same as in the previous version)

# def compute_psnr(a, b):
#     mse = torch.mean((a - b)**2, dim=[1,2,3])
#     return -10 * torch.log10(mse)

# def compute_msssim(a, b):
#     msssim_values = []
#     for i in range(a.size(0)):
#         msssim_val = ms_ssim(a[i:i+1], b[i:i+1], data_range=1.).item()
#         msssim_values.append(-10 * math.log10(1 - msssim_val))
#     return torch.tensor(msssim_values)

# def compute_bpp_batch(out_net):
#     batch_size = out_net['x_hat'].size(0)
#     height, width = out_net['x_hat'].size(2), out_net['x_hat'].size(3)
#     num_pixels = height * width
    
#     bpp_values = []
#     for b in range(batch_size):
#         total_bits = 0
#         for key, likelihoods in out_net['likelihoods'].items():
#             total_bits += torch.log(likelihoods[b]).sum() / (-math.log(2))
#         bpp_values.append(total_bits / num_pixels)
    
#     return torch.tensor(bpp_values)

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
#     parser = argparse.ArgumentParser(description="Image compression/decompression script.")
#     parser.add_argument("--cuda", action="store_true", help="Use cuda")
#     parser.add_argument(
#         "--clip_max_norm",
#         default=1.0,
#         type=float,
#         help="gradient clipping max norm (default: %(default)s",
#     )
#     parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default="60.5checkpoint_best.pth.tar")
#     parser.add_argument("--data", type=str, help="Path to dataset", default='/home/xie/datasets/dummy/valid')
#     parser.add_argument("--save_path", default='eval', type=str, help="Path to save results")
#     parser.add_argument("--compressed_path", default="eval/compressed_gpu", type=str, help="Path to save/load compressed data")
#     parser.add_argument(
#         "--mode", 
#         choices=["compress", "decompress", "both", "size_analysis"], 
#         default="both",
#         help="Choose operation mode: compress, decompress, both, or size_analysis (default: both)"
#     )
#     parser.add_argument(
#         "--real", action="store_true", default=True
#     )
#     parser.set_defaults(real=False)
#     args = parser.parse_args(argv)
#     return args

# def compress_images(net, img_list, device, args, p=128, batch_size=1):
#     """Compress images and save compressed data"""
#     print("=== COMPRESSION MODE ===")
    
#     # Create compressed data directory
#     os.makedirs(args.compressed_path, exist_ok=True)
    
#     net.update()
#     num_batches = (len(img_list) + batch_size - 1) // batch_size
    
#     total_time = 0
#     count = 0
    
#     # Binary strings size tracking
#     total_strings_size = 0
#     total_images_processed = 0
#     strings_size_per_batch = []
    
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(img_list))
#         batch_img_paths = img_list[start_idx:end_idx]
#         current_batch_size = len(batch_img_paths)
        
#         print(f"Compressing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
        
#         # Load batch of images
#         x_batch = load_images_batch(batch_img_paths, device)
#         x_batch_padded, padding = pad_batch(x_batch, p)
        
#         # Only count images from batch 2 onwards for timing
#         if batch_idx > 0:
#             count += current_batch_size
        
#         # Always count all images for data size analysis
#         total_images_processed += current_batch_size
        
#         with torch.no_grad():
#             if args.cuda:
#                 torch.cuda.synchronize()
#             s = time.time()
#             out_enc = net.compress(x_batch_padded)
#             if args.cuda:
#                 torch.cuda.synchronize()
#             e = time.time()
            
#             batch_time = (e - s)
            
#             # Only add time from batch 2 onwards
#             if batch_idx > 0:
#                 total_time += batch_time
            
#             # Calculate binary strings size for this batch
#             batch_strings_size = 0
#             for string_list in out_enc["strings"]:
#                 for binary_string in string_list:
#                     if isinstance(binary_string, bytes):
#                         batch_strings_size += len(binary_string)
#                     elif isinstance(binary_string, str):
#                         batch_strings_size += len(binary_string.encode('utf-8'))
#                     else:
#                         # Handle other types (e.g., if it's a tensor or list)
#                         batch_strings_size += get_size_in_bytes(binary_string)
            
#             total_strings_size += batch_strings_size
#             strings_size_per_batch.append({
#                 'batch_idx': batch_idx + 1,
#                 'size_bytes': batch_strings_size,
#                 'size_kb': batch_strings_size / 1024,
#                 'avg_per_image': batch_strings_size / current_batch_size,
#                 'num_images': current_batch_size
#             })
            
#             # Save compressed data for each image (regardless of batch)
#             for i in range(current_batch_size):
#                 img_name = os.path.basename(batch_img_paths[i]).split('.')[0]
#                 compressed_filename = os.path.join(args.compressed_path, f"{img_name}_compressed.pkl")
                
#                 # Extract data for single image
#                 single_compressed = {
#                     "strings": [[s[0]] for s in out_enc["strings"]],
#                     "shape": out_enc["shape"],
#                     "padding": padding,
#                     "original_shape": (x_batch[i].size(1), x_batch[i].size(2))
#                 }
                
#                 save_compressed_data(single_compressed, compressed_filename)
                
#                 # Calculate and save compression stats including strings size
#                 num_pixels = x_batch[i].size(1) * x_batch[i].size(2)
#                 bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                
#                 # Calculate individual image strings size
#                 img_strings_size = sum(len(s[0]) for s in out_enc["strings"])
                
#                 if args.save_path:
#                     os.makedirs(args.save_path, exist_ok=True)
#                     with open(os.path.join(args.save_path, f"{img_name}_compression_stats.txt"), 'w') as f:
#                         f.write(f'Image: {img_name}\n')
#                         f.write(f'Bitrate: {bit_rate:.3f} bpp\n')
#                         f.write(f'Strings size: {img_strings_size} bytes ({img_strings_size/1024:.2f} KB)\n')
#                         f.write(f'Compression time: {batch_time/current_batch_size*1000:.3f} ms\n')
        
#         timing_note = " (excluded from timing)" if batch_idx == 0 else ""
#         print(f'Batch {batch_idx + 1} - Compression time: {batch_time*1000:.3f} ms, Strings size: {batch_strings_size/1024:.2f} KB{timing_note}')
    
#     # Print detailed binary strings size analysis
#     print(f'\n{"="*60}')
#     print(f'BINARY STRINGS SIZE ANALYSIS')
#     print(f'{"="*60}')
    
#     print(f'\nPer-batch breakdown:')
#     print(f'{"Batch":<8} {"Images":<8} {"Size (KB)":<12} {"Avg/Image (KB)":<15} {"Size (bytes)":<12}')
#     print(f'{"-"*60}')
    
#     for batch_info in strings_size_per_batch:
#         print(f'{batch_info["batch_idx"]:<8} {batch_info["num_images"]:<8} '
#               f'{batch_info["size_kb"]:<12.2f} {batch_info["avg_per_image"]/1024:<15.2f} '
#               f'{batch_info["size_bytes"]:<12}')
    
#     # Overall summary
#     avg_strings_size_per_image = total_strings_size / total_images_processed if total_images_processed > 0 else 0
    
#     print(f'\n{"="*40}')
#     print(f'OVERALL STRINGS SIZE SUMMARY')
#     print(f'{"="*40}')
#     print(f'Total images processed: {total_images_processed}')
#     print(f'Total strings size: {total_strings_size:,} bytes ({total_strings_size/1024:.2f} KB, {total_strings_size/1024/1024:.2f} MB)')
#     print(f'Average strings size per image: {avg_strings_size_per_image:.0f} bytes ({avg_strings_size_per_image/1024:.2f} KB)')
    
#     if total_images_processed > 0:
#         # Calculate additional statistics
#         min_batch_size = min(batch_info["avg_per_image"] for batch_info in strings_size_per_batch)
#         max_batch_size = max(batch_info["avg_per_image"] for batch_info in strings_size_per_batch)
        
#         print(f'Min average per image: {min_batch_size:.0f} bytes ({min_batch_size/1024:.2f} KB)')
#         print(f'Max average per image: {max_batch_size:.0f} bytes ({max_batch_size/1024:.2f} KB)')
        
#         # Calculate estimated bitrate from strings size
#         if len(img_list) > 0:
#             # Estimate average image dimensions (this is approximate)
#             sample_img = transforms.ToTensor()(Image.open(img_list[0]).convert('RGB'))
#             avg_pixels = sample_img.size(1) * sample_img.size(2)  # H * W
#             estimated_bpp = (avg_strings_size_per_image * 8) / avg_pixels
#             print(f'Estimated bitrate from strings: {estimated_bpp:.3f} bpp (based on sample image dimensions)')
    
#     # Timing summary
#     print(f'\n{"="*40}')
#     print(f'TIMING SUMMARY')
#     print(f'{"="*40}')
    
#     if count > 0:
#         avg_time = (total_time / count) * 1000
#         print(f'Images used for timing: {count} (excluding first batch)')
#         print(f'Average compression time per image: {avg_time:.3f} ms')
#     else:
#         print(f'Only first batch processed, no timing data available')
    
#     print(f'Compressed data saved to: {args.compressed_path}')
    
#     # Save detailed strings analysis to file if save_path is provided
#     if args.save_path:
#         os.makedirs(args.save_path, exist_ok=True)
#         analysis_file = os.path.join(args.save_path, "strings_size_analysis.txt")
        
#         with open(analysis_file, 'w') as f:
#             f.write("BINARY STRINGS SIZE ANALYSIS REPORT\n")
#             f.write("="*50 + "\n\n")
            
#             f.write("Per-batch breakdown:\n")
#             f.write(f'{"Batch":<8} {"Images":<8} {"Size (KB)":<12} {"Avg/Image (KB)":<15} {"Size (bytes)":<12}\n')
#             f.write("-"*60 + "\n")
            
#             for batch_info in strings_size_per_batch:
#                 f.write(f'{batch_info["batch_idx"]:<8} {batch_info["num_images"]:<8} '
#                        f'{batch_info["size_kb"]:<12.2f} {batch_info["avg_per_image"]/1024:<15.2f} '
#                        f'{batch_info["size_bytes"]:<12}\n')
            
#             f.write(f'\nOVERALL SUMMARY:\n')
#             f.write(f'Total images processed: {total_images_processed}\n')
#             f.write(f'Total strings size: {total_strings_size:,} bytes ({total_strings_size/1024:.2f} KB)\n')
#             f.write(f'Average strings size per image: {avg_strings_size_per_image:.0f} bytes ({avg_strings_size_per_image/1024:.2f} KB)\n')
            
#             if total_images_processed > 0:
#                 f.write(f'Min average per image: {min_batch_size:.0f} bytes ({min_batch_size/1024:.2f} KB)\n')
#                 f.write(f'Max average per image: {max_batch_size:.0f} bytes ({max_batch_size/1024:.2f} KB)\n')
#                 if len(img_list) > 0:
#                     f.write(f'Estimated bitrate from strings: {estimated_bpp:.3f} bpp\n')
        
#         print(f'📄 Detailed strings analysis saved to: {analysis_file}')

# def decompress_images(net, device, args):
#     """Load compressed data and decompress images"""
#     print("=== DECOMPRESSION MODE ===")
#     net.update()
#     if not os.path.exists(args.compressed_path):
#         print(f"Error: Compressed data path {args.compressed_path} does not exist!")
#         return
    
#     # Find all compressed files
#     compressed_files = [f for f in os.listdir(args.compressed_path) if f.endswith('.pkl')]
    
#     if not compressed_files:
#         print(f"Error: No compressed files found in {args.compressed_path}")
#         return
    
#     if args.save_path:
#         os.makedirs(args.save_path, exist_ok=True)
    
#     total_time = 0
#     count = 0
    
#     for file_idx, compressed_file in enumerate(compressed_files):
#         print(f"Decompressing: {compressed_file}")
        
#         # Load compressed data
#         compressed_data = load_compressed_data(os.path.join(args.compressed_path, compressed_file))
        
#         with torch.no_grad():
#             if args.cuda:
#                 torch.cuda.synchronize()
#             s = time.time()
#             out_dec = net.decompress(compressed_data["strings"], compressed_data["shape"])
#             if args.cuda:
#                 torch.cuda.synchronize()
#             e = time.time()
            
#             decode_time = (e - s)
            
#             # Only count time from second file onwards
#             if file_idx > 0:
#                 total_time += decode_time
#                 count += 1
            
#             # Crop the padded image
#             out_dec["x_hat"] = crop_batch(out_dec["x_hat"], compressed_data["padding"])
#             out_dec["x_hat"].clamp_(0, 1)
            
#             # Save decompressed image
#             if args.save_path:
#                 img_name = compressed_file.replace('_compressed.pkl', '')
#                 save_image(out_dec["x_hat"], os.path.join(args.save_path, f"{img_name}_decompressed.png"))
                
#                 # Save decompression stats
#                 with open(os.path.join(args.save_path, f"{img_name}_decompression_stats.txt"), 'w') as f:
#                     f.write(f'Image: {img_name}\n')
#                     f.write(f'Decompression time: {decode_time*1000:.3f} ms\n')
#                     f.write(f'Output shape: {out_dec["x_hat"].shape}\n')
        
#         timing_note = " (excluded from timing)" if file_idx == 0 else ""
#         print(f'Decompression time: {decode_time*1000:.3f} ms{timing_note}')
    
#     if count > 0:
#         avg_time = (total_time / count) * 1000
#         print(f'\nDecompression Summary (excluding first file):')
#         print(f'Files used for timing: {count}')
#         print(f'Average decompression time per image: {avg_time:.3f} ms')
#     else:
#         print(f'\nDecompression Summary: Only first file processed, no timing data available')
#     if args.save_path:
#         print(f'Decompressed images saved to: {args.save_path}')


# def main(argv):
#     torch.backends.cudnn.enabled = False
#     args = parse_args(argv)
    
#     if args.cuda:
#         device = 'cuda'
#     else:
#         device = 'cpu'
        
#     # For size_analysis mode, we don't need to load the model
#     if args.mode == "size_analysis":
#         analyze_data_size(device, args)
#         return
    
#     net = DCAE()
#     net = net.to(device)

#     param_size = 0
#     buffer_size = 0

#     for param in net.parameters():
#         param_size += param.nelement() * param.element_size()

#     for buffer in net.buffers():
#         buffer_size += buffer.nelement() * buffer.element_size()

#     size_all_mb = (param_size + buffer_size) / 1024**2
    
    
#     net.eval()
    
#     dictory = {}
    
#     if args.checkpoint:  
#         print("Loading", args.checkpoint)
#         checkpoint = torch.load(args.checkpoint, map_location=device)
#         for k, v in checkpoint["state_dict"].items():
#             print(f'keys: {k}')
#             dictory[k.replace("module.", "")] = v
#         net.load_state_dict(dictory)
    
#     if args.mode == "compress":
#         # Get image list for compression
#         path = args.data
#         img_list = []
#         for file in os.listdir(path):
#             if file[-3:] in ["jpg", "png", "peg"]:
#                 img_list.append(os.path.join(path, file))
        
#         if not img_list:
#             print(f"No images found in {path}")
#             return
            
#         compress_images(net, img_list, device, args)
#         print(f'Model size: {size_all_mb:.2f} MB')
        
#     elif args.mode == "decompress":
#         decompress_images(net, device, args)
#         print(f'Model size: {size_all_mb:.2f} MB')
#         print(f'Model size: {size_all_mb:.2f} MB')
        
        
#     # For the "both" mode, modify the relevant section:
#     elif args.mode == "both":
#         # ... (earlier code remains the same until the batch processing loop)
#         img_list = []
#         for file in os.listdir(path):
#             if file[-3:] in ["jpg", "png", "peg"]:
#                 img_list.append(os.path.join(path, file))
        
#         if not img_list:
#             print(f"No images found in {path}")
#             return
        
#         batch_size = 1
#         num_batches = (len(img_list) + batch_size - 1) // batch_size
#         if args.real:
#             net.update()
#             for batch_idx in range(num_batches):
#                 start_idx = batch_idx * batch_size
#                 end_idx = min((batch_idx + 1) * batch_size, len(img_list))
#                 batch_img_paths = img_list[start_idx:end_idx]
#                 current_batch_size = len(batch_img_paths)
                
#                 print(f"Processing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
                
#                 x_batch = load_images_batch(batch_img_paths, device)
#                 x_batch_padded, padding = pad_batch(x_batch, p)
                
#                 # Only count images from batch 2 onwards for timing
#                 if batch_idx > 0:
#                     count += current_batch_size
                
#                 with torch.no_grad():
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     s = time.time()
#                     out_enc = net.compress(x_batch_padded)
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     e = time.time()
#                     batch_encode_time = (e - s)
                    
#                     # Only add timing from batch 2 onwards
#                     if batch_idx > 0:
#                         encode_time += batch_encode_time
#                         total_time += batch_encode_time
                    
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     s = time.time()
#                     out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
#                     if args.cuda:
#                         torch.cuda.synchronize()
#                     e = time.time()
#                     batch_decode_time = (e - s)
                    
#                     # Only add timing from batch 2 onwards
#                     if batch_idx > 0:
#                         decode_time += batch_decode_time
#                         total_time += batch_decode_time
                    
#                     out_dec["x_hat"] = crop_batch(out_dec["x_hat"], padding)
                    
#                     # Calculate metrics for all batches (not just timing)
#                     for i in range(current_batch_size):
#                         x_single = x_batch[i:i+1]
#                         x_hat_single = out_dec["x_hat"][i:i+1]
#                         num_pixels = x_single.size(2) * x_single.size(3)
                        
#                         bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / (num_pixels * current_batch_size)
                        
#                         # Count metrics for all images
#                         total_Bit_rate += bit_rate
#                         total_PSNR += compute_psnr(x_single, x_hat_single).item()
#                         total_MS_SSIM += compute_msssim(x_single, x_hat_single).item()
                    
#                     timing_note = " (excluded from timing)" if batch_idx == 0 else ""
#                     print(f'Batch {batch_idx + 1} - Encoding time: {batch_encode_time*1000:.3f} ms, Decoding time: {batch_decode_time*1000:.3f} ms{timing_note}')
        
#         # Calculate averages
#         total_images = len(img_list)
#         avg_PSNR = total_PSNR / total_images  # Use all images for quality metrics
#         avg_MS_SSIM = total_MS_SSIM / total_images
#         avg_Bit_rate = total_Bit_rate / total_images
        
#         # Use only counted images for timing averages
#         if count > 0:
#             avg_total_time = (total_time / count) * 1000
#             avg_encode_time = (encode_time / count) * 1000
#             avg_decode_time = (decode_time / count) * 1000
            
#             print(f'Total images processed: {total_images}')
#             print(f'Images used for timing: {count}')
#             print(f'Average PSNR: {avg_PSNR:.2f} dB')
#             print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
#             print(f'Average Bit-rate: {avg_Bit_rate:.3f} bpp')
#             print(f'Average total time per image (excluding first batch): {avg_total_time:.3f} ms')
#             print(f'Average encode time per image (excluding first batch): {avg_encode_time:.3f} ms')
#             print(f'Average decode time per image (excluding first batch): {avg_decode_time:.3f} ms')
#         else:
#             print(f'Total images processed: {total_images}')
#             print(f'Average PSNR: {avg_PSNR:.2f} dB')
#             print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
#             print(f'Average Bit-rate: {avg_Bit_rate:.3f} bpp')
#             print('No timing data available (only first batch processed)')
        
#         print(f'Model size: {size_all_mb:.2f} MB')

# if __name__ == "__main__":
#     print(torch.cuda.is_available())
#     main(sys.argv[1:])


## load CDF from checkpoint
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# Force deterministic algorithms where possible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

import torch.nn.functional as F
from torchvision import transforms
from models import (
    DCAE    
)
import warnings
import torch
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

import sys
import math
import argparse
import time
import warnings
import pickle
from pytorch_msssim import ms_ssim
from PIL import Image
from thop import profile
warnings.filterwarnings("ignore")
torch.set_num_threads(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(42)
import numpy as np
np.random.seed(42)

print(torch.cuda.is_available())



# torch.set_default_dtype(torch.float64)

def save_image(tensor, filename):
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    img.save(filename)

def save_metrics(filename, psnr, bitrate, msssim):
    with open(filename, 'w') as f:
        f.write(f'PSNR: {psnr:.2f}dB\n')
        f.write(f'Bitrate: {bitrate:.3f}bpp\n')
        f.write(f'MS-SSIM: {msssim:.4f}\n')

def save_compressed_data(compressed_data, filename):
    """Save compressed data to a file"""
    with open(filename, 'wb') as f:
        pickle.dump(compressed_data, f)
    print(f"Compressed data saved to: {filename}")

def load_compressed_data(filename):
    """Load compressed data from a file"""
    with open(filename, 'rb') as f:
        compressed_data = pickle.load(f)
    print(f"Compressed data loaded from: {filename}")
    return compressed_data

def get_size_in_bytes(obj):
    """Calculate size of an object in bytes"""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    elif isinstance(obj, (list, tuple)):
        total_size = 0
        for item in obj:
            total_size += get_size_in_bytes(item)
        return total_size
    elif isinstance(obj, str):
        return len(obj.encode('utf-8'))
    elif isinstance(obj, bytes):
        return len(obj)
    elif isinstance(obj, dict):
        total_size = 0
        for key, value in obj.items():
            total_size += get_size_in_bytes(key) + get_size_in_bytes(value)
        return total_size
    else:
        return sys.getsizeof(obj)

def get_shape_info(obj, name="object"):
    """Get detailed shape information for various data types"""
    shape_info = {
        'type': type(obj).__name__,
        'shape': None,
        'detailed_shape': None,
        'description': None
    }
    
    if isinstance(obj, torch.Tensor):
        shape_info['shape'] = list(obj.shape)
        shape_info['detailed_shape'] = f"torch.Size({list(obj.shape)})"
        shape_info['description'] = f"Tensor shape: {list(obj.shape)}"
        if hasattr(obj, 'dtype'):
            shape_info['dtype'] = str(obj.dtype)
    
    elif isinstance(obj, (list, tuple)):
        shape_info['shape'] = [len(obj)]
        shape_info['detailed_shape'] = f"Length: {len(obj)}"
        
        # For nested structures, get more details
        if len(obj) > 0:
            first_elem = obj[0]
            if isinstance(first_elem, (list, tuple)):
                nested_lens = [len(item) if isinstance(item, (list, tuple)) else 1 for item in obj]
                shape_info['detailed_shape'] = f"Outer length: {len(obj)}, Inner lengths: {nested_lens[:5]}{'...' if len(nested_lens) > 5 else ''}"
                
                # Check if it's a list of byte strings
                if len(obj) > 0 and len(obj[0]) > 0 and isinstance(obj[0][0], bytes):
                    byte_lens = []
                    for sublist in obj:
                        if isinstance(sublist, list) and len(sublist) > 0:
                            byte_lens.extend([len(item) for item in sublist if isinstance(item, bytes)])
                    if byte_lens:
                        shape_info['description'] = f"List of {len(obj)} sublists, containing byte strings of lengths: {byte_lens[:10]}{'...' if len(byte_lens) > 10 else ''}"
                        shape_info['detailed_shape'] += f", Total byte strings: {len(byte_lens)}"
            else:
                elem_types = [type(item).__name__ for item in obj[:5]]
                shape_info['description'] = f"List of {len(obj)} elements, types: {elem_types}{'...' if len(obj) > 5 else ''}"
        else:
            shape_info['description'] = "Empty list"
    
    elif isinstance(obj, dict):
        shape_info['shape'] = [len(obj)]
        shape_info['detailed_shape'] = f"Dict with {len(obj)} keys: {list(obj.keys())}"
        shape_info['description'] = f"Dictionary with keys: {list(obj.keys())}"
    
    elif isinstance(obj, (str, bytes)):
        shape_info['shape'] = [len(obj)]
        shape_info['detailed_shape'] = f"Length: {len(obj)}"
        shape_info['description'] = f"{'String' if isinstance(obj, str) else 'Bytes'} of length {len(obj)}"
    
    elif isinstance(obj, (int, float)):
        shape_info['shape'] = []  # Scalar
        shape_info['detailed_shape'] = "Scalar"
        shape_info['description'] = f"Scalar {type(obj).__name__}: {obj}"
    
    else:
        # Try to get shape for other types
        if hasattr(obj, 'shape'):
            shape_info['shape'] = list(obj.shape) if hasattr(obj.shape, '__iter__') else [obj.shape]
            shape_info['detailed_shape'] = f"Shape: {obj.shape}"
        elif hasattr(obj, '__len__'):
            try:
                shape_info['shape'] = [len(obj)]
                shape_info['detailed_shape'] = f"Length: {len(obj)}"
            except:
                shape_info['detailed_shape'] = "Unknown shape"
        else:
            shape_info['detailed_shape'] = "No shape information"
        
        shape_info['description'] = f"Object of type {type(obj).__name__}"
    
    return shape_info

def analyze_data_size(device, args):
    """Analyze and display sizes, shapes, and types of compressed data and original images"""
    print("=== SIZE ANALYSIS MODE ===")
    
    if not os.path.exists(args.compressed_path):
        print(f"Error: Compressed data path {args.compressed_path} does not exist!")
        return
    
    if not os.path.exists(args.data):
        print(f"Error: Original images path {args.data} does not exist!")
        return
    
    # Find compressed files
    compressed_files = [f for f in os.listdir(args.compressed_path) if f.endswith('_compressed.pkl')]
    
    if not compressed_files:
        print(f"Error: No compressed files found in {args.compressed_path}")
        return
    
    # Get original image files
    img_list = []
    for file in os.listdir(args.data):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(os.path.join(args.data, file))
    
    print(f"\nFound {len(compressed_files)} compressed files and {len(img_list)} original images")
    print("=" * 100)
    
    total_strings_size = 0
    total_shape_size = 0
    total_original_size = 0
    total_compressed_file_size = 0
    
    analysis_results = []
    
    for i, compressed_file in enumerate(compressed_files):
        img_name = compressed_file.replace('_compressed.pkl', '')
        compressed_path = os.path.join(args.compressed_path, compressed_file)
        
        # Find corresponding original image
        original_img_path = None
        for img_path in img_list:
            if img_name in os.path.basename(img_path):
                original_img_path = img_path
                break
        
        if original_img_path is None:
            print(f"Warning: No original image found for {img_name}")
            continue
        
        # Load compressed data
        compressed_data = load_compressed_data(compressed_path)
        
        # Load original image
        original_img = transforms.ToTensor()(Image.open(original_img_path).convert('RGB'))
        
        # Calculate sizes
        strings_size = get_size_in_bytes(compressed_data["strings"])
        shape_size = get_size_in_bytes(compressed_data["shape"])
        original_size = get_size_in_bytes(original_img)
        compressed_file_size = os.path.getsize(compressed_path)
        
        # Get detailed shape information
        strings_shape_info = get_shape_info(compressed_data["strings"], "strings")
        shape_shape_info = get_shape_info(compressed_data["shape"], "shape")
        original_shape_info = get_shape_info(original_img, "original")
        
        # Store results
        result = {
            'img_name': img_name,
            'strings_size': strings_size,
            'shape_size': shape_size,
            'original_size': original_size,
            'compressed_file_size': compressed_file_size,
            'strings_shape_info': strings_shape_info,
            'shape_shape_info': shape_shape_info,
            'original_shape_info': original_shape_info
        }
        analysis_results.append(result)
        
        # Update totals
        total_strings_size += strings_size
        total_shape_size += shape_size
        total_original_size += original_size
        total_compressed_file_size += compressed_file_size
        
        # Print individual results
        print(f"\n--- Image {i+1}: {img_name} ---")
        
        # Original Image Analysis
        print(f"📷 Original Image:")
        print(f"  • Type: {original_shape_info['type']}")
        print(f"  • Shape: {original_shape_info['shape']}")
        print(f"  • Detailed Shape: {original_shape_info['detailed_shape']}")
        if 'dtype' in original_shape_info:
            print(f"  • Data Type: {original_shape_info['dtype']}")
        print(f"  • Size: {original_size:,} bytes ({original_size/1024:.2f} KB)")
        print(f"  • Description: {original_shape_info['description']}")
        
        # Compressed Strings Analysis
        print(f"\n🗜️ Compressed Strings:")
        print(f"  • Type: {strings_shape_info['type']}")
        print(f"  • Shape: {strings_shape_info['shape']}")
        print(f"  • Detailed Shape: {strings_shape_info['detailed_shape']}")
        print(f"  • Size: {strings_size:,} bytes ({strings_size/1024:.2f} KB)")
        print(f"  • Description: {strings_shape_info['description']}")
        
        # Compressed Shape Analysis  
        print(f"\n📐 Compressed Shape:")
        print(f"  • Type: {shape_shape_info['type']}")
        print(f"  • Shape: {shape_shape_info['shape']}")
        print(f"  • Detailed Shape: {shape_shape_info['detailed_shape']}")
        print(f"  • Size: {shape_size:,} bytes")
        print(f"  • Description: {shape_shape_info['description']}")
        if hasattr(compressed_data["shape"], '__iter__'):
            try:
                print(f"  • Value: {list(compressed_data['shape']) if hasattr(compressed_data['shape'], '__iter__') else compressed_data['shape']}")
            except:
                print(f"  • Value: {compressed_data['shape']}")
        
        # File size and compression info
        print(f"\n💾 File Information:")
        print(f"  • Total compressed file size: {compressed_file_size:,} bytes ({compressed_file_size/1024:.2f} KB)")
        
        # Compression ratio
        compression_ratio = original_size / compressed_file_size if compressed_file_size > 0 else float('inf')
        print(f"  • Compression ratio: {compression_ratio:.2f}:1")
        
        # Component size breakdown
        strings_percentage = (strings_size / compressed_file_size * 100) if compressed_file_size > 0 else 0
        shape_percentage = (shape_size / compressed_file_size * 100) if compressed_file_size > 0 else 0
        print(f"  • Strings component: {strings_percentage:.1f}% of total")
        print(f"  • Shape component: {shape_percentage:.1f}% of total")
        
        print("-" * 60)
    
    # Print summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    print(f"Total files analyzed: {len(analysis_results)}")
    
    print(f"\n📊 Total sizes:")
    print(f"  • Original images: {total_original_size:,} bytes ({total_original_size/1024/1024:.2f} MB)")
    print(f"  • Compressed strings: {total_strings_size:,} bytes ({total_strings_size/1024:.2f} KB)")
    print(f"  • Compressed shapes: {total_shape_size:,} bytes ({total_shape_size/1024:.2f} KB)")
    print(f"  • Total compressed files: {total_compressed_file_size:,} bytes ({total_compressed_file_size/1024:.2f} KB)")
    
    # Average sizes and shapes
    num_files = len(analysis_results)
    if num_files > 0:
        print(f"\n📈 Average per image:")
        print(f"  • Original size: {total_original_size/num_files:,.0f} bytes")
        print(f"  • Compressed strings size: {total_strings_size/num_files:,.0f} bytes")
        print(f"  • Compressed shapes size: {total_shape_size/num_files:,.0f} bytes")
        print(f"  • Total compressed size: {total_compressed_file_size/num_files:,.0f} bytes")
        
        overall_compression_ratio = total_original_size / total_compressed_file_size if total_compressed_file_size > 0 else float('inf')
        print(f"  • Overall compression ratio: {overall_compression_ratio:.2f}:1")
        
        # Average component percentages
        avg_strings_percentage = (total_strings_size / total_compressed_file_size * 100) if total_compressed_file_size > 0 else 0
        avg_shape_percentage = (total_shape_size / total_compressed_file_size * 100) if total_compressed_file_size > 0 else 0
        print(f"  • Average strings component: {avg_strings_percentage:.1f}% of compressed size")
        print(f"  • Average shape component: {avg_shape_percentage:.1f}% of compressed size")
    
    # Save detailed analysis to file if save_path is provided
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        analysis_file = os.path.join(args.save_path, "detailed_size_analysis_report.txt")
        
        with open(analysis_file, 'w') as f:
            f.write("=== DETAILED SIZE AND SHAPE ANALYSIS REPORT ===\n\n")
            
            for result in analysis_results:
                f.write(f"Image: {result['img_name']}\n")
                f.write("-" * 40 + "\n")
                
                # Original image info
                orig = result['original_shape_info']
                f.write(f"Original Image:\n")
                f.write(f"  Type: {orig['type']}\n")
                f.write(f"  Shape: {orig['shape']}\n")
                f.write(f"  Detailed Shape: {orig['detailed_shape']}\n")
                f.write(f"  Size: {result['original_size']:,} bytes\n")
                f.write(f"  Description: {orig['description']}\n\n")
                
                # Compressed strings info
                strings = result['strings_shape_info']
                f.write(f"Compressed Strings:\n")
                f.write(f"  Type: {strings['type']}\n")
                f.write(f"  Shape: {strings['shape']}\n")
                f.write(f"  Detailed Shape: {strings['detailed_shape']}\n")
                f.write(f"  Size: {result['strings_size']:,} bytes\n")
                f.write(f"  Description: {strings['description']}\n\n")
                
                # Compressed shape info
                shape = result['shape_shape_info']
                f.write(f"Compressed Shape:\n")
                f.write(f"  Type: {shape['type']}\n")
                f.write(f"  Shape: {shape['shape']}\n")
                f.write(f"  Detailed Shape: {shape['detailed_shape']}\n")
                f.write(f"  Size: {result['shape_size']:,} bytes\n")
                f.write(f"  Description: {shape['description']}\n\n")
                
                f.write(f"Total compressed file size: {result['compressed_file_size']:,} bytes\n")
                f.write(f"Compression ratio: {result['original_size']/result['compressed_file_size']:.2f}:1\n")
                f.write("-" * 60 + "\n\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"Total files: {len(analysis_results)}\n")
            f.write(f"Total original size: {total_original_size:,} bytes\n")
            f.write(f"Total compressed size: {total_compressed_file_size:,} bytes\n")
            f.write(f"Overall compression ratio: {total_original_size/total_compressed_file_size:.2f}:1\n")
        
        print(f"\n📄 Detailed analysis saved to: {analysis_file}")

# ... (rest of the functions remain the same as in the previous version)

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2, dim=[1,2,3])
    return -10 * torch.log10(mse)

def compute_msssim(a, b):
    msssim_values = []
    for i in range(a.size(0)):
        msssim_val = ms_ssim(a[i:i+1], b[i:i+1], data_range=1.).item()
        msssim_values.append(-10 * math.log10(1 - msssim_val))
    return torch.tensor(msssim_values)

def compute_bpp_batch(out_net):
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
        # img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(torch.float64)
        images.append(img)
    return torch.stack(images).to(device)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Image compression/decompression script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default="checkpoint_7_gpu_0dot003.pth")
    parser.add_argument("--data", type=str, help="Path to dataset", default='/home/xie/datasets/dummy/valid')
    parser.add_argument("--save_path", default='eval', type=str, help="Path to save results")
    parser.add_argument("--compressed_path", default="eval/compressed_gpu", type=str, help="Path to save/load compressed data")
    parser.add_argument(
        "--mode", 
        choices=["compress", "decompress", "both", "size_analysis"], 
        default="both",
        help="Choose operation mode: compress, decompress, both, or size_analysis (default: both)"
    )
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args

def compress_images(net, img_list, device, args, p=128, batch_size=1):
    """Compress images and save compressed data"""
    print("=== COMPRESSION MODE ===")
    
    # Create compressed data directory
    os.makedirs(args.compressed_path, exist_ok=True)
    
    # net.update()
    num_batches = (len(img_list) + batch_size - 1) // batch_size
    
    total_time = 0
    count = 0
    
    # Binary strings size tracking
    total_strings_size = 0
    total_images_processed = 0
    strings_size_per_batch = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(img_list))
        batch_img_paths = img_list[start_idx:end_idx]
        current_batch_size = len(batch_img_paths)
        
        print(f"Compressing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
        
        # Load batch of images
        x_batch = load_images_batch(batch_img_paths, device)
        x_batch_padded, padding = pad_batch(x_batch, p)
        
        # Only count images from batch 2 onwards for timing
        if batch_idx > 0:
            count += current_batch_size
        
        # Always count all images for data size analysis
        total_images_processed += current_batch_size
        
        with torch.no_grad():
            if args.cuda:
                torch.cuda.synchronize()
            s = time.time()
            out_enc = net.compress(x_batch_padded)
            if args.cuda:
                torch.cuda.synchronize()
            e = time.time()
            
            batch_time = (e - s)
            
            # Only add time from batch 2 onwards
            if batch_idx > 0:
                total_time += batch_time
            
            # Calculate binary strings size for this batch
            batch_strings_size = 0
            for string_list in out_enc["strings"]:
                for binary_string in string_list:
                    if isinstance(binary_string, bytes):
                        batch_strings_size += len(binary_string)
                    elif isinstance(binary_string, str):
                        batch_strings_size += len(binary_string.encode('utf-8'))
                    else:
                        # Handle other types (e.g., if it's a tensor or list)
                        batch_strings_size += get_size_in_bytes(binary_string)
            
            total_strings_size += batch_strings_size
            strings_size_per_batch.append({
                'batch_idx': batch_idx + 1,
                'size_bytes': batch_strings_size,
                'size_kb': batch_strings_size / 1024,
                'avg_per_image': batch_strings_size / current_batch_size,
                'num_images': current_batch_size
            })
            
            # Save compressed data for each image (regardless of batch)
            for i in range(current_batch_size):
                img_name = os.path.basename(batch_img_paths[i]).split('.')[0]
                compressed_filename = os.path.join(args.compressed_path, f"{img_name}_compressed.pkl")
                
                # Extract data for single image
                single_compressed = {
                    "strings": [[s[0]] for s in out_enc["strings"]],
                    "shape": out_enc["shape"],
                    "padding": padding,
                    "original_shape": (x_batch[i].size(1), x_batch[i].size(2))
                }
                
                save_compressed_data(single_compressed, compressed_filename)
                
                # Calculate and save compression stats including strings size
                num_pixels = x_batch[i].size(1) * x_batch[i].size(2)
                bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                
                # Calculate individual image strings size
                img_strings_size = sum(len(s[0]) for s in out_enc["strings"])
                
                if args.save_path:
                    os.makedirs(args.save_path, exist_ok=True)
                    with open(os.path.join(args.save_path, f"{img_name}_compression_stats.txt"), 'w') as f:
                        f.write(f'Image: {img_name}\n')
                        f.write(f'Bitrate: {bit_rate:.3f} bpp\n')
                        f.write(f'Strings size: {img_strings_size} bytes ({img_strings_size/1024:.2f} KB)\n')
                        f.write(f'Compression time: {batch_time/current_batch_size*1000:.3f} ms\n')
        
        timing_note = " (excluded from timing)" if batch_idx == 0 else ""
        print(f'Batch {batch_idx + 1} - Compression time: {batch_time*1000:.3f} ms, Strings size: {batch_strings_size/1024:.2f} KB{timing_note}')
    
    # Print detailed binary strings size analysis
    print(f'\n{"="*60}')
    print(f'BINARY STRINGS SIZE ANALYSIS')
    print(f'{"="*60}')
    
    print(f'\nPer-batch breakdown:')
    print(f'{"Batch":<8} {"Images":<8} {"Size (KB)":<12} {"Avg/Image (KB)":<15} {"Size (bytes)":<12}')
    print(f'{"-"*60}')
    
    for batch_info in strings_size_per_batch:
        print(f'{batch_info["batch_idx"]:<8} {batch_info["num_images"]:<8} '
              f'{batch_info["size_kb"]:<12.2f} {batch_info["avg_per_image"]/1024:<15.2f} '
              f'{batch_info["size_bytes"]:<12}')
    
    # Overall summary
    avg_strings_size_per_image = total_strings_size / total_images_processed if total_images_processed > 0 else 0
    
    print(f'\n{"="*40}')
    print(f'OVERALL STRINGS SIZE SUMMARY')
    print(f'{"="*40}')
    print(f'Total images processed: {total_images_processed}')
    print(f'Total strings size: {total_strings_size:,} bytes ({total_strings_size/1024:.2f} KB, {total_strings_size/1024/1024:.2f} MB)')
    print(f'Average strings size per image: {avg_strings_size_per_image:.0f} bytes ({avg_strings_size_per_image/1024:.2f} KB)')
    
    if total_images_processed > 0:
        # Calculate additional statistics
        min_batch_size = min(batch_info["avg_per_image"] for batch_info in strings_size_per_batch)
        max_batch_size = max(batch_info["avg_per_image"] for batch_info in strings_size_per_batch)
        
        print(f'Min average per image: {min_batch_size:.0f} bytes ({min_batch_size/1024:.2f} KB)')
        print(f'Max average per image: {max_batch_size:.0f} bytes ({max_batch_size/1024:.2f} KB)')
        
        # Calculate estimated bitrate from strings size
        if len(img_list) > 0:
            # Estimate average image dimensions (this is approximate)
            sample_img = transforms.ToTensor()(Image.open(img_list[0]).convert('RGB'))
            avg_pixels = sample_img.size(1) * sample_img.size(2)  # H * W
            estimated_bpp = (avg_strings_size_per_image * 8) / avg_pixels
            print(f'Estimated bitrate from strings: {estimated_bpp:.3f} bpp (based on sample image dimensions)')
    
    # Timing summary
    print(f'\n{"="*40}')
    print(f'TIMING SUMMARY')
    print(f'{"="*40}')
    
    if count > 0:
        avg_time = (total_time / count) * 1000
        print(f'Images used for timing: {count} (excluding first batch)')
        print(f'Average compression time per image: {avg_time:.3f} ms')
    else:
        print(f'Only first batch processed, no timing data available')
    
    print(f'Compressed data saved to: {args.compressed_path}')
    
    # Save detailed strings analysis to file if save_path is provided
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        analysis_file = os.path.join(args.save_path, "strings_size_analysis.txt")
        
        with open(analysis_file, 'w') as f:
            f.write("BINARY STRINGS SIZE ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("Per-batch breakdown:\n")
            f.write(f'{"Batch":<8} {"Images":<8} {"Size (KB)":<12} {"Avg/Image (KB)":<15} {"Size (bytes)":<12}\n')
            f.write("-"*60 + "\n")
            
            for batch_info in strings_size_per_batch:
                f.write(f'{batch_info["batch_idx"]:<8} {batch_info["num_images"]:<8} '
                       f'{batch_info["size_kb"]:<12.2f} {batch_info["avg_per_image"]/1024:<15.2f} '
                       f'{batch_info["size_bytes"]:<12}\n')
            
            f.write(f'\nOVERALL SUMMARY:\n')
            f.write(f'Total images processed: {total_images_processed}\n')
            f.write(f'Total strings size: {total_strings_size:,} bytes ({total_strings_size/1024:.2f} KB)\n')
            f.write(f'Average strings size per image: {avg_strings_size_per_image:.0f} bytes ({avg_strings_size_per_image/1024:.2f} KB)\n')
            
            if total_images_processed > 0:
                f.write(f'Min average per image: {min_batch_size:.0f} bytes ({min_batch_size/1024:.2f} KB)\n')
                f.write(f'Max average per image: {max_batch_size:.0f} bytes ({max_batch_size/1024:.2f} KB)\n')
                if len(img_list) > 0:
                    f.write(f'Estimated bitrate from strings: {estimated_bpp:.3f} bpp\n')
        
        print(f'📄 Detailed strings analysis saved to: {analysis_file}')

def decompress_images(net, device, args):
    """Load compressed data and decompress images"""
    print("=== DECOMPRESSION MODE ===")
    # net.update()
    if not os.path.exists(args.compressed_path):
        print(f"Error: Compressed data path {args.compressed_path} does not exist!")
        return
    os.makedirs(os.path.join(args.save_path, 'decompressed'), exist_ok=True)
    # Find all compressed files
    compressed_files = [f for f in os.listdir(args.compressed_path) if f.endswith('.pkl')]
    
    if not compressed_files:
        print(f"Error: No compressed files found in {args.compressed_path}")
        return
    
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
    
    total_time = 0
    count = 0
    
    for file_idx, compressed_file in enumerate(compressed_files):
        print(f"Decompressing: {compressed_file}")
        
        # Load compressed data
        compressed_data = load_compressed_data(os.path.join(args.compressed_path, compressed_file))
        
        with torch.no_grad():
            if args.cuda:
                torch.cuda.synchronize()
            s = time.time()
            out_dec = net.decompress(compressed_data["strings"], compressed_data["shape"])
            if args.cuda:
                torch.cuda.synchronize()
            e = time.time()
            
            decode_time = (e - s)
            
            # Only count time from second file onwards
            if file_idx > 0:
                total_time += decode_time
                count += 1
            
            # Crop the padded image
            out_dec["x_hat"] = crop_batch(out_dec["x_hat"], compressed_data["padding"])
            out_dec["x_hat"].clamp_(0, 1)
            
            # Save decompressed image
            if args.save_path:
                img_name = compressed_file.replace('_compressed.pkl', '')
                save_image(out_dec["x_hat"], os.path.join(args.save_path, 'decompressed', f"{img_name}_decompressed.png"))
                
                # Save decompression stats
                # with open(os.path.join(args.save_path, f"{img_name}_decompression_stats.txt"), 'w') as f:
                #     f.write(f'Image: {img_name}\n')
                #     f.write(f'Decompression time: {decode_time*1000:.3f} ms\n')
                #     f.write(f'Output shape: {out_dec["x_hat"].shape}\n')
        
        timing_note = " (excluded from timing)" if file_idx == 0 else ""
        print(f'Decompression time: {decode_time*1000:.3f} ms{timing_note}')
    
    if count > 0:
        avg_time = (total_time / count) * 1000
        print(f'\nDecompression Summary (excluding first file):')
        print(f'Files used for timing: {count}')
        print(f'Average decompression time per image: {avg_time:.3f} ms')
    else:
        print(f'\nDecompression Summary: Only first file processed, no timing data available')
    if args.save_path:
        print(f'Decompressed images saved to: {args.save_path}')


def main(argv):
    torch.backends.cudnn.enabled = False
    args = parse_args(argv)
    
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
        
    # For size_analysis mode, we don't need to load the model
    if args.mode == "size_analysis":
        analyze_data_size(device, args)
        return
    
    net = DCAE()
    net = net.to(device)
    # net = net.to(device).to(torch.float64) 

    param_size = 0
    buffer_size = 0

    for param in net.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    
    
    net.eval()
    
    dictory = {}
    
    if args.checkpoint:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            print(f'keys: {k}')
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
    
    device = 'cuda' if args.cuda else 'cpu'
    # net.g_a.to(device)
    # net.g_s.to(device)
    # net.h_a.to(device)
    # net.entropy_bottleneck.to('cpu')
    # net.h_z_s1.to('cpu')
    # net.h_z_s2.to('cpu')
    # net.gaussian_conditional.to('cpu')
    # for i in range(net.num_slices):
    #     net.dt_cross_attention[i].to('cpu')
    #     net.cc_mean_transforms[i].to('cpu')
    #     net.cc_scale_transforms[i].to('cpu')
    #     net.lrp_transforms[i].to('cpu')
    # net.dt.data = net.dt.data.to('cpu')

    if args.mode == "compress":
        # Get image list for compression
        path = args.data
        img_list = []
        for file in os.listdir(path):
            if file[-3:] in ["jpg", "png", "peg"]:
                img_list.append(os.path.join(path, file))
        
        if not img_list:
            print(f"No images found in {path}")
            return
            
        compress_images(net, img_list, device, args)
        print(f'Model size: {size_all_mb:.2f} MB')
        
    elif args.mode == "decompress":
        decompress_images(net, device, args)
        print(f'Model size: {size_all_mb:.2f} MB')
        print(f'Model size: {size_all_mb:.2f} MB')
        
        
    # For the "both" mode, modify the relevant section:
    elif args.mode == "both":
        # ... (earlier code remains the same until the batch processing loop)
        img_list = []
        for file in os.listdir(path):
            if file[-3:] in ["jpg", "png", "peg"]:
                img_list.append(os.path.join(path, file))
        
        if not img_list:
            print(f"No images found in {path}")
            return
        
        batch_size = 1
        num_batches = (len(img_list) + batch_size - 1) // batch_size
        if args.real:
            # net.update()
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(img_list))
                batch_img_paths = img_list[start_idx:end_idx]
                current_batch_size = len(batch_img_paths)
                
                print(f"Processing batch {batch_idx + 1}/{num_batches} with {current_batch_size} images")
                
                x_batch = load_images_batch(batch_img_paths, device)
                x_batch_padded, padding = pad_batch(x_batch, p)
                
                # Only count images from batch 2 onwards for timing
                if batch_idx > 0:
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
                    
                    # Only add timing from batch 2 onwards
                    if batch_idx > 0:
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
                    
                    # Only add timing from batch 2 onwards
                    if batch_idx > 0:
                        decode_time += batch_decode_time
                        total_time += batch_decode_time
                    
                    out_dec["x_hat"] = crop_batch(out_dec["x_hat"], padding)
                    
                    # Calculate metrics for all batches (not just timing)
                    for i in range(current_batch_size):
                        x_single = x_batch[i:i+1]
                        x_hat_single = out_dec["x_hat"][i:i+1]
                        num_pixels = x_single.size(2) * x_single.size(3)
                        
                        bit_rate = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / (num_pixels * current_batch_size)
                        
                        # Count metrics for all images
                        total_Bit_rate += bit_rate
                        total_PSNR += compute_psnr(x_single, x_hat_single).item()
                        total_MS_SSIM += compute_msssim(x_single, x_hat_single).item()
                    
                    timing_note = " (excluded from timing)" if batch_idx == 0 else ""
                    print(f'Batch {batch_idx + 1} - Encoding time: {batch_encode_time*1000:.3f} ms, Decoding time: {batch_decode_time*1000:.3f} ms{timing_note}')
        
        # Calculate averages
        total_images = len(img_list)
        avg_PSNR = total_PSNR / total_images  # Use all images for quality metrics
        avg_MS_SSIM = total_MS_SSIM / total_images
        avg_Bit_rate = total_Bit_rate / total_images
        
        # Use only counted images for timing averages
        if count > 0:
            avg_total_time = (total_time / count) * 1000
            avg_encode_time = (encode_time / count) * 1000
            avg_decode_time = (decode_time / count) * 1000
            
            print(f'Total images processed: {total_images}')
            print(f'Images used for timing: {count}')
            print(f'Average PSNR: {avg_PSNR:.2f} dB')
            print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
            print(f'Average Bit-rate: {avg_Bit_rate:.3f} bpp')
            print(f'Average total time per image (excluding first batch): {avg_total_time:.3f} ms')
            print(f'Average encode time per image (excluding first batch): {avg_encode_time:.3f} ms')
            print(f'Average decode time per image (excluding first batch): {avg_decode_time:.3f} ms')
        else:
            print(f'Total images processed: {total_images}')
            print(f'Average PSNR: {avg_PSNR:.2f} dB')
            print(f'Average MS-SSIM: {avg_MS_SSIM:.4f}')
            print(f'Average Bit-rate: {avg_Bit_rate:.3f} bpp')
            print('No timing data available (only first batch processed)')
        
        print(f'Model size: {size_all_mb:.2f} MB')

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])


