import torch
import torch.nn.functional as F
from torchvision import transforms
from models import (
    DCAE_6, get_scale_table    
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
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

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

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda", default=True)
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default='60.5checkpoint_best.pth.tar')
    parser.add_argument("--data", type=str, help="Path to dataset", default='dataset/test')
    parser.add_argument("--save_path", default='evaluation_results', type=str, help="Path to save")
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
    if path is None:
        raise RuntimeError("Please provide --data path to images")
    img_list = sorted([f for f in os.listdir(path) if f.lower().endswith(("jpg", "png", "jpeg"))])
    if len(img_list) == 0:
        raise RuntimeError(f"No images found in {path} (extensions: jpg, png, jpeg)")
    else:
        print(f'found {len(img_list)} images in dataset')

    # === Step 1. Build CPU model for compression ===
    net_cpu = DCAE_6().to("cpu").eval()
    dictory = {}

    if args.checkpoint:
        print("Loading checkpoint", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net_cpu.load_state_dict(dictory)

    # ensure entropy tables are updated on CPU model
    net_cpu.update(get_scale_table(), force=True)

    # === Step 2. Build GPU model for decompression ===
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for GPU decompression test")
    device_gpu = "cuda:0"
    net_gpu = DCAE_6().to(device_gpu).eval()
    # load same weights onto GPU model
    net_gpu.load_state_dict(dictory)
    # update entropy tables on GPU model too
    net_gpu.update(get_scale_table(), force=True)

    # === Metrics ===
    count = 0
    PSNR = 0.0
    Bit_rate = 0.0
    MS_SSIM = 0.0
    encode_time = 0.0
    decode_time = 0.0
    total_time = 0.0

    for img_name in img_list:
        img_path = os.path.join(path, img_name)
        # keep image on CPU for compressor
        img = transforms.ToTensor()(Image.open(img_path).convert("RGB")).unsqueeze(0).to("cpu")
        x_padded, padding = pad(img, p)
        x_padded = x_padded.to("cpu")  # be explicit

        count += 1
        with torch.no_grad():
            # --- CPU compress ---
            t0 = time.time()
            out_enc = net_cpu.compress(x_padded)
            t1 = time.time()
            encode_time += (t1 - t0)
            total_time += (t1 - t0)
            cpu_tables = out_enc["tables"]
            # inspect first few values
            print("cdf[0][:10]:", cpu_tables["cdf"][0][:10])
            print("cdf_lengths[:5]:", cpu_tables["cdf_lengths"][:5])
            print("offsets[:20]:", cpu_tables["offsets"][:20])


            # --- GPU decompress ---
            # optionally move tables into strings if your compress returned them separately;
            # here we assume compress() returns strings and shape only (as before).
            try:
                torch.cuda.synchronize()
                t0 = time.time()
                out_dec = net_gpu.decompress(out_enc["strings"], out_enc["shape"], out_enc["tables"], out_enc["indexes"])
                torch.cuda.synchronize()
                t1 = time.time()
            except Exception as exc:
                print(f"Decoding failed for {img_name}: {exc}")
                raise

            decode_time += (t1 - t0)
            total_time += (t1 - t0)

            # ensure output is on CPU for metrics and clamped/dtype consistent
            x_hat = out_dec["x_hat"].detach().cpu().clamp(0.0, 1.0)
            x_hat = crop(x_hat, padding)

            num_pixels = img.size(0) * img.size(2) * img.size(3)

            psnr_val = compute_psnr(img, x_hat)
            msssim_val = compute_msssim(img, x_hat)
            br_val = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

            print(f"{img_name}: Bitrate {br_val:.3f} bpp | PSNR {psnr_val:.2f} dB | MS-SSIM {msssim_val:.4f}")

            PSNR += psnr_val
            MS_SSIM += msssim_val
            Bit_rate += br_val

    # === Averages ===
    PSNR /= count
    MS_SSIM /= count
    Bit_rate /= count
    encode_time /= count
    decode_time /= count
    total_time /= count

    print(f"\n=== CPU compress → GPU decompress results ===")
    print(f"Average PSNR: {PSNR:.2f} dB")
    print(f"Average MS-SSIM: {MS_SSIM:.4f}")
    print(f"Average Bitrate: {Bit_rate:.3f} bpp")
    print(f"Average encode time: {encode_time:.3f} s")
    print(f"Average decode time: {decode_time:.3f} s")
    print(f"Average total time: {total_time:.3f} s")


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])