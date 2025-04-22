import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from models import (
    DCAE,
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
import pandas as pd
import numpy as np
import torch.nn as nn
import struct

import socket
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    padding = (padding_left, padding_right, padding_top, padding_bottom)
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, padding

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--M", type=int, default=320,
    )
    parser.add_argument("--checkpoint", type=str, default="./60.5checkpoint_best.pth.tar", help="Path to a checkpoint")
    parser.add_argument("--save_path", type=str, help="Path to save")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument("--mode", type=str, choices=['compress', "decompress"])
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args

   
def save_bin(string, size, img_name, bin_path):
    file_name, ext = os.path.splitext(img_name)
    bin_name = file_name + '.bin'
    compress_path = os.path.join(bin_path,'bin/', bin_name)
    os.makedirs(os.path.dirname(compress_path), exist_ok=True)
    with open(compress_path, 'wb') as f:
        f.write(struct.pack(">H", size[0]))
        f.write(struct.pack(">H", size[1]))
        f.write(struct.pack(">I", len(string[0][0]))) 
        f.write(string[0][0])
        f.write(struct.pack(">I", len(string[1][0])))  
        f.write(string[1][0])

def send_file(args, img_name, host, port):
    # extract file name
    file_name, ext = os.path.splitext(img_name)
    file_name = file_name + '.bin'
    file_name = os.path.join(args.save_path, 'bin', file_name)
    # create TCP/IP protocal
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # connect to the host
        server_address = (host, port)
        print(f'connecting {server_address}')
        sock.connect(server_address)

        with open(file_name, 'rb') as f:
            file_size = f.seek(0, 2)
            f.seek(0)
            print(f'sending: {file_name},size: {file_size} bytes')
            header = f'{file_name}|{file_size}'
            sock.sendall(header.encode())

            ack = sock.recv(1024).decode()
            if ack != 'ACK':
                print('The server has not been confirmed and the transmission has been suspended.')
                return
            print('The server has been confirmed and the transmission has started.')
            while True:
                data = f.read(4096)
                if not data:
                    break
                sock.sendall(data)
            print('END')
    finally:
        sock.close()

def main(argv):
    torch.backends.cudnn.enabled = False
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"] and args.mode == "compress":
            img_list.append(file)
        elif file[-3:] in ["bin"] and args.mode == "decompress":
            # print(file)
            img_list.append(file)
        
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    base_path = args.save_path
    net = DCAE()
    net = net.to(device)
    net.eval()
    
    dictory = {}
    if args.checkpoint:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
        
    net.update()

    for img_name in img_list:    
        with torch.no_grad():
            img_path = os.path.join(path, img_name)
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
            x = img.unsqueeze(0)
            x_size = x.size()[-2:]
            x_padded, padding = pad(x, p)
            x_padded.to(device)
            out_enc = net.compress(x_padded)
            save_bin(out_enc["strings"], x_size, img_name, base_path)
            send_file(args, img_name, host='172.16.29.231', port=8888)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
    