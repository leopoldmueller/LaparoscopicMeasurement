import os
import sys

sys.path.append("Raft-Stereo")
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

DEVICE = 'cuda'

class SimulateArgs():
    def __init__(self, config):
        self.hidden_dims = [128]*3
        self.n_gru_layers = 3
        self.slow_fast_gru = False
        self.corr_implementation = "reg"
        self.shared_backbone = False
        self.corr_levels = 4
        self.corr_radius = 4
        self.n_downsample = 2
        self.restore_ckpt = config['raftstereo']['restore_ckpt']
        self.save_numpy = config['raftstereo']['save_numpy']
        self.left_imgs = config['raftstereo']['path_left_images']
        self.right_imgs = config['raftstereo']['path_right_images']
        self.mixed_precision = config['raftstereo']['mixed_precision']
        self.valid_iters = config['raftstereo']['valid_iters']
        self.output_directory = config['raftstereo']['path_output_dir']
        self.divis_by = config['raftstereo']['divis_by']
        self.rgb = config['raftstereo']['rgb']

def load_image(imfile, rgb: bool):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # Uncomment for normal rgb images
    if not rgb:
        img = np.repeat(img[..., np.newaxis], 3, -1)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)
    #left_images = sorted(os.listdir(args.left_imgs))
    left_images = sorted([os.path.join(args.left_imgs, f) for f in os.listdir(args.left_imgs) if os.path.isfile(os.path.join(args.left_imgs, f))])
    right_images = sorted([os.path.join(args.right_imgs, f) for f in os.listdir(args.right_imgs) if os.path.isfile(os.path.join(args.right_imgs, f))])

    with torch.no_grad():
        #left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        #right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1, args.rgb)
            image2 = load_image(imfile2, args.rgb)

            padder = InputPadder(image1.shape, divis_by=args.divis_by)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            file_stem = imfile1.split('/')[-2]
            if args.save_numpy:
                np.save(output_directory / f"{os.path.basename(imfile1)[:-4]}.npy", flow_up.cpu().numpy().squeeze())
            plt.imsave(output_directory / f"{os.path.basename(imfile1)}", -flow_up.cpu().numpy().squeeze(), cmap='jet')
    

def run_RAFT_stereo(config):
    args = SimulateArgs(config)
    demo(args)
