"""
FID Evaluation Script: https://huggingface.co/docs/diffusers/v0.21.0/en/conceptual/evaluation
"""

import torch_fidelity
import torch
from PIL import Image
import torch
import os
import numpy as np
import wandb
import argparse

from torchvision.transforms import functional as F
from src.eval.utils import get_image_paths, get_original_image_paths, setup_seed, get_result_path


def calculate_fid(original_generated_images, new_generated_images, args):
    global _fid_model
    os.environ['TORCH_HOME'] = f"{args.work_path}/models/eval"
    
    metrics = torch_fidelity.calculate_metrics(
        input1=original_generated_images,
        input2=new_generated_images,
        fid=True,
        cuda=args.device.startswith("cuda"),
        batch_size=32,
        verbose=False
    )
    return round(float(metrics["frechet_inception_distance"]), 3)

def fid_main(args):
    # seed
    setup_seed(args.seed)
    
    # get image paths
    original_image_dir, _ = get_original_image_paths(args)
    new_image_dir, _ = get_image_paths(args)
    print(f"Original images: {original_image_dir}")
    print(f"New images: {new_image_dir}")
    
    # calculate fid
    fid_score = calculate_fid(original_image_dir, new_image_dir, args)

    # check counts
    ori_count = len([f for f in os.listdir(original_image_dir) if f.endswith('.png')])
    new_count = len([f for f in os.listdir(new_image_dir) if f.endswith('.png')])

    # save results
    results_dir = f"{args.work_path}/{args.results_dir}"
    if not os.path.exists(f"{results_dir}"):
        os.makedirs(f"{results_dir}")
    
    result_path = get_result_path(results_dir, 'fid', args)
    with open(result_path, 'w') as f:
        f.write(f"FID score: {fid_score}\n")
        f.write(f"Original Total images: {ori_count}\n")
        f.write(f"New Total images: {new_count}\n")
    
    # print results
    print("-------- FID Evaluation --------")
    print(f"FID score: {fid_score}")
    print(f"Ori Total images: {ori_count}")
    print(f"New Total images: {new_count}")
    
    return fid_score


import argparse
if __name__=='__main__':
    base_parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=202504)
    parser.add_argument("--work_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--guidance_scale", type=float, default=10)
    parser.add_argument("--guidance_type", type=str, default="sd")
    parser.add_argument("--sld_strength", type=str, default="max")
    parser.add_argument("--safety_classes", type=str, default="default")
    parser.add_argument("--dataset", type=str, default="I2P")
    parser.add_argument("--classes", type=str, default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--fid_compare", type=str, default="sd")
    args = parser.parse_args()

    fid_main(args)
