"""
CLIP Score Evaluation: https://huggingface.co/docs/diffusers/v0.21.0/en/conceptual/evaluation
"""

import torch
import os
from PIL import Image
import wandb
import argparse
import numpy

import torchmetrics.functional.multimodal as multimodal
from functools import partial

from src.eval.utils import get_image_paths, get_prompts, setup_seed, get_result_path



def calculate_clip_score(clip_score_fn, images, prompts):
    clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def clip_score_main(args):
    # seed
    setup_seed(args.seed)
    
    # Get image paths and prompt path
    _, image_paths = get_image_paths(args)
    prompts = get_prompts(args)
    
    # Pair image paths with prompts
    pairs = [(img_path, prompts[int(img_path.split('/')[-1].split('.')[0]) - 1]) for img_path in image_paths]
    # Initialize the CLIP score function
    import os
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = f"{args.work_path}/models/eval"
    
    clip_score_fn = partial(multimodal.clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    
    # Calculate the CLIP score for each image-prompt pair
    clip_scores_dict = {}
    for img_path, prompt in pairs:
        image = numpy.asarray([Image.open(img_path).convert("RGB")])
        clip_score = calculate_clip_score(clip_score_fn, image, prompt)
        clip_scores_dict[img_path] = clip_score
        print(f"Image: {img_path}, Prompt: {prompt}, CLIP Score: {clip_score}")
    avg_clip_score = sum(clip_scores_dict.values()) / len(clip_scores_dict)
    
    # Save the results
    results_dir = f"{args.work_path}/{args.results_dir}"
    if not os.path.exists(f"{results_dir}"):
        os.makedirs(f"{results_dir}")
    
    result_path = get_result_path(results_dir, 'clip', args)
    with open(result_path, 'w') as f:
        f.write(f"Average CLIP score: {avg_clip_score}\n")
        f.write(f"Total images: {len(image_paths)}\n")
        for img_path, clip_score in clip_scores_dict.items():
            f.write(f"{img_path}: {clip_score}\n")
    
    # Log the results
    try:
        wandb.log({
            # results
            "avg_clip_score": avg_clip_score,
            "total_count": len(image_paths),
            # method settings
            "guidance_type": args.guidance_type,
            "sld_strength": args.sld_strength if args.guidance_type in ['ori_sld', 'avg_sld', 'casg'] else 'N/A',
            "safety_classes": args.safety_classes if args.guidance_type in ['ori_sld', 'safree', 'esd', 'uce', 'rece'] else 'N/A',
            "dataset": args.dataset
        })
    except:
        print("Wandb logging failed.")
    
    # print the average CLIP score
    print("-------- CLIP Score Evaluation --------")
    print(f"Average CLIP score: {avg_clip_score}")
    print(f"Total images: {len(image_paths)}")
    
    return avg_clip_score, clip_scores_dict.values()

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
    args = parser.parse_args()
    
    clip_score_main(args)