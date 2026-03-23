import os
import torch
import numpy as np
import random
from src.utils import auto_output_dir

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_result_path(results_dir, eval_metric, args):
    if args.guidance_type in ('sld'):
        result_path = f"{results_dir}/{eval_metric}_{args.dataset}_{args.guidance_type}_{args.sld_strength}_{args.classes}_erease-{args.safety_classes}.txt"
    elif args.guidance_type in ('casg_sld', 'gpt_sld'):
        result_path = f"{results_dir}/{eval_metric}_{args.dataset}_{args.guidance_type}_{args.sld_strength}_{args.classes}.txt"
    elif args.guidance_type in ('safree'):
        result_path = f"{results_dir}/{eval_metric}_{args.dataset}_{args.guidance_type}_{args.classes}_erease-{args.safety_classes}.txt"
    else:  # including 'safree','sd', etc.
        result_path = f"{results_dir}/{eval_metric}_{args.dataset}_{args.guidance_type}_{args.classes}.txt"

    print(f"Results will be saved to: {result_path}")
    
    return result_path
        
def get_image_paths(args):
    # get the output path
    output_path, _ = auto_output_dir(args)
    
    # set the start and end
    image_paths = [f for f in os.listdir(output_path) if f.endswith('.png')]
    end = len(image_paths) if args.num == -1 else min(len(image_paths), args.start+args.num)
    
    # get the image paths
    image_paths = [f'{i+1}.png' for i in range(args.start, end)]
    image_paths = [os.path.join(output_path, image_path) for image_path in image_paths]
    
    print(f"Loaded {len(image_paths)} images from {output_path}")
    
    return output_path, image_paths


def get_original_image_paths(args):
    # get the original output path
    output_dir = f"{args.work_path}/{args.output_dir}/sd"
    if args.dataset != "user_input":
        output_path = f'{output_dir}/{args.dataset}/{args.classes}'
    else:
        output_path = f'{output_dir}/{args.dataset}'
            
    # set the start and end
    image_paths = [f for f in os.listdir(output_path) if f.endswith('.png')]
    end = len(image_paths) if args.num == -1 else min(len(image_paths), args.start+args.num)
    
    # get the image paths
    image_paths = [f'{i+1}.png' for i in range(args.start, end)]
    image_paths = [os.path.join(output_path, image_path) for image_path in image_paths]
    
    return output_path, image_paths

def get_prompts(args):
    with open(f"prompts/{args.dataset}/{args.classes}.txt", 'r', encoding='utf-8', errors='replace') as f:
        prompts = [line.strip() for line in f if line.strip()] 

    print(f"Loaded {len(prompts)} prompts from {args.dataset} dataset with safety class {args.classes}")
    
    return prompts

def get_category(image_path_list, args):
    if args.classes == 'all':
        detail_path = f"prompts/{args.dataset}/all_detail.txt"
    elif args.classes == 'multi_all':
        detail_path = f"prompts/{args.dataset}/multi_all_detail.txt"
    else:
        category_dict  = {}
        for image_path in image_path_list:
            category_dict[image_path] = args.classes
        return category_dict
    
    # load detail to get the category via mathcing image_id
    match_dic = {}
    with open(detail_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        for line in lines:
            split_item = line.strip().split(': ')
            image_id, category = split_item[0], split_item[1]
            
            # if multiple categories
            if category[0] == '[':
                category = category[1:-1].split(', ')
            
            match_dic[image_id] = category
    
    # get the category of the image
    category_dict = {}
    for image_path in image_path_list:
        image_name = os.path.basename(image_path)
        image_id = image_name.split('.')[0]
        
        category = match_dic.get(image_id, None)
        category_dict[image_path] = category
        # print(f"Image {image_id} belongs to category {category}")
    
    return category_dict
            
