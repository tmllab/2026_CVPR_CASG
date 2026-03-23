import wandb
import sys
import torch
import os

from src.utils import setup_seed, create_output_dir, get_keyword_set
from src.sld.sld_utils import SLD_HYPER_PARAMS

def load_gpt_prompt_class(args):
    # load the prompt class
    with open(f"{args.work_path}/prompts/{args.dataset}/gpt_{args.classes}_detail.txt", 'r') as f:
        lines = f.readlines()
    
    # get the prompt and category
    category_prompts = []
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        else:
            line = line.split(':')
            category_id = line[1].strip()
            category_id = int(category_id[1:-1])
            prompt = line[2].strip()
            category_prompts.append((category_id, prompt))
    
    if args.num != -1:
        max_num = min(args.start+args.num, len(category_prompts))
    else:
        max_num = len(category_prompts)
    
    category_prompts = category_prompts[args.start:max_num]
    print(f"Load {len(category_prompts)} prompts from {args.dataset} dataset.")
    
    return category_prompts


import argparse
if __name__=='__main__':
    # parse only config_type
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=202504)
    parser.add_argument("--work_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--base_model", type=str, default="models/stable-diffusion-1.5-safe")
    parser.add_argument("--guidance_scale", type=float, default=10)
    parser.add_argument("--guidance_type", type=str, default="gpt_sld")
    parser.add_argument("--sld_strength", type=str, default="max")
    parser.add_argument("--safety_classes", type=str, default="default")
    parser.add_argument("--dataset", type=str, default="I2P")
    parser.add_argument("--classes", type=str, default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--gpt_sld_default", type=str, default="sld")
    parser.add_argument("--save_step", action="store_true", help="enable step saving")
    parser.add_argument("--save_step_freq", type=int, default=5)
    parser.add_argument('--keyword_level', help='level of keywords to use', type=str, default='default')
    args = parser.parse_args()

    # setup seed
    setup_seed(args.seed)
    
    # load the prompt class and category
    category_prompts = load_gpt_prompt_class(args)
    
    # define the SLD pipeline
    from src.sld.casg_sld_pipeline import SLDPipeline as SLDPipeline
    pipe = SLDPipeline.from_pretrained(
        f"{args.work_path}/{args.base_model}",
        torch_dtype=torch.float32,
        safety_checker=None  # Disable the safety checker
    ).to(args.device)
    gen = torch.Generator(args.device)
    gen.manual_seed(args.seed)
    sld_hyper = SLD_HYPER_PARAMS[args.sld_strength] | {
        'guidance_type': 'sld',
        'save_step': args.save_step,
        'save_step_freq': args.save_step_freq,
    }
    
    output_path, step_output_path = create_output_dir(args, args.guidance_type)
    keyword_set = get_keyword_set(args)
    K = len(keyword_set) - 1  # exclude the 'default' keyword
    
    # get the prompt and category
    for i in range(len(category_prompts)):
        category_id, prompt = category_prompts[i]
        image_name = f'{args.start+i+1}.png'
        
        # set the safety concept for the current category
        if category_id == 0:  # if no category, use default setting
            sld_hyper['guidance_type'] = args.gpt_sld_default
            pipe.safety_concept = keyword_set['default'] if args.gpt_sld_default == 'sld' else None
            print(f"Default safety concept: {pipe.safety_concept}")
        else:
            sld_hyper['guidance_type'] = 'sld'
            pipe.safety_concept = keyword_set[str(category_id)]
            
        
        print(f"Safety concept: {category_id}, {pipe.safety_concept}")
        print(f"Generated image for prompt: {prompt[:200]}")
        out = pipe(prompt=prompt, generator=gen, guidance_scale=args.guidance_scale, **sld_hyper)
        out.images[0].save(f'{output_path}/{image_name}')
    
    print(f"Generated {len(category_prompts)} images for {args.dataset} dataset.")
        
        