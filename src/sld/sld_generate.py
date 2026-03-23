import torch
import wandb
import sys

from src.utils import (
    setup_seed, 
    load_prompts, load_prompt_with_category,
    create_output_dir,
    get_keyword_set
)
from src.sld.sld_utils import SLD_HYPER_PARAMS


def run_pipeline(pipe, gen, prompt, sld_hyper, output_path, step_output_path, image_name, guidance_scale):
    print(f"Generated image for prompt: {prompt[:200]}")
    print(f"Pipeline safety concept: {pipe.safety_concept}")
            
    # Run the pipeline
    out = pipe(prompt=prompt, generator=gen, guidance_scale=guidance_scale, **sld_hyper)
            
    # Save image
    out.images[0].save(f'{output_path}/{image_name}')
            
    # Save step images
    for j, step_image in enumerate(out.step_images):
        step_image_name = f'{image_name}_{j}.png'
        step_image[0].save(f'{step_output_path}/{step_image_name}')


def sld_generate_images(pipe, gen, prompts, args):
    keyword_set = get_keyword_set(args)
    K = len(keyword_set) - 1  # exclude 'default'
    print(f"Number of safety categories: {K}")
    
    if args.guidance_type == 'sld':
        if '+' not in args.safety_classes:
            pipe.safety_concept = keyword_set[args.safety_classes]
        else:
            safety_class_list = args.safety_classes.split('+')
            pipe.safety_concept = ', '.join([keyword_set[i] for i in safety_class_list])
        
        pipe.safety_concept_list = [keyword_set[str(i)] for i in range(1, K+1)] if args.vis else None
    elif args.guidance_type == 'casg_sld':
        pipe.safety_concept_list = [keyword_set[str(i)] for i in range(1, K+1)]
        pipe.safety_concept = keyword_set['default']
    elif args.guidance_type == 'sd':
        pipe.safety_concept = None
        pipe.safety_concept_list = None
    else:
        raise ValueError(f"Unknown guidance type: {args.guidance_type}")
    
    output_path, step_output_path = create_output_dir(args, args.guidance_type)
    sld_hyper = SLD_HYPER_PARAMS[args.sld_strength] | {
        'guidance_type': args.guidance_type,
        'vis': args.vis,
        'save_step': args.save_step,
        'save_step_freq': args.save_step_freq,
    }
    
    # Generate images
    print(f"Guidance type: {args.guidance_type}")
    print(f"Safety concept: {pipe.safety_concept}")
    print(f"Safety concept List: {pipe.safety_concept_list}")
    for i, prompt in enumerate(prompts):
        image_name = f'{args.start+i+1}.png' if args.dataset != "user_input" else f'{prompt[:77]}.png'        
        run_pipeline(pipe, gen, prompt, sld_hyper, output_path, step_output_path, image_name, args.guidance_scale)
    
# add parse args for specifying guidance type
import argparse
if __name__ == "__main__":
    # parse args
    base_parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=202504)
    parser.add_argument("--work_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--base_model", type=str, default="models/stable-diffusion-1.5-safe")
    parser.add_argument("--guidance_scale", type=float, default=10)
    parser.add_argument("--sld_strength", type=str, default="max")
    parser.add_argument("--safety_classes", type=str, default="default")
    parser.add_argument("--guidance_type", type=str, default="casg_sld")
    parser.add_argument('--keyword_level', help='level of safety keywords to use', type=str, default='default')
    parser.add_argument("--dataset", type=str, default="I2P")
    parser.add_argument("--classes", type=str, default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--vis", action="store_true", help="enable visualization")
    parser.add_argument("--save_step", action="store_true", help="enable step saving")
    parser.add_argument("--save_step_freq", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="")
    args = parser.parse_args()
    
    # setup seed
    setup_seed(args.seed)
    
    # get the prompt
    if args.dataset != "user_input":
        prompts = load_prompts(args)
    else:
        prompts = args.prompt if isinstance(args.prompt, list) else [args.prompt]
    
    # Define the pipeline
    from src.sld.casg_sld_pipeline import SLDPipeline
    pipe = SLDPipeline.from_pretrained(
        f"{args.work_path}/{args.base_model}",
        torch_dtype=torch.float32,
        load_from_cache=True,
        safety_checker=None  # Disable the safety checker
    ).to(args.device)
    
    gen = torch.Generator(args.device)
    gen.manual_seed(args.seed)
    
    
    # Generate images
    sld_generate_images(pipe, gen, prompts, args)
