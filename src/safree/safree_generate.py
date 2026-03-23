import argparse
import torch
import wandb
import sys
import os

from src.utils import setup_seed, load_prompts, create_output_dir, get_keyword_set
from src.safree.safree_utils import Logger
from diffusers import DPMSolverMultistepScheduler



def load_sd(args, pipeline_func, device, weight_dtype, unet_ckpt=None):
    model_path = f"{args.work_path}/{args.base_model}"
    print(f"Loading model from {model_path}...")
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
    if 'xl' in model_path:
        pipe = pipeline_func.from_pretrained(
            model_path,
            scheduler=scheduler
            )
    else:
        pipe = pipeline_func.from_pretrained(
            model_path,
            scheduler=scheduler,
            torch_dtype=weight_dtype,
            safety_checker=None
        )
    
    if unet_ckpt is not None:
        unet_weight = torch.load(unet_ckpt, map_location='cpu')
        try:
            pipe.unet.load_state_dict(unet_weight)
        except:
            pipe.unet.load_state_dict(unet_weight['unet'])
        print(f"ESD unet: {unet_ckpt} is loaded...")
        
    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)
    pipe.unet.train()

    return pipe

def generate_images(args, pipe, gen, prompts, negative_prompt, negative_prompt_space, negative_prompt_space_list,
                    guidance_scale, num_inference_steps):
    # Create output directory
    output_dir, _ = create_output_dir(args, args.guidance_type)

    # set logger
    logger = Logger(f"{args.work_path}/{args.output_dir}/{args.guidance_type}/log.txt")
    
    # set safree hyperparameters
    re_attn_t = "-1,1001"
    sf_alpha = 0.01
    up_t = 10
    svf = True
    lra = True
    if args.guidance_type == "safree":
        safree = True
        casg_safree = False
    elif args.guidance_type == "casg_safree":
        safree = False
        casg_safree = True
    
    i = 0
    for prompt in prompts:
        print(f"Generating image for prompt {i+args.start+1}: {prompt}")
        imgs = pipe(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                negative_prompt_space=negative_prompt_space,
                negative_prompt_space_list=negative_prompt_space_list,
                height=512,
                width=512,
                generator=gen,
                vis=args.vis,
                safree_dict={"re_attn_t": [int(tr) for tr in re_attn_t.split(",")],
                                "alpha": sf_alpha,
                                "svf": svf,
                                "lra": lra,
                                "up_t": up_t,
                                'logger': logger,
                                "safree": safree,
                                'casg_safree': casg_safree
                                }
            )
        if args.dataset != "user_input":
            imgs[0].save(f'{output_dir}/{i+args.start+1}.png')
        else:
            imgs[0].save(f'{output_dir}/{prompt[:77]}.png')
        i += 1


import argparse
if __name__ == "__main__":
    # parse only config_type
    base_parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=202504)
    parser.add_argument("--work_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--base_model", type=str, default="models/stable-diffusion-1.5")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--guidance_type", type=str, default="casg_safree")
    parser.add_argument("--dataset", type=str, default="I2P")
    parser.add_argument("--classes", type=str, default="all")
    parser.add_argument("--safety_classes", type=str, default="default")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument('--keyword_level', help='level of keywords to use', type=str, default='default')
    parser.add_argument("--vis", action="store_true", help="visualize the direction conflict scores")
    args = parser.parse_args()
    
    # setup seed
    setup_seed(args.seed)
    
    # get the prompt and safety concept
    if args.dataset != "user_input":
        prompts = load_prompts(args) 
    else:
        prompts = args.prompt if isinstance(args.prompt, list) else [args.prompt]
    
    # Define the pipeline
    from src.safree.casg_safree_pipeline import SafreePipeline
    pipe= load_sd(args, SafreePipeline, args.device, torch.float32, None)
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    
    ################################################
    from src.safree.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
    freeu_hyp = "1.0-1.0-0.9-0.2"
    freeu_hyps = freeu_hyp.split('-')
    b1, b2, s1, s2 = float(freeu_hyps[0]), float(freeu_hyps[1]), float(freeu_hyps[2]), float(freeu_hyps[3])
        
    register_free_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)
    register_free_crossattn_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2)
    ################################################
    
    # get negative prompt, set it in ', ' separated str format
    safe_config = None
    keyword_set = get_keyword_set(args)
    try:
        if '+' not in args.safety_classes:
            negative_prompt_space = keyword_set[args.safety_classes]
        else:
            safety_class_list = args.safety_classes.split('+')
            negative_prompt_space = ', '.join([keyword_set[i] for i in safety_class_list])
    except:
        raise NotImplementedError(f"Not implemented safety_classes {args.safety_classes}")
    if isinstance(negative_prompt_space, list):
        negative_prompt = ", ".join(negative_prompt_space)
    else:
        negative_prompt = negative_prompt_space
    print(f"Negative prompt: {negative_prompt}")
    
    if args.guidance_type == "casg_safree" or args.vis == True:
        K = len(keyword_set) - 1 # exclude 'default'
        print(f"Number of safety categories: {K}")
        negative_prompt_space_list = [keyword_set[str(i)] for i in range(1, K+1)]
    else:
        negative_prompt_space_list = None
    print(f"Negative prompt space list: {negative_prompt_space_list}")
    
    # generate images
    generate_images(args, pipe, gen, prompts, negative_prompt, negative_prompt_space, negative_prompt_space_list,
                    guidance_scale=args.guidance_scale, num_inference_steps=50)

    
    
    