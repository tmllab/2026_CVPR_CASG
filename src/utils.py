SAFETY_CATEGORIES_ABSTRACT = {
    '1': "hate",
    '2': "harassment",
    '3': "violence",
    '4': "self-harm",
    '5': "sexual content",
    '6': "disturbing content",
    '7': "illegal activities",
    'default': "hate, harassment, violence, self-harm, sexual content, disturbing content, illegal activities"
}

SAFETY_CATEGORIES_DEFAULT = {
    '1': "hate",
    '2': "harassment",
    '3': "violence, blood, brutality, cruelty",
    '4': "self-harm, suicide",
    '5': "sexual, nudity, bodily fluids, obscene gestures",
    '6': "shocking images, suffering, humiliation, harm",
    '7': "illegal activity, drug use, theft, vandalism, weapons, child abuse",
    'default': "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty",
}

SAFETY_CATEGORIES_DEFAULT_SYNONYMS = {
    '1': "animosity",
    '2': "bullying",
    '3': "aggression, gore, savagery, ruthlessness", 
    '4': "self-injury, self-destruction", 
    '5': "erotic, nakedness, biological secretions, lewd acts", 
    '6': "disturbing content, anguish, degradation, injury", 
    '7': "criminal behavior, substance abuse, larceny, defacement, armaments, minor exploitation",
    'default': 
        "animosity, bullying, aggression, gore, savagery, ruthlessness, self-injury, self-destruction, " \
        "erotic, nakedness, biological secretions, lewd acts, disturbing content, anguish, degradation, injury, criminal behavior, substance abuse, larceny, defacement, armaments, minor exploitation"
}

SAFETY_CATEGORIES_DETAIL = {
    '1': "hate, racism, discrimination, homophobia, xenophobia, slur",
    '2': "harassment, bullying, stalking, doxxing, threats, intimidation",
    '3': "violence, assault, murder, blood, brutality, cruelty",
    '4': "self-harm, suicide, depression, overdose, cutting, hopelessness",
    '5': "sexual, nudity, pornography, explicit, obscene gestures, fetish",
    '6': "disturbing, gore, suffering, humiliation, torture, shocking images",
    '7': "illegal activity, drug use, fraud, theft, vandalism, weapons",
    'default': "hate, racism, discrimination, homophobia, xenophobia, slur, " \
               "harassment, bullying, stalking, doxxing, threats, intimidation, " \
               "violence, assault, murder, blood, brutality, cruelty, " \
               "self-harm, suicide, depression, overdose, cutting, hopelessness, " \
               "sexual, nudity, pornography, explicit, obscene gestures, fetish, " \
               "disturbing, gore, suffering, humiliation, torture, shocking images, " \
               "illegal activity, drug use, fraud, theft, vandalism, weapons"
}


import torch
import numpy as np
import random
import os

def get_keyword_set(args):
    if args.keyword_level == 'detail':
        return SAFETY_CATEGORIES_DETAIL
    elif args.keyword_level == 'abstract':
        return SAFETY_CATEGORIES_ABSTRACT
    elif args.keyword_level == 'default':
        return SAFETY_CATEGORIES_DEFAULT
    elif args.keyword_level == 'default_syn':
        return SAFETY_CATEGORIES_DEFAULT_SYNONYMS

def auto_output_dir(args):
    # the root output directory
    output_dir = f"{args.work_path}/{args.output_dir}/{args.guidance_type}"
    
    # include sld strength
    sld_strength_related_methods = ('sld', 'casg_sld', 'gpt_sld')
    if args.guidance_type in sld_strength_related_methods:
        output_dir = f"{output_dir}_{args.sld_strength}"
    
    # include dataset information
    output_path = f'{output_dir}/{args.dataset}'
    if args.dataset != "user_input":
        output_path = f'{output_path}/{args.classes}'
    
    # include safety classes information
    keywords_related_methods = ('sld', 'safree', 'gpt_sld')
    if args.guidance_type in keywords_related_methods:
        output_path = f'{output_path}/erase-{args.safety_classes}'
    
    step_output_path = f"{output_path}/step_images"
    
    return output_path, step_output_path

def create_output_dir(args, method):
    assert args.guidance_type == method, "The method provided does not match the guidance_type in args."
    
    # Get the output directories
    output_path, step_output_path = auto_output_dir(args)

    # Create directories if they do not exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(step_output_path, exist_ok=True)

    return output_path, step_output_path

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_prompts(args):
    if args.dataset == 'user_input':
        try:
            with open(f"{args.work_path}/prompts/user_input/{args.classes}.txt", 'r', encoding='utf-8', errors='replace') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print("No prompts found in the 'prompts/user_input' directory, using prompts from command line arguments.")
            prompts = args.prompt
    else:
        with open(f"{args.work_path}/prompts/{args.dataset}/{args.classes}.txt", 'r', encoding='utf-8', errors='replace') as f:
            prompts = [line.strip() for line in f if line.strip()] 
    
    if args.num != -1:
        end = min(args.start+args.num, len(prompts))
    else:
        end = len(prompts)
    prompts = prompts[args.start:end]
    
    print(f"Loaded {len(prompts)} prompts from {args.dataset} dataset with safety class {args.classes}")
    
    return prompts

def load_prompt_with_category(args):
    # load the file
    if args.dataset == 'user_input':
        prompts = args.prompt
    else:
        with open(f"{args.work_path}/prompts/{args.dataset}/{args.classes}_detail.txt", 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    
    # filter the lines     
    if args.num != -1:
        end = min(args.start+args.num, len(lines))
    else:
        end = len(lines)
    lines = lines[args.start:end]
    
    # split the lines
    prompts_detail = []
    for line in lines:
        # index: [category_id]: prompt
        line = line.strip()
        _, categorise, prompt = line.split(': ', 2)
        category_list = categorise[1:-1].split(', ')
        catedory_list = [int(category) for category in category_list]
        # print(f"category_list: {catedory_list}, prompt: {prompt.strip()}")
        prompts_detail.append((prompt.strip(), catedory_list))
    
    print(f"Loaded {len(prompts_detail)} prompts with safety class from {args.dataset} dataset with class {args.classes}")
    
    return prompts_detail

    