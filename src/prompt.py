import os
from datasets import load_dataset
import numpy as np

import torch
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def I2P():
    # load dataset
    data = load_dataset("AIML-TUDA/i2p")
    prompts = data['train']['prompt']
    categories = data['train']['categories']
    
    # prepare the output directory
    prompt_dir = f"prompts/I2P"
    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)
    if os.path.exists(f"{prompt_dir}/all.txt"):
        os.remove(f"{prompt_dir}/all.txt")
    if os.path.exists(f"{prompt_dir}/all_detail.txt"):
        os.remove(f"{prompt_dir}/all_detail.txt")
    
    safety_categories = {
        "hate": 1,
        "harassment": 2,
        "violence": 3,
        "self-harm": 4,
        "sexual": 5,
        "shocking": 6,
        "illegal activity": 7
    }
    
    # include all prompts and details
    for i in range(len(prompts)):
        prompt, category = prompts[i], categories[i]
        prompt = prompt.replace('\n', ' ')
            
        # convert the category to a list of integers
        category = category.split(', ')
        category_id = []
        for c in category:
            category_id.append(safety_categories[c.strip()])
            
        with open(f"{prompt_dir}/all.txt", 'a') as f:
            f.write(prompt + '\n')
            
        with open(f"{prompt_dir}/all_detail.txt", 'a') as f:
            f.write(f"{i+1}: {category_id}: {prompt}\n")
    print(f"Get {len(prompts)} prompts from I2P dataset, saved to prompts/I2P/all.txt")

    # split the prompts into different files according to categories
    for safety_category, id in safety_categories.items():
        save_prompts = []
        for i in range(len(prompts)):
            prompt, category = prompts[i], categories[i]
            if str(safety_category) in category:
                save_prompts.append(prompt)
                    
        with open(f"{prompt_dir}/{id}.txt", 'w') as f:
            for prompt in save_prompts:
                f.write(prompt + '\n')
            
        with open(f"{prompt_dir}/{id}_detail.txt", 'w') as f:
            n_id = 1
            for prompt in save_prompts:
                f.write(f"{n_id}: {id}: {prompt}\n")
                n_id += 1
            n_id = 1
    print(f"Split prompts into different files according to categories, saved to prompts/I2P/xx.txt")

def T2VSafetyBench():
    # Define the directory where the prompts are stored
    prompts_dir = 'prompts/T2VSafetyBench'
    assert os.path.exists(prompts_dir), f"Prompts directory {prompts_dir} does not exist. Please make sure to download the T2VSafetyBench dataset and place the prompt files in the {prompts_dir} directory."
    
    # prepare the output directory
    if os.path.exists(os.path.join(prompts_dir, 'all.txt')):
        os.remove(os.path.join(prompts_dir, 'all.txt'))
    if os.path.exists(os.path.join(prompts_dir, 'all_detail.txt')):
        os.remove(os.path.join(prompts_dir, 'all_detail.txt'))

    # only keep relavant prompt files
    prompt_path_list = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '10.txt']
    prompt_path_list = [prompt_path for prompt_path in prompt_path_list if prompt_path.endswith('.txt')]
    
    id = 1
    for prompt_path in prompt_path_list:
        class_id = prompt_path.split('.')[0]
        with open(os.path.join(prompts_dir, prompt_path), 'r') as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
            prompts = [prompt.replace('\n', ' ') for prompt in prompts]

        with open(os.path.join(prompts_dir, 'all.txt'), 'a') as f:
            for prompt in prompts:
                f.write(prompt + '\n')
            
        with open(os.path.join(prompts_dir, 'all_detail.txt'), 'a') as f:
            for prompt in prompts:
                f.write(f"{id}: {class_id}: {prompt}\n")
                id += 1
        
        with open(os.path.join(prompts_dir, f'{class_id}_detail.txt'), 'w') as f:
            n_id = 1
            for prompt in prompts:
                f.write(f"{n_id}: {class_id}: {prompt}\n")
                n_id += 1
            n_id = 1
                
    print(f"Get {len(prompt_path_list)} classes prompts from T2VSafetyBench dataset, saved to {prompts_dir}")
   
def UD():
    # Define the directory where the prompts are stored
    prompts_dir = 'prompts/UD'
    assert os.path.exists(prompts_dir), f"Prompts directory {prompts_dir} does not exist. Please make sure to download the UD dataset and place the prompt files in the {prompts_dir} directory."
    
    # delete the all.txt file if it exists
    if os.path.exists(os.path.join(prompts_dir, 'all.txt')):
        os.remove(os.path.join(prompts_dir, 'all.txt'))
    if os.path.exists(os.path.join(prompts_dir, 'all_detail.txt')):
        os.remove(os.path.join(prompts_dir, 'all_detail.txt'))

    prompt_path_list = os.listdir(prompts_dir)
    prompt_path_list = ['4chan_prompts.txt', 'Lexica_prompts.txt']
    
    id = 1
    class_id = 1
    for prompt_path in prompt_path_list:
        with open(os.path.join(prompts_dir, prompt_path), 'r') as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
            prompts = [prompt.replace('\n', ' ') for prompt in prompts]
        
        with open(os.path.join(prompts_dir, 'all.txt'), 'a') as f:
            for prompt in prompts:
                f.write(prompt + '\n')
        
        with open(os.path.join(prompts_dir, 'all_detail.txt'), 'a') as f:
            for prompt in prompts:
                f.write(f"{id}: {class_id}: {prompt}\n")
                id += 1
        class_id += 1
                
    print(f"Get {len(prompt_path_list)} classes prompts from UD dataset, saved to {prompts_dir}/all.txt")

def coco(sample_size):
    # Load the COCO-30k dataset
    data = load_dataset("sayakpaul/coco-30-val-2014")
    captions = data['train']['caption']
    images = data['train']['image']

    # Randomly sample image paths and captions
    sampled_indices = np.random.choice(len(captions), sample_size, replace=False)
    sampled_captions = [captions[i] for i in sampled_indices]
    sampled_images = [images[i] for i in sampled_indices]

    # Create the directory if it does not exist
    prompt_dir = 'prompts/coco'
    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)
    # Remove the all.txt file if it exists
    if os.path.exists(os.path.join(prompt_dir, 'all.txt')):
        os.remove(os.path.join(prompt_dir, 'all.txt'))
    
    # Save the sampled captions to a file
    with open(os.path.join(prompt_dir, 'all.txt'), 'w') as f:
        for caption in sampled_captions:
            f.write(caption.replace('\n', ' ') + '\n')
    
    # Save the sampled images to a file
    if not os.path.exists('outputs/coco'):
        os.makedirs('outputs/coco')
    for i, image in enumerate(sampled_images):
        image.save(os.path.join('outputs/coco', f'{i+1}.jpg'))
        
    print(f"Sampled {sample_size} captions from COCO-30k dataset.")

def CoProv2(sample_size):
    data = load_dataset("Visualignment/CoProv2-SD15")
    prompts = data['train']['caption']
    
    # sample
    sampled_indices = np.random.choice(len(prompts), sample_size, replace=False)
    sampled_prompts = [prompts[i] for i in sampled_indices]
    prompts = sampled_prompts
    
    # Save all prompts to a file
    if not os.path.exists('prompts/CoProv2'):
        os.makedirs('prompts/CoProv2')
    if os.path.exists('prompts/CoProv2/all.txt'):
        os.remove('prompts/CoProv2/all.txt')
    if os.path.exists('prompts/CoProv2/all_detail.txt'):
        os.remove('prompts/CoProv2/all_detail.txt')
    
    with open('prompts/CoProv2/all.txt', 'w') as f:
        for prompt in prompts:
            f.write(prompt.replace('\n', ' ') + '\n')
    
    with open('prompts/CoProv2/all_detail.txt', 'w') as f:
        id = 1
        for prompt in prompts:
            f.write(f"{id}: 1: {prompt}\n")
            id += 1
    
    print(f"Sampled {sample_size} captions from CoProv2 dataset.")

if __name__ == '__main__':
    setup_seed(202504)
    
    I2P()
    T2VSafetyBench()
    UD()
    coco(sample_size=1000)
    CoProv2(sample_size=1000)

    print("All prompts have been generated and saved to the prompts directory.")