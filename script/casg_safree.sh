# set working path from current path
work_path=/work/SAIC2025/yxia0023/CASG-release
cd $work_path
echo "Working directory set to $work_path"

# set parameters
output_dir=outputs  # Output directory
seed=202504  # Random seed for reproducibility
device=cuda  # Device to use for image generation

base_model=models/stable-diffusion-1.5  # Base model
guidance_scale=7.5  # Guidance scale
datasets=('I2P')  # Datasets to generate
classes=all  # Classes of prompts to generate, use the ids in the prompt files, or "all" for all classes
start=0  # Starting prompt index
num=1  # Number of images to generate, -1 for all

guidance_type=casg_safree  # Guidance type
safety_classes='default'  # Safety classes
keyword_level='default'  # Keyword level

# run the script
for dataset in "${datasets[@]}"; do
    python -m src.safree.safree_generate \
        --work_path $work_path \
        --output_dir $output_dir \
        --device $device \
        --seed $seed \
        --base_model $base_model \
        --guidance_scale $guidance_scale \
        --guidance_type $guidance_type \
        --dataset $dataset \
        --classes $classes \
        --safety_classes $safety_classes \
        --keyword_level $keyword_level \
        --start $start \
        --num $num
done
