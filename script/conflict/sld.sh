# set working path from current path
work_path=.
cd $work_path
echo "Working directory set to $work_path"

# set parameters
output_dir=outputs  # Output directory
seed=202504  # Random seed for reproducibility
device=cuda  # Device to use for image generation

base_model=models/stable-diffusion-1.5  # Base model
guidance_scale=7.5  # Guidance scale
dataset='T2VSafetyBench'  # Datasets to generate
class=1  # Classes of prompts to generate, use the ids in the prompt files, or "all" for all classes
start=0  # Starting prompt index
num=500  # Number of images to generate, -1 for all

guidance_type=sld  # Guidance type
safety_classes=(1 5 '1+5' 'default')  # Safety classe
sld_strength=max  # SLD strength
keyword_level='default'  # Keyword level

# run the script
for safety_class in "${safety_classes[@]}"; do
    python -m src.sld.sld_generate \
        --output_dir $output_dir \
        --work_path $work_path \
        --device $device \
        --seed $seed \
        --base_model $base_model \
        --guidance_scale $guidance_scale \
        --guidance_type $guidance_type \
        --sld_strength $sld_strength \
        --safety_classes $safety_classes \
        --keyword_level $keyword_level \
        --dataset $dataset \
        --classes $class \
        --start $start \
        --num $num
done
