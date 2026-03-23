# set working path from current path
work_path=.
cd $work_path
echo "Working directory set to $work_path"

# set parameters
output_dir=outputs  # Output directory
result_dir=results  # Result directory
seed=202504  # Random seed for reproducibility
device=cuda  # Device to use for image generation

base_model=models/stable-diffusion-1.5-safe  # Base model
guidance_scale=7.5  # Guidance scale
datasets=('I2P')  # Datasets to generate
classes=all  # Classes of prompts to generate, use the ids in the prompt files, or "all" for all classes
start=0  # Starting prompt index
num=-1  # Number of images to generate, -1 for all

guidance_types=('sd' 'safree' 'casg_safree' 'sld' 'casg_sld')  # Guidance types
sld_strength='max'  # SLD strengths
safety_classes='default'  # Safety classes
keyword_level='default'  # Keyword level

for dataset in "${datasets[@]}"; do
    for guidance_type in "${guidance_types[@]}"; do
            echo ">>> Starting CLIP score evaluation for dataset: $dataset with guidance type: $guidance_type ..."
            python -m src.eval.clip_score_eval \
            --work_path $work_path \
            --output_dir $output_dir \
            --results_dir $result_dir \
            --device $device \
            --seed $seed \
            --guidance_scale $guidance_scale \
            --guidance_type $guidance_type \
            --sld_strength $sld_strength \
            --safety_classes $safety_classes \
            --dataset $dataset \
            --classes $classes \
            --start $start \
            --num $num
            echo ">>> CLIP score evaluation for dataset: $dataset with guidance type: $guidance_type finished."
    done
done
