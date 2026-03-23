# When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance
[![arXiv](https://img.shields.io/badge/Arxiv-2502.17198-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2602.20880)

This repository is the official implementation of our CVPR 2026 paper (CASG), "When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance".

![](figs/harmful_conflict.png)

**Safety Notice:** This repository is intended for safety research only. Some prompts or evaluation data may involve sensitive or harmful content.

## Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/tmllab/2026_CVPR_CASG.git
cd 2026_CVPR_CASG

# Install dependencies (compatible with Python 3.11.9)
pip install -r requirements.txt

# Download pretrained models and required assets
bash download.sh
```

### 2. Prompt Preparation
Prompt datasets are provided in the `prompts/` directory, including the raw prompts and related metadata (e.g., grounded category and GPT-evaluated category).

- For T2VSafetyBench, we only keep the prompts that are relevant to our 7 safety categories for a fair comparison.
- You can also provide your own prompts by creating a `.txt` file with one prompt per line and placing it under `prompts/{dataset_name}/{class_name}.txt`. You can then specify the dataset name and class name in the generation command (`python -m src.prompt`).

### 3. Image Generation
We support multiple T2I safety methods, including our proposed CASG and several baselines: `sd` (Stable Diffusion v1.5), `sld` (Safe Latent Diffusion), `gpt_sld` (GPT-based safety guidance on SLD), `casg_sld` (CASG on SLD), `safree`, and `casg_safree` (CASG on safree).

Specify the mode and other parameters in `script/xx.sh`, then run:
```bash
bash script/casg_sld.sh
bash script/casg_safree.sh
```

Important parameters:
- We focus on 7 harmful content categories in our paper, which are commonly used in T2I safety research (`1. Hate`,`2. Harassment`,`3. Violence`,`4. Self-harm`, `5. Sexual content`, `6. Disturbing content`, `7. Illegal activities`).  For convenience, we use category IDs in subsequent commands, where the ID definitions for safety classes follow the above ordering, and the mappings for different prompt datasets can be found in `prompts/definition.txt`.
- keywords_level: controls the predefined harmful keyword granularity used for safeguards. We support `abstract`, `default`, and `detail`, and recommend `default`.
- sld_strength: controls the strength of SLD. We support `medium`, `strong`, and `max`, and recommend `max` for better safety performance.

### 4. Evaluation
Evaluation results are automatically saved under `eval_results/`. We report three metrics: 
- Attack Success Rate (ASR): which measures the percentage of harmful images detected for harmful prompts (lower is safer); 
- CLIP Score, which measures text-image alignment for benign prompts (higher is better); 
- FID Score, which measures image realism for benign prompts (lower is better).

Run the evaluation scripts as follows:
```bash
bash script/eval/asr.sh
bash script/eval/clip_score.sh
bash script/eval/fid.sh
```

## Harmful Conflict
### Harmful Conflict Analysis
You can enable `--vis` in the script to visualize the *Cross-Category Directional Conflict* and *Aggregated Directional Attenuation* discussed in Section 3. The figures will be saved in `fig/`.

### Safety Degradation Analysis
You can verify the safety degradation caused by harmful conflict by modifying the combination of `--classes` and `--safety_classes` in the script. In particular, mismatched category pairs can be used to evaluate *safety misalignment degradation*, while setting `--safety_classes default` can be used to evaluate *safety averaging degradation*. The resulting safety performance can be measured with ASR. Example scripts are provided in `script/conflict` for reference.

## Citation
If you find this work useful in your research, please consider citing our paper:
```
@article{xiang2026safety,
  title={When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance},
  author={Xiang, Yongli and Hong, Ziming and Wang, Zhaoqing and Zhao, Xiangyu and Han, Bo and Liu, Tongliang},
  journal={arXiv preprint arXiv:2602.20880},
  year={2026}
}
```

## Acknowledgement
Parts of this project were inspired by the following projects. We thank their contributors for their excellent work: 
- https://github.com/ml-research/safe-latent-diffusion 
- https://github.com/jaehong31