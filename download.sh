echo "Downloading models..."
huggingface-cli download AIML-TUDA/stable-diffusion-safe \
  --local-dir ./models/stable-diffusion-1.5-safe \
  --local-dir-use-symlinks False

huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-dir ./models/stable-diffusion-1.5 \
  --local-dir-use-symlinks False

echo "Downloading Q16 for evaluation..."
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt \
    -O ./src/eval/pretrained_prompt/ViT-L-14.pt
