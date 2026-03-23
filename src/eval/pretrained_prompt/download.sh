# download the prompts.p files for different model architectures and save them in different name files
wget https://github.com/ml-research/Q16/raw/main/data/ViT-B-16/prompts.p -O ViT-B-16_prompts.p
wget https://github.com/ml-research/Q16/raw/main/data/ViT-B-32/prompts.p -O ViT-B-32_prompts.p
wget https://github.com/ml-research/Q16/raw/main/data/ViT-L-14/prompts.p -O ViT-L-14_prompts.p

# download the model checkpoint files for different model architectures and save them in different name files
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt -O ViT-B-16.pt
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -O ViT-B-32.pt
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt -O ViT-L-14.pt
