from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

import PIL
from PIL import Image

from diffusers.utils import BaseOutput, is_torch_available, is_transformers_available

__version__ = "0.0.1"

@dataclass
class SLDPipelineOutput(BaseOutput):
    """
    Output class for Safe Latent Diffusion pipelines.
    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        step_images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of PIL images or numpy array of shape `(batch_size, height, width, num_channels)` representing the
            intermediate denoised images of the diffusion pipeline. The length of the list is equal to the number of
            denoising steps.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
        applied_safety_concept (`str`)
            The safety concept that was applied for SLD, or `None` if SLD was disabled
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    step_images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    applied_safety_concept: Optional[str]
    applied_safety_concept_list: Optional[List[str]]

@dataclass
class StableDiffusion3PipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


