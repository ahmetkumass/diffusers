import torch

from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    SchedulerMixin,
    DiffusionPipeline,
    StableDiffusionMixin,
)


import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel, BrushNetModel
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from ..stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker

##############
class StableDiffusionControlNetBrushNetPipeline(DiffusionPipeline, StableDiffusionMixin):
    def __init__(
        self, 
        vae, 
        text_encoder, 
        tokenizer, 
        unet, 
        scheduler, 
        safety_checker, 
        feature_extractor, 
        controlnet, 
        brushnet,
        torch_dtype=torch.float32
    ):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.safety_checker = safety_checker
        self.feature_extractor = feature_extractor
        self.controlnet = controlnet
        self.brushnet = brushnet
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            controlnet=controlnet,
            brushnet=brushnet
        )

    def __call__(
        self, 
        prompt, 
        image, 
        mask_image, 
        control_image, 
        num_inference_steps=50, 
        strength=0.8, 
        guidance_scale=7.5, 
        generator=None, 
        brushnet_conditioning_scale=1.0, 
        controlnet_conditioning_scale=1.0
    ):
        latents = self.vae.encode(image).latent_dist.sample(generator=generator)
        latents = latents * strength + self.vae.encode(self.vae.decode(latents)).latent_dist.sample(generator=generator) * (1 - strength)
        
        # Encode prompt
        text_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

        # Encode control and brush images
        control_latents = self.controlnet(control_image)
        brush_latents = self.brushnet(mask_image)

        # Conditioning scales
        control_latents *= controlnet_conditioning_scale
        brush_latents *= brushnet_conditioning_scale

        # Run the denoising diffusion process
        for i in range(num_inference_steps):
            t = self.scheduler.timesteps[-1 - i]
            noise_pred = self.unet(latents, t, text_embeddings=text_embeddings, noise=None)
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)["prev_sample"]

            # Combine control and brush outputs
            combined_latents = control_latents + brush_latents
            latents += combined_latents * guidance_scale

        # Decode the final latents to an image
        image = self.vae.decode(latents)

        return image

    @classmethod
    def from_pretrained(cls, sd_path, controlnet, brushnet, torch_dtype=torch.float32):
        # Load other necessary components for the pipeline
        model = super().from_pretrained(sd_path, torch_dtype=torch_dtype)
        model.controlnet = controlnet
        model.brushnet = brushnet
        return model


