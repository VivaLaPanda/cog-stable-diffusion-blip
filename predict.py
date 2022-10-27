import os
from typing import Optional, List

import torch
from torch import autocast
from diffusers import PNDMScheduler, LMSDiscreteScheduler
from PIL import Image
from cog import BasePredictor, Input, Path

from image_to_image import (
    StableDiffusionImg2ImgPipeline,
    preprocess_init_image,
)

from blip import ImageDescriber

MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        scheduler = PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

        self.blip = ImageDescriber()

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        init_image: Path = Input(
            description="Inital image to provide structural or conceptual guidance",
            default=None,
        ),
        captioning_model: str = Input(
            description="Captioning model to use. One of 'blip' or 'clip-interrogator-v1'",
            default="blip",
            choices=["blip", "clip-interrogator-v1"],
        ),
        structural_image_strength: float = Input(
            description="Structural (standard) image strength. 0.0 corresponds to full destruction of information, and does not use the initial image for structure.",
            default=0.15,
        ),
        conceptual_image_strength: float = Input(
            description="Conceptual image strength. 0.0 doesn't use the image conceptually at all, 1.0 only uses the image concept and ignores the prompt.",
            default=0.4
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if init_image is None:
            raise ValueError(
                "Please select an initial image."
            )

        original_image = Image.open(init_image).convert("RGB")
        init_image, width, height = preprocess_init_image(original_image, width, height).to("cuda")

        blip_prompt, clip_inter_prompt = self.blip.interrogate(original_image, models=["ViT-L/14"])
        
        img_prompt = blip_prompt if captioning_model == "blip" else clip_inter_prompt
        print("Captioning using", captioning_model)
        print("Image prompt:", img_prompt)

        scheduler = PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )

        self.pipe.scheduler = scheduler
        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.no_grad():
            output = self.pipe(
                prompt=[prompt] * 4 if prompt is not None else None,
                init_image=init_image,
                image_prompt=img_prompt,
                width=width,
                height=height,
                prompt_strength=(1-structural_image_strength),
                conceptual_prompt_strength=conceptual_image_strength,
                generator=generator,
            )
        if any(output["nsfw_content_detected"]):
            raise Exception("NSFW content detected, please try a different prompt")

        output_paths = []
        for i, sample in enumerate(output["sample"]):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
