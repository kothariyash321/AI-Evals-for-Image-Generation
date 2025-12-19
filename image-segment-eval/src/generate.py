"""Generate images using Stable Diffusion for different customer segments."""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm

from src.utils import (
    load_segments,
    load_base_prompt,
    build_segment_prompt,
    ensure_dir,
    get_device,
)


def generate_images(
    segments: Dict[str, Any],
    base_prompt: str,
    out_dir: str,
    num_images: int,
    steps: int,
    guidance: float,
    device: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
):
    """Generate images for each segment."""
    print(f"Loading model {model_id} on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    
    if device == "cpu":
        # Use less memory on CPU
        pipe.enable_attention_slicing()
    
    ensure_dir(out_dir)
    
    all_metadata = []
    
    for segment_name, segment_data in segments.items():
        print(f"\nGenerating images for segment: {segment_name}")
        segment_dir = os.path.join(out_dir, segment_name)
        ensure_dir(segment_dir)
        
        prompt = build_segment_prompt(base_prompt, segment_data["style_modifiers"])
        print(f"Prompt: {prompt}")
        
        for i in tqdm(range(num_images), desc=f"  {segment_name}"):
            seed = 42 + i * 1000  # Deterministic seeds
            generator = torch.Generator(device=device).manual_seed(seed)
            
            # Generate image and track inference time
            start_time = time.time()
            with torch.no_grad():
                image = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                ).images[0]
            inference_time = time.time() - start_time
            
            # Save image
            image_filename = f"{segment_name}_{i:03d}.png"
            image_path = os.path.join(segment_dir, image_filename)
            image.save(image_path)
            
            # Store metadata
            metadata = {
                "segment": segment_name,
                "image_path": image_path,
                "prompt": prompt,
                "seed": seed,
                "model": model_id,
                "timestamp": datetime.now().isoformat(),
                "steps": steps,
                "guidance": guidance,
                "inference_time": round(inference_time, 2),
            }
            all_metadata.append(metadata)
            
            # Save metadata JSONL per segment
            metadata_file = os.path.join(segment_dir, "metadata.jsonl")
            with open(metadata_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metadata) + "\n")
    
    print(f"\nGenerated {len(all_metadata)} images total")
    return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Generate images for customer segments")
    parser.add_argument(
        "--segments",
        type=str,
        default="prompts/segments.json",
        help="Path to segments JSON file",
    )
    parser.add_argument(
        "--base_prompt",
        type=str,
        default="prompts/base_prompt.txt",
        help="Path to base prompt text file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="images",
        help="Output directory for images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate per segment",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model ID",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage",
    )
    
    args = parser.parse_args()
    
    segments = load_segments(args.segments)
    base_prompt = load_base_prompt(args.base_prompt)
    device = get_device(force_cpu=args.cpu)
    
    generate_images(
        segments=segments,
        base_prompt=base_prompt,
        out_dir=args.out_dir,
        num_images=args.num_images,
        steps=args.steps,
        guidance=args.guidance,
        device=device,
        model_id=args.model,
    )


if __name__ == "__main__":
    main()

