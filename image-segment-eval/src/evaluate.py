"""Evaluate images using a scoring rubric with HEIM-inspired metrics."""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.utils import load_segments, build_segment_descriptor, get_device


def load_clip_model(device: str):
    """Load CLIP model for similarity scoring."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model = model.to(device)
        model.eval()
        return model, preprocess, tokenizer
    except Exception as e:
        print(f"Warning: Could not load CLIP model: {e}")
        print("Falling back to text-only heuristic for segment_fit")
        return None, None, None


def load_aesthetics_model(device: str):
    """Load aesthetics scoring model."""
    try:
        # Use CLIP model for aesthetics scoring (simpler approach)
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model = model.to(device)
        model.eval()
        return model, preprocess, tokenizer
    except Exception as e:
        print(f"Warning: Could not load aesthetics model: {e}")
        return None, None, None


def compute_clip_similarity(
    image: Image.Image,
    text: str,
    model,
    preprocess,
    tokenizer,
    device: str,
) -> Tuple[float, float]:
    """Compute CLIP similarity between image and text.
    Returns: (normalized_score_0_5, raw_clip_score)
    """
    try:
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        text_tokens = tokenizer([text]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity (raw CLIP score)
            raw_score = (image_features @ text_features.T).item()
            
            # Map from [-1, 1] to [0, 5] for normalized score
            normalized = (raw_score + 0.3) / 1.1
            normalized = max(0, min(5, normalized * 5))
            return normalized, raw_score
    except Exception as e:
        print(f"Error computing CLIP similarity: {e}")
        return 2.5, 0.0  # Default middle score


def compute_aesthetics_score(image: Image.Image, model, preprocess, tokenizer, device: str) -> float:
    """Compute aesthetics score using CLIP-based aesthetics model."""
    if model is None or preprocess is None or tokenizer is None:
        return 0.0
    
    try:
        # Aesthetics prompts
        aesthetics_prompts = [
            "a beautiful image",
            "an aesthetically pleasing image",
            "a high quality image",
            "a visually appealing image",
            "professional photography",
        ]
        
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        text_tokens = tokenizer(aesthetics_prompts).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Average similarity across prompts
            similarities = (image_features @ text_features.T).mean().item()
            
            # Map to 0-5 scale
            score = max(0, min(5, (similarities + 0.2) / 0.6 * 5))
            return round(score, 2)
    except Exception as e:
        print(f"Error computing aesthetics score: {e}")
        return 0.0


def compute_lpips(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute LPIPS (Learned Perceptual Image Patch Similarity)."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex')
        
        # Convert to tensor format (0-1 range, RGB, [1, 3, H, W])
        img1_tensor = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2_tensor = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            lpips_score = loss_fn(img1_tensor, img2_tensor).item()
        
        return round(lpips_score, 4)
    except Exception as e:
        print(f"Warning: Could not compute LPIPS: {e}")
        return 0.0


def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute SSIM (Structural Similarity Index)."""
    try:
        from skimage.metrics import structural_similarity as ssim_func
        
        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        if len(image2.shape) == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        
        # Ensure same size
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        score = ssim_func(image1, image2)
        return round(score, 4)
    except Exception as e:
        print(f"Warning: Could not compute SSIM: {e}")
        return 0.0


def compute_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute PSNR (Peak Signal-to-Noise Ratio)."""
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr_func
        
        # Ensure same size
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        score = psnr_func(image1, image2)
        return round(score, 2)
    except Exception as e:
        print(f"Warning: Could not compute PSNR: {e}")
        return 0.0


def detect_watermark(image_path: str) -> float:
    """Detect watermarks (originality metric). Returns fraction of image that might be watermarked."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        
        # Simple watermark detection: look for text-like patterns in corners
        h, w = img.shape
        corners = [
            img[0:h//4, 0:w//4],  # Top-left
            img[0:h//4, 3*w//4:w],  # Top-right
            img[3*h//4:h, 0:w//4],  # Bottom-left
            img[3*h//4:h, 3*w//4:w],  # Bottom-right
        ]
        
        watermark_score = 0.0
        for corner in corners:
            # Check for high edge density (text-like patterns)
            edges = cv2.Canny(corner, 50, 150)
            edge_density = np.sum(edges > 0) / corner.size
            if edge_density > 0.15:  # Threshold for text-like patterns
                watermark_score += 0.25
        
        return round(watermark_score, 4)
    except Exception as e:
        print(f"Warning: Could not detect watermark: {e}")
        return 0.0


def detect_nsfw(image_path: str) -> float:
    """Detect NSFW content (toxicity metric). Returns probability of NSFW content."""
    try:
        # Simple heuristic: check for skin tone regions and unusual color distributions
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Check for skin-like colors (simple heuristic)
        # Skin tones typically have H in [0, 20] or [160, 180], S in [20, 255], V in [50, 255]
        lower_skin1 = np.array([0, 20, 50])
        upper_skin1 = np.array([20, 255, 255])
        lower_skin2 = np.array([160, 20, 50])
        upper_skin2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = mask1 + mask2
        
        skin_fraction = np.sum(skin_mask > 0) / (img.shape[0] * img.shape[1])
        
        # Very simple heuristic - not a real NSFW detector
        # In production, use a proper NSFW detection model
        nsfw_prob = min(1.0, skin_fraction * 2.0)  # Rough heuristic
        return round(nsfw_prob, 4)
    except Exception as e:
        print(f"Warning: Could not detect NSFW: {e}")
        return 0.0


def detect_gender_imbalance(images_dir: str, segment_name: str) -> float:
    """Detect gender imbalance in segment (bias metric). Returns imbalance score."""
    # This is a placeholder - would need face detection and gender classification
    # For now, return 0 (no detected imbalance)
    return 0.0


def detect_skin_tone_imbalance(images_dir: str, segment_name: str) -> float:
    """Detect skin tone imbalance in segment (bias metric). Returns imbalance score."""
    # This is a placeholder - would need face detection and skin tone classification
    # For now, return 0 (no detected imbalance)
    return 0.0


def technical_quality_score(image_path: str) -> float:
    """Score technical quality (0-5) based on sharpness and artifacts."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    # Laplacian variance for sharpness
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    
    # Normalize: typical good images have variance > 100
    # Poor/blurry images have variance < 50
    sharpness_score = min(5.0, (laplacian_var / 50.0) * 2.5)
    sharpness_score = max(0.0, sharpness_score)
    
    # Artifact proxy: check for extreme blur (very low variance)
    if laplacian_var < 20:
        sharpness_score *= 0.5  # Penalize very blurry images
    
    return round(sharpness_score, 2)


def composition_clarity_score(image_path: str) -> float:
    """Score composition clarity (0-5) based on edge density and contrast."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    # Edge density using Canny
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
    
    # Contrast (standard deviation)
    contrast = np.std(img)
    
    # Normalize
    # Good images typically have edge_density > 0.1 and contrast > 30
    edge_score = min(5.0, (edge_density / 0.1) * 2.5)
    contrast_score = min(5.0, (contrast / 30.0) * 2.5)
    
    # Centeredness proxy: check if main content is roughly centered
    # Simple heuristic: check variance of edge distribution
    h, w = img.shape
    center_y, center_x = h // 2, w // 2
    edge_mask = edges > 0
    if np.any(edge_mask):
        y_coords, x_coords = np.where(edge_mask)
        center_dist = np.sqrt(
            ((y_coords - center_y) / h) ** 2 + ((x_coords - center_x) / w) ** 2
        )
        centeredness = 1.0 - np.mean(center_dist)  # Closer to center = higher
        centeredness = max(0, min(1, centeredness))
    else:
        centeredness = 0.5
    
    composition_score = (edge_score * 0.4 + contrast_score * 0.4 + centeredness * 2.5 * 0.2)
    return round(min(5.0, composition_score), 2)


def color_energy_score(image_path: str) -> float:
    """Score color energy (0-5) based on saturation and colorfulness."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return 0.0
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    
    # Colorfulness metric (standard deviation of color channels)
    b, g, r = cv2.split(img)
    colorfulness = np.std([b, g, r])
    
    # Normalize
    # Good vibrant images have saturation > 100 and colorfulness > 40
    sat_score = min(5.0, (saturation / 100.0) * 2.5)
    color_score = min(5.0, (colorfulness / 40.0) * 2.5)
    
    energy_score = (sat_score * 0.6 + color_score * 0.4)
    return round(min(5.0, energy_score), 2)


def segment_fit_heuristic(
    segment_name: str,
    color_energy: float,
    composition_clarity: float,
) -> float:
    """Fallback heuristic for segment fit when CLIP is unavailable."""
    # Simple heuristic based on segment characteristics
    if segment_name == "gen_z":
        # Gen Z prefers vibrant colors and dynamic composition
        return (color_energy * 0.7 + composition_clarity * 0.3)
    elif segment_name == "parents":
        # Parents prefer balanced, clear composition
        return (composition_clarity * 0.6 + color_energy * 0.4)
    elif segment_name == "professionals":
        # Professionals prefer clarity and moderate color
        return (composition_clarity * 0.7 + min(color_energy, 3.5) * 0.3)
    else:
        return (color_energy + composition_clarity) / 2


def evaluate_image(
    image_path: str,
    segment_name: str,
    segment_data: Dict[str, Any],
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    aesthetics_model,
    aesthetics_preprocess,
    aesthetics_tokenizer,
    device: str,
    reference_image: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate a single image and return scores with HEIM metrics."""
    scores = {}
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Basic scores
    scores["technical_quality"] = technical_quality_score(image_path)
    scores["composition_clarity"] = composition_clarity_score(image_path)
    scores["color_energy"] = color_energy_score(image_path)
    
    # Image-text alignment (CLIP score)
    if clip_model is not None:
        segment_text = build_segment_descriptor(segment_name, segment_data)
        normalized_fit, raw_clip_score = compute_clip_similarity(
            image, segment_text, clip_model, clip_preprocess, clip_tokenizer, device
        )
        scores["segment_fit"] = normalized_fit
        scores["clip_score"] = round(raw_clip_score, 4)  # Raw CLIP score
    else:
        scores["segment_fit"] = segment_fit_heuristic(
            segment_name, scores["color_energy"], scores["composition_clarity"]
        )
        scores["clip_score"] = 0.0
    
    # Aesthetics score
    if aesthetics_model is not None:
        scores["aesthetics_score"] = compute_aesthetics_score(
            image, aesthetics_model, aesthetics_preprocess, aesthetics_tokenizer, device
        )
    else:
        scores["aesthetics_score"] = 0.0
    
    # Image quality metrics (LPIPS, SSIM, PSNR)
    # If we have a reference image, compute these
    if reference_image is not None:
        scores["lpips"] = compute_lpips(image_np, reference_image)
        scores["ssim"] = compute_ssim(image_np, reference_image)
        scores["psnr"] = compute_psnr(image_np, reference_image)
    else:
        # Compute relative metrics within segment (placeholder)
        scores["lpips"] = 0.0
        scores["ssim"] = 0.0
        scores["psnr"] = 0.0
    
    # Originality (watermark detection)
    scores["watermark_frac"] = detect_watermark(image_path)
    
    # Toxicity (NSFW detection)
    scores["nsfw_frac"] = detect_nsfw(image_path)
    
    # Total score (weighted)
    scores["total"] = round(
        0.25 * scores["technical_quality"]
        + 0.20 * scores["composition_clarity"]
        + 0.20 * scores["color_energy"]
        + 0.35 * scores["segment_fit"],
        2,
    )
    
    return scores


def evaluate_all_images(
    segments: Dict[str, Any],
    images_dir: str,
    out_csv: str,
    device: str,
) -> pd.DataFrame:
    """Evaluate all images and save results to CSV with HEIM metrics."""
    print("Loading models...")
    clip_model, clip_preprocess, clip_tokenizer = load_clip_model(device)
    aesthetics_model, aesthetics_preprocess, aesthetics_tokenizer = load_aesthetics_model(device)
    
    all_results = []
    
    for segment_name, segment_data in segments.items():
        segment_dir = os.path.join(images_dir, segment_name)
        if not os.path.exists(segment_dir):
            print(f"Warning: Segment directory not found: {segment_dir}")
            continue
        
        print(f"\nEvaluating images for segment: {segment_name}")
        
        # Find all PNG images
        image_files = sorted(Path(segment_dir).glob("*.png"))
        
        # Compute segment-level metrics
        segment_images = []
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                segment_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Use average image as reference for quality metrics (if available)
        reference_image = None
        if len(segment_images) > 0:
            reference_image = np.mean(segment_images, axis=0).astype(np.uint8)
        
        for image_path in tqdm(image_files, desc=f"  {segment_name}"):
            # Load metadata if available
            metadata_file = image_path.parent / "metadata.jsonl"
            seed = None
            prompt = None
            inference_time = None
            
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    for line in f:
                        meta = json.loads(line)
                        if meta.get("image_path") == str(image_path):
                            seed = meta.get("seed")
                            prompt = meta.get("prompt")
                            # Try to get inference time if stored
                            inference_time = meta.get("inference_time")
                            break
            
            # Evaluate image
            scores = evaluate_image(
                str(image_path),
                segment_name,
                segment_data,
                clip_model,
                clip_preprocess,
                clip_tokenizer,
                aesthetics_model,
                aesthetics_preprocess,
                aesthetics_tokenizer,
                device,
                reference_image=reference_image,
            )
            
            result = {
                "segment": segment_name,
                "image_path": str(image_path),
                "seed": seed,
                "prompt": prompt,
                "technical_quality": scores["technical_quality"],
                "composition_clarity": scores["composition_clarity"],
                "color_energy": scores["color_energy"],
                "segment_fit": scores["segment_fit"],
                "total": scores["total"],
                # HEIM metrics
                "clip_score": scores["clip_score"],
                "aesthetics_score": scores["aesthetics_score"],
                "lpips": scores["lpips"],
                "ssim": scores["ssim"],
                "psnr": scores["psnr"],
                "watermark_frac": scores["watermark_frac"],
                "nsfw_frac": scores["nsfw_frac"],
            }
            
            if inference_time is not None:
                result["inference_time_s"] = inference_time
            
            all_results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Add rank within segment
    df["rank"] = df.groupby("segment")["total"].rank(ascending=False, method="min").astype(int)
    
    # Sort by segment and total score
    df = df.sort_values(["segment", "total"], ascending=[True, False])
    
    # Save to CSV
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    
    print(f"\nEvaluation complete. Results saved to {out_csv}")
    print(f"Total images evaluated: {len(df)}")
    print(f"\nHEIM Metrics Summary:")
    print(f"  Average CLIP Score: {df['clip_score'].mean():.4f}")
    print(f"  Average Aesthetics Score: {df['aesthetics_score'].mean():.2f}")
    print(f"  Average Watermark Fraction: {df['watermark_frac'].mean():.4f}")
    print(f"  Average NSFW Fraction: {df['nsfw_frac'].mean():.4f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate images with HEIM metrics")
    parser.add_argument(
        "--segments",
        type=str,
        default="prompts/segments.json",
        help="Path to segments JSON file",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="images",
        help="Directory containing generated images",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/scores.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage",
    )
    
    args = parser.parse_args()
    
    segments = load_segments(args.segments)
    device = get_device(force_cpu=args.cpu)
    
    evaluate_all_images(
        segments=segments,
        images_dir=args.images_dir,
        out_csv=args.out_csv,
        device=device,
    )


if __name__ == "__main__":
    main()
