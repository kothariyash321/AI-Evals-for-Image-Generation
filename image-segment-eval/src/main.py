"""Main orchestration script: generate -> evaluate -> report."""
import argparse
import os
import sys

from src.generate import generate_images
from src.evaluate import evaluate_all_images
from src.report import generate_html_report
from src.utils import load_segments, load_base_prompt, get_device


def main():
    parser = argparse.ArgumentParser(
        description="Generate, evaluate, and report on images for customer segments"
    )
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
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate per segment",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps for generation",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale for generation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model ID",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="images",
        help="Directory for generated images",
    )
    parser.add_argument(
        "--scores_csv",
        type=str,
        default="results/scores.csv",
        help="Output CSV file for scores",
    )
    parser.add_argument(
        "--report_html",
        type=str,
        default="results/report.html",
        help="Output HTML report file",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage",
    )
    parser.add_argument(
        "--skip_generate",
        action="store_true",
        help="Skip image generation (use existing images)",
    )
    parser.add_argument(
        "--skip_evaluate",
        action="store_true",
        help="Skip evaluation (use existing scores CSV)",
    )
    
    args = parser.parse_args()
    
    device = get_device(force_cpu=args.cpu)
    print(f"Using device: {device}")
    
    segments = load_segments(args.segments)
    base_prompt = load_base_prompt(args.base_prompt)
    
    # Step 1: Generate images
    if not args.skip_generate:
        print("\n" + "="*60)
        print("STEP 1: Generating Images")
        print("="*60)
        generate_images(
            segments=segments,
            base_prompt=base_prompt,
            out_dir=args.images_dir,
            num_images=args.num_images,
            steps=args.steps,
            guidance=args.guidance,
            device=device,
            model_id=args.model,
        )
    else:
        print("Skipping image generation (using existing images)")
    
    # Step 2: Evaluate images
    if not args.skip_evaluate:
        print("\n" + "="*60)
        print("STEP 2: Evaluating Images")
        print("="*60)
        evaluate_all_images(
            segments=segments,
            images_dir=args.images_dir,
            out_csv=args.scores_csv,
            device=device,
        )
    else:
        print("Skipping evaluation (using existing scores CSV)")
    
    # Step 3: Generate report
    print("\n" + "="*60)
    print("STEP 3: Generating Report")
    print("="*60)
    generate_html_report(
        scores_csv=args.scores_csv,
        out_html=args.report_html,
    )
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Images: {args.images_dir}/")
    print(f"Scores: {args.scores_csv}")
    print(f"Report: {args.report_html}")
    print("\nOpen the HTML report in your browser to view results.")


if __name__ == "__main__":
    main()

