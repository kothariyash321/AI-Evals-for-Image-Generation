# Image Segment Evaluation

A complete pipeline for generating images with Stable Diffusion, evaluating them per customer segment, and generating HTML reports with rankings.

## What This Project Does

This tool:
1. **Generates** images using Stable Diffusion for different customer segments (Gen Z, Parents, Professionals)
2. **Evaluates** each image using a comprehensive scoring rubric inspired by HEIM (Holistic Evaluation of Text-To-Image Models):
   - **Core Metrics**: Technical Quality, Composition Clarity, Color Energy, Segment Fit
   - **HEIM Metrics**: Image-text alignment (CLIP), Aesthetics, Originality (watermark detection), Toxicity (NSFW detection), Efficiency (inference time)
   - **Image Quality Metrics**: LPIPS, SSIM, PSNR
3. **Reports** results in an HTML report with thumbnails, scores, rankings, and HEIM metrics summary

## Quickstart

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** On first run, the system will download:
- Stable Diffusion model (~4GB): `runwayml/stable-diffusion-v1-5`
- CLIP model (~350MB): `ViT-B-32` from OpenAI
- LPIPS model (~233MB): AlexNet weights for perceptual similarity

These downloads happen automatically but may take a few minutes depending on your connection.

### 2. Run End-to-End

```bash
python -m src.main
```

This will:
- Generate 4 images per segment (3 segments = 12 images total)
- Evaluate all images
- Generate an HTML report

### 3. View Results

Open `results/report.html` in your browser to see:
- Top 3 images per segment with score breakdowns
- Full table of all images sorted by total score with HEIM metrics
- HEIM Metrics Summary section with averages, min, and max values
- Thumbnails, prompts, and metadata

## Output Locations

- **Images**: `images/<segment_name>/` (e.g., `images/gen_z/`, `images/parents/`)
- **Scores CSV**: `results/scores.csv` (includes all core metrics + HEIM metrics)
- **HTML Report**: `results/report.html` (includes HEIM metrics summary section)
- **Metadata**: `images/<segment_name>/metadata.jsonl` (JSONL format with prompt, seed, model info, inference time)

### CSV Columns

The `results/scores.csv` file includes:
- **Core metrics**: `technical_quality`, `composition_clarity`, `color_energy`, `segment_fit`, `total`, `rank`
- **HEIM metrics**: `clip_score`, `aesthetics_score`, `watermark_frac`, `nsfw_frac`, `inference_time_s`
- **Image quality metrics**: `lpips`, `ssim`, `psnr` (when reference available)
- **Metadata**: `segment`, `image_path`, `seed`, `prompt`

## Command-Line Options

### Main Pipeline

```bash
python -m src.main [OPTIONS]
```

Options:
- `--segments PATH`: Path to segments JSON (default: `prompts/segments.json`)
- `--base_prompt PATH`: Path to base prompt file (default: `prompts/base_prompt.txt`)
- `--num_images N`: Number of images per segment (default: 4)
- `--steps N`: Inference steps for generation (default: 20)
- `--guidance FLOAT`: Guidance scale (default: 7.5)
- `--model MODEL_ID`: Stable Diffusion model (default: `runwayml/stable-diffusion-v1-5`)
- `--cpu`: Force CPU usage (auto-detects GPU otherwise)
- `--skip_generate`: Skip generation (use existing images)
- `--skip_evaluate`: Skip evaluation (use existing scores CSV)

### Individual Steps

**Generate images only:**
```bash
python -m src.generate --segments prompts/segments.json --base_prompt prompts/base_prompt.txt --out_dir images --num_images 4 --steps 20 --guidance 7.5
```

**Evaluate images only:**
```bash
python -m src.evaluate --segments prompts/segments.json --images_dir images --out_csv results/scores.csv
```

**Generate report only:**
```bash
python -m src.report --scores results/scores.csv --out_html results/report.html
```

## Customization

### Adding New Segments

Edit `prompts/segments.json`:

```json
{
  "your_segment": {
    "description": "Description of your segment...",
    "keywords": ["keyword1", "keyword2"],
    "style_modifiers": ["modifier1", "modifier2"]
  }
}
```

The `style_modifiers` will be appended to the base prompt when generating images.

### Changing the Base Prompt

Edit `prompts/base_prompt.txt` with your desired base prompt. This will be combined with segment-specific style modifiers.

### Adjusting Evaluation Weights

Edit `src/evaluate.py`, in the `evaluate_image()` function:

```python
scores["total"] = round(
    0.25 * scores["technical_quality"]      # Adjust weights here
    + 0.20 * scores["composition_clarity"]
    + 0.20 * scores["color_energy"]
    + 0.35 * scores["segment_fit"],
    2,
)
```

## Device Support

- **GPU (CUDA)**: Automatically detected and used if available
- **CPU**: Falls back automatically, or use `--cpu` flag
- **CPU Mode**: Works but will be slower. Generation may take 1-2 minutes per image on CPU.

## Scoring Rubric Details

### Core Metrics

#### Technical Quality (0-5)
- Based on Laplacian variance (sharpness)
- Penalizes very blurry images
- Higher variance = sharper image

#### Composition Clarity (0-5)
- Edge density (Canny edge detection)
- Contrast (standard deviation of pixel values)
- Centeredness proxy (edge distribution)

#### Color Energy (0-5)
- Saturation (HSV color space)
- Colorfulness (standard deviation across RGB channels)

#### Segment Fit (0-5)
- **Primary**: CLIP similarity between image and segment text descriptor
- **Fallback**: Heuristic based on color energy + composition (if CLIP unavailable)
- Segment descriptor: "An ad image that appeals to <segment>. Keywords: ... Description: ..."

#### Total Score
Weighted combination:
- 25% Technical Quality
- 20% Composition Clarity
- 20% Color Energy
- 35% Segment Fit

### HEIM Metrics (Inspired by Holistic Evaluation of Text-To-Image Models)

#### Image-Text Alignment
- **CLIP Score**: Raw cosine similarity between image and segment descriptor (range: -1 to 1, higher is better)
- Measures how well the generated image aligns with the intended segment characteristics

#### Aesthetics (0-5)
- CLIP-based aesthetics score using multiple aesthetic prompts
- Evaluates visual appeal and professional quality
- Higher scores indicate more aesthetically pleasing images

#### Originality
- **Watermark Fraction**: Detects text-like patterns in image corners (range: 0-1, lower is better)
- Simple heuristic to identify potential watermarks or overlaid text
- Lower values suggest more original content

#### Toxicity
- **NSFW Fraction**: Simple heuristic for inappropriate content risk (range: 0-1, lower is better)
- Based on skin tone detection and color distribution analysis
- **Note**: This is a basic heuristic. For production use, consider dedicated NSFW detection models.

#### Image Quality Metrics
- **LPIPS** (Learned Perceptual Image Patch Similarity): Perceptual similarity metric (lower is better)
- **SSIM** (Structural Similarity Index): Structural similarity (0-1, higher is better)
- **PSNR** (Peak Signal-to-Noise Ratio): Signal quality metric (higher is better)
- **Note**: These metrics use segment average as reference when no ground truth is available

#### Efficiency
- **Inference Time (seconds)**: Time taken to generate each image
- Tracked automatically during generation
- Useful for comparing generation speed across different settings

## Limitations & Next Steps

### Current Limitations

1. **Automated Scoring**: The rubric is a proxy for human judgment. Real-world validation with human evaluators is recommended.
2. **CLIP Model**: Uses a lightweight CLIP model. Larger models may provide better segment fit scores.
3. **Deterministic Seeds**: Uses fixed seeds (42, 1042, 2042, ...) for reproducibility. For more variety, randomize seeds.
4. **Single Model**: Uses one Stable Diffusion model. Different models may produce different results.
5. **HEIM Metrics**: Some metrics use simplified heuristics:
   - NSFW detection is a basic heuristic (not a production-grade detector)
   - Watermark detection looks for text-like patterns in corners
   - Aesthetics uses CLIP-based scoring (not a dedicated aesthetics model)
6. **Reference Images**: LPIPS, SSIM, and PSNR use segment average as reference when no ground truth is available

### Potential Enhancements

1. **Human Evaluation**: Add interface for human evaluators to score images
2. **A/B Testing**: Compare different prompts, models, or generation parameters
3. **Advanced Metrics**: 
   - Add FID (Fréchet Inception Distance) and IS (Inception Score)
   - Implement proper NSFW detection models
   - Add bias detection (gender, skin tone imbalance)
   - Add reasoning and knowledge evaluation metrics
4. **Batch Processing**: Support for larger image sets
5. **Model Comparison**: Generate with multiple models and compare results
6. **Prompt Engineering**: Automated prompt optimization based on scores
7. **Full HEIM Coverage**: Implement all 12 HEIM aspects (currently covers: alignment, quality, aesthetics, originality, toxicity, efficiency)

## Troubleshooting

### CLIP Model Download Fails
The system will fall back to a text-only heuristic for segment fit. This is acceptable but less accurate.

### Out of Memory (GPU)
- Use `--cpu` flag to force CPU mode
- Reduce `--num_images` or `--steps`
- Use a smaller model (e.g., `stabilityai/stable-diffusion-2-1-base`)

### Images Not Appearing in Report
- Check that image paths in CSV are relative to the HTML file location
- Ensure images exist at the specified paths
- Check browser console for 404 errors

### Slow Generation
- Ensure GPU is being used (check device output)
- Reduce `--steps` (20 is a good balance)
- Use fewer images for testing

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list
- ~5.5GB disk space for models (first run):
  - Stable Diffusion: ~4GB
  - CLIP: ~350MB
  - LPIPS: ~233MB
  - Additional dependencies: ~1GB
- GPU recommended but not required (CPU mode supported)

## License

This project uses:
- Stable Diffusion (CreativeML Open RAIL-M License)
- CLIP (MIT License)
- Other dependencies (see their respective licenses)

## Contributing

Feel free to extend this project with:
- Additional HEIM metrics (bias detection, reasoning, knowledge evaluation)
- Support for more image generation models
- Better visualization in reports
- Human evaluation interfaces
- Production-grade NSFW detection
- Advanced bias and fairness metrics

## References

This project is inspired by [HEIM (Holistic Evaluation of Text-To-Image Models)](https://crfm.stanford.edu/heim/), which evaluates text-to-image models across 12 different aspects important for real-world deployment.

