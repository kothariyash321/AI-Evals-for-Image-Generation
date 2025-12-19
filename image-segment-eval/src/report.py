"""Generate HTML report from evaluation scores."""
import argparse
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd


def generate_html_report(scores_csv: str, out_html: str) -> None:
    """Generate HTML report with thumbnails and scores."""
    df = pd.read_csv(scores_csv)
    
    # Group by segment
    segments = df["segment"].unique()
    
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Segment Evaluation Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 40px;
            border-left: 4px solid #4CAF50;
            padding-left: 15px;
        }
        .segment-section {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .top-images {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }
        .image-card {
            flex: 1;
            min-width: 250px;
            background: #fafafa;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .image-card img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .scores {
            font-size: 0.9em;
            color: #666;
        }
        .score-item {
            margin: 5px 0;
        }
        .score-label {
            font-weight: bold;
            display: inline-block;
            width: 150px;
        }
        .total-score {
            font-size: 1.2em;
            font-weight: bold;
            color: #4CAF50;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 2px solid #ddd;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .rank-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .rank-1 { background-color: #FFD700; color: #333; }
        .rank-2 { background-color: #C0C0C0; color: #333; }
        .rank-3 { background-color: #CD7F32; color: #333; }
        .rank-other { background-color: #e0e0e0; color: #666; }
        .prompt-text {
            font-size: 0.85em;
            color: #666;
            font-style: italic;
            margin-top: 10px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .seed-info {
            font-size: 0.8em;
            color: #999;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>Image Segment Evaluation Report</h1>
    <p>Generated from evaluation scores. Images are ranked by total score within each segment.</p>
""")
    
    # Top images per segment
    for segment in segments:
        segment_df = df[df["segment"] == segment].copy()
        segment_df = segment_df.sort_values("total", ascending=False)
        top_3 = segment_df.head(3)
        
        html_parts.append(f'<div class="segment-section">')
        html_parts.append(f'<h2>Segment: {segment.replace("_", " ").title()}</h2>')
        html_parts.append('<div class="top-images">')
        
        for idx, row in top_3.iterrows():
            image_path = row["image_path"]
            # Convert to relative path for HTML
            rel_path = os.path.relpath(image_path, os.path.dirname(out_html))
            # URL encode the path
            image_url = quote(rel_path.replace(os.sep, "/"))
            
            rank_class = f"rank-{int(row['rank'])}" if row['rank'] <= 3 else "rank-other"
            
            html_parts.append(f"""
            <div class="image-card">
                <img src="{image_url}" alt="Image {row['rank']}" />
                <div class="scores">
                    <div class="score-item">
                        <span class="score-label">Technical Quality:</span>
                        <span>{row['technical_quality']:.2f}/5.0</span>
                    </div>
                    <div class="score-item">
                        <span class="score-label">Composition:</span>
                        <span>{row['composition_clarity']:.2f}/5.0</span>
                    </div>
                    <div class="score-item">
                        <span class="score-label">Color Energy:</span>
                        <span>{row['color_energy']:.2f}/5.0</span>
                    </div>
                    <div class="score-item">
                        <span class="score-label">Segment Fit:</span>
                        <span>{row['segment_fit']:.2f}/5.0</span>
                    </div>""")
            
            # Add HEIM metrics if available
            heim_metrics = []
            if 'clip_score' in row and pd.notna(row.get('clip_score')):
                heim_metrics.append(f'<div class="score-item"><span class="score-label">CLIP Score:</span><span>{row["clip_score"]:.4f}</span></div>')
            if 'aesthetics_score' in row and pd.notna(row.get('aesthetics_score')):
                heim_metrics.append(f'<div class="score-item"><span class="score-label">Aesthetics:</span><span>{row["aesthetics_score"]:.2f}/5.0</span></div>')
            if 'watermark_frac' in row and pd.notna(row.get('watermark_frac')):
                heim_metrics.append(f'<div class="score-item"><span class="score-label">Watermark:</span><span>{row["watermark_frac"]:.4f}</span></div>')
            if 'nsfw_frac' in row and pd.notna(row.get('nsfw_frac')):
                heim_metrics.append(f'<div class="score-item"><span class="score-label">NSFW Risk:</span><span>{row["nsfw_frac"]:.4f}</span></div>')
            if 'inference_time_s' in row and pd.notna(row.get('inference_time_s')):
                heim_metrics.append(f'<div class="score-item"><span class="score-label">Inference Time:</span><span>{row["inference_time_s"]:.2f}s</span></div>')
            
            html_parts.append("".join(heim_metrics))
            
            html_parts.append(f"""
                    <div class="total-score">
                        Total: {row['total']:.2f}/5.0
                        <span class="rank-badge {rank_class}">Rank #{int(row['rank'])}</span>
                    </div>
                    {f'<div class="seed-info">Seed: {row["seed"]}</div>' if pd.notna(row.get("seed")) else ""}
                </div>
                {f'<div class="prompt-text">Prompt: {row["prompt"]}</div>' if pd.notna(row.get("prompt")) else ""}
            </div>
            """)
        
        html_parts.append('</div>')
        html_parts.append('</div>')
    
    # Full table
    html_parts.append('<div class="segment-section">')
    html_parts.append('<h2>All Images (Sorted by Total Score)</h2>')
    html_parts.append('<table>')
    html_parts.append('<thead><tr>')
    html_parts.append('<th>Rank</th>')
    html_parts.append('<th>Segment</th>')
    html_parts.append('<th>Image</th>')
    html_parts.append('<th>Technical</th>')
    html_parts.append('<th>Composition</th>')
    html_parts.append('<th>Color</th>')
    html_parts.append('<th>Segment Fit</th>')
    html_parts.append('<th>CLIP</th>')
    html_parts.append('<th>Aesthetics</th>')
    html_parts.append('<th>Total</th>')
    html_parts.append('</tr></thead>')
    html_parts.append('<tbody>')
    
    # Sort all by total descending
    df_sorted = df.sort_values("total", ascending=False)
    
    for idx, row in df_sorted.iterrows():
        image_path = row["image_path"]
        rel_path = os.path.relpath(image_path, os.path.dirname(out_html))
        image_url = quote(rel_path.replace(os.sep, "/"))
        
        rank_class = f"rank-{int(row['rank'])}" if row['rank'] <= 3 else "rank-other"
        
        html_parts.append('<tr>')
        html_parts.append(f'<td><span class="rank-badge {rank_class}">{int(row["rank"])}</span></td>')
        html_parts.append(f'<td>{row["segment"].replace("_", " ").title()}</td>')
        html_parts.append(f'<td><img src="{image_url}" alt="Thumbnail" style="max-width: 150px; height: auto;" /></td>')
        html_parts.append(f'<td>{row["technical_quality"]:.2f}</td>')
        html_parts.append(f'<td>{row["composition_clarity"]:.2f}</td>')
        html_parts.append(f'<td>{row["color_energy"]:.2f}</td>')
        html_parts.append(f'<td>{row["segment_fit"]:.2f}</td>')
        clip_score = row.get("clip_score", 0.0) if pd.notna(row.get("clip_score")) else 0.0
        html_parts.append(f'<td>{clip_score:.4f}</td>')
        aesthetics = row.get("aesthetics_score", 0.0) if pd.notna(row.get("aesthetics_score")) else 0.0
        html_parts.append(f'<td>{aesthetics:.2f}</td>')
        html_parts.append(f'<td><strong>{row["total"]:.2f}</strong></td>')
        html_parts.append('</tr>')
    
    html_parts.append('</tbody>')
    html_parts.append('</table>')
    html_parts.append('</div>')
    
    # HEIM Metrics Summary Section
    html_parts.append('<div class="segment-section">')
    html_parts.append('<h2>HEIM Metrics Summary</h2>')
    html_parts.append('<table>')
    html_parts.append('<thead><tr>')
    html_parts.append('<th>Metric</th>')
    html_parts.append('<th>Average</th>')
    html_parts.append('<th>Min</th>')
    html_parts.append('<th>Max</th>')
    html_parts.append('<th>Description</th>')
    html_parts.append('</tr></thead>')
    html_parts.append('<tbody>')
    
    heim_metric_info = [
        ("CLIP Score", "clip_score", "Image-text alignment (higher is better)"),
        ("Aesthetics Score", "aesthetics_score", "Visual appeal (0-5, higher is better)"),
        ("Watermark Fraction", "watermark_frac", "Originality indicator (lower is better)"),
        ("NSFW Fraction", "nsfw_frac", "Toxicity risk (lower is better)"),
    ]
    
    if "inference_time_s" in df.columns:
        heim_metric_info.append(("Inference Time (s)", "inference_time_s", "Efficiency metric (lower is better)"))
    
    for metric_name, col_name, description in heim_metric_info:
        if col_name in df.columns:
            col_data = df[col_name].dropna()
            if len(col_data) > 0:
                html_parts.append('<tr>')
                html_parts.append(f'<td><strong>{metric_name}</strong></td>')
                html_parts.append(f'<td>{col_data.mean():.4f}</td>')
                html_parts.append(f'<td>{col_data.min():.4f}</td>')
                html_parts.append(f'<td>{col_data.max():.4f}</td>')
                html_parts.append(f'<td>{description}</td>')
                html_parts.append('</tr>')
    
    html_parts.append('</tbody>')
    html_parts.append('</table>')
    html_parts.append('</div>')
    
    html_parts.append("""
    <div style="margin-top: 40px; padding: 20px; text-align: center; color: #999; font-size: 0.9em;">
        <p>Report generated automatically from image evaluation scores.</p>
    </div>
</body>
</html>
""")
    
    # Write HTML file
    os.makedirs(os.path.dirname(out_html) if os.path.dirname(out_html) else ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
    
    print(f"HTML report generated: {out_html}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from scores")
    parser.add_argument(
        "--scores",
        type=str,
        default="results/scores.csv",
        help="Path to scores CSV file",
    )
    parser.add_argument(
        "--out_html",
        type=str,
        default="results/report.html",
        help="Output HTML file path",
    )
    
    args = parser.parse_args()
    
    generate_html_report(args.scores, args.out_html)


if __name__ == "__main__":
    main()

