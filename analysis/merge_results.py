"""
Merge all analysis results into unified files
Combines KNU, KoELECTRA, and HateScore analysis results
"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

print("=" * 70)
print("Merging Analysis Results")
print("=" * 70)
print()

# Create output directory
output_dir = Path("output/results")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Merge Comment-level Results
# ============================================================================
print("1. Merging comment-level results...")
print("-" * 70)

# Load individual analysis results
print("Loading KNU sentiment analysis...")
knu_df = pd.read_csv('output/results/knu_lexicon/comments_with_knu_sentiment.csv', encoding='utf-8')
print(f"✓ Loaded {len(knu_df):,} comments")

print("Loading KoELECTRA emotion analysis...")
koelectra_df = pd.read_csv('output/results/bert_based_classifier/comments_with_koelectra_emotion.csv', encoding='utf-8')
print(f"✓ Loaded {len(koelectra_df):,} comments")

print("Loading HateScore analysis...")
hate_df = pd.read_csv('output/results/hatescore/comments_with_hatescore.csv', encoding='utf-8')
print(f"✓ Loaded {len(hate_df):,} comments")

# Get base columns (columns that are common across all files)
base_columns = [col for col in knu_df.columns if not col.startswith('knu_')]

# Get analysis-specific columns
knu_columns = [col for col in knu_df.columns if col.startswith('knu_')]
koelectra_columns = [col for col in koelectra_df.columns if col.startswith('koelectra_')]
hate_columns = [col for col in hate_df.columns if col.startswith('hate_')]

print(f"\nBase columns: {len(base_columns)}")
print(f"KNU columns: {len(knu_columns)}")
print(f"KoELECTRA columns: {len(koelectra_columns)}")
print(f"HateScore columns: {len(hate_columns)}")

# Merge all results
print("\nMerging dataframes...")
merged_df = knu_df[base_columns + knu_columns].copy()

# Add KoELECTRA columns
for col in koelectra_columns:
    merged_df[col] = koelectra_df[col]

# Add HateScore columns
for col in hate_columns:
    merged_df[col] = hate_df[col]

print(f"✓ Merged dataframe shape: {merged_df.shape}")

# Save merged comments
output_file = output_dir / "comments_full_analysis.csv"
print(f"\nSaving to: {output_file}")
merged_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"✓ Saved {len(merged_df):,} comments with {len(merged_df.columns)} columns")

# ============================================================================
# 2. Merge Video-level Summaries
# ============================================================================
print("\n" + "=" * 70)
print("2. Merging video-level summaries...")
print("-" * 70)

# Check if video summaries exist in root or subdirectories
knu_video_path = 'output/results/knu_lexicon/video_summary_knu.csv'
koelectra_video_path = 'output/results/bert_based_classifier/video_summary_koelectra.csv'
hate_video_path = 'output/results/hatescore/video_summary_hatescore.csv'

# Try to load video summaries
try:
    knu_video = pd.read_csv(knu_video_path, encoding='utf-8')
    koelectra_video = pd.read_csv(koelectra_video_path, encoding='utf-8')
    hate_video = pd.read_csv(hate_video_path, encoding='utf-8')
    
    print(f"KNU videos: {len(knu_video)}")
    print(f"KoELECTRA videos: {len(koelectra_video)}")
    print(f"HateScore videos: {len(hate_video)}")
    
    # Merge on video_id
    merged_video = knu_video.copy()
    
    # Add KoELECTRA columns (excluding common columns)
    koelectra_cols = [col for col in koelectra_video.columns 
                      if col not in ['video_id', 'total_comments', 'program', 'video_title']]
    for col in koelectra_cols:
        merged_video[col] = koelectra_video[col]
    
    # Add HateScore columns (excluding common columns)
    hate_cols = [col for col in hate_video.columns 
                 if col not in ['video_id', 'total_comments', 'program', 'video_title']]
    for col in hate_cols:
        merged_video[col] = hate_video[col]
    
    # Save merged video summary
    output_file = output_dir / "video_summary_full.csv"
    print(f"\nSaving to: {output_file}")
    merged_video.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ Saved {len(merged_video)} videos with {len(merged_video.columns)} columns")
except FileNotFoundError as e:
    print(f"⚠ Video summaries not found, skipping: {e}")

# ============================================================================
# 3. Merge Program-level Summaries
# ============================================================================
print("\n" + "=" * 70)
print("3. Merging program-level summaries...")
print("-" * 70)

# Check if program summaries exist in root or subdirectories
knu_program_path = 'output/results/knu_lexicon/program_summary_knu.csv'
koelectra_program_path = 'output/results/bert_based_classifier/program_summary_koelectra.csv'
hate_program_path = 'output/results/hatescore/program_summary_hatescore.csv'

# Try to load program summaries
try:
    knu_program = pd.read_csv(knu_program_path, encoding='utf-8')
    koelectra_program = pd.read_csv(koelectra_program_path, encoding='utf-8')
    hate_program = pd.read_csv(hate_program_path, encoding='utf-8')

    print(f"KNU programs: {len(knu_program)}")
    print(f"KoELECTRA programs: {len(koelectra_program)}")
    print(f"HateScore programs: {len(hate_program)}")

    # Merge on program
    merged_program = knu_program.copy()

    # Add KoELECTRA columns (excluding common columns)
    koelectra_cols = [col for col in koelectra_program.columns 
                      if col not in ['program', 'video_count', 'total_comments']]
    for col in koelectra_cols:
        merged_program[col] = koelectra_program[col]

    # Add HateScore columns (excluding common columns)
    hate_cols = [col for col in hate_program.columns 
                 if col not in ['program', 'video_count', 'total_comments']]
    for col in hate_cols:
        merged_program[col] = hate_program[col]

    # Save merged program summary
    output_file = output_dir / "program_summary_full.csv"
    print(f"\nSaving to: {output_file}")
    merged_program.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ Saved {len(merged_program)} programs with {len(merged_program.columns)} columns")
except FileNotFoundError as e:
    print(f"⚠ Program summaries not found, skipping: {e}")

# ============================================================================
# 4. Merge Overall Summaries
# ============================================================================
print("\n" + "=" * 70)
print("4. Merging overall summaries...")
print("-" * 70)

# Check if overall summaries exist in root or subdirectories
knu_overall_path = 'output/results/knu_lexicon/overall_summary_knu.json'
koelectra_overall_path = 'output/results/bert_based_classifier/overall_summary_koelectra.json'
hate_overall_path = 'output/results/hatescore/overall_summary_hatescore.json'

# Try to load overall summaries
try:
    with open(knu_overall_path, 'r', encoding='utf-8') as f:
        knu_overall = json.load(f)

    with open(koelectra_overall_path, 'r', encoding='utf-8') as f:
        koelectra_overall = json.load(f)

    with open(hate_overall_path, 'r', encoding='utf-8') as f:
        hate_overall = json.load(f)

    # Merge all summaries
    merged_overall = {
        'total_comments': knu_overall['total_comments'],
        'total_videos': knu_overall['total_videos'],
        'total_programs': knu_overall['total_programs'],
        'knu': knu_overall,
        'koelectra': koelectra_overall,
        'hatescore': hate_overall
    }

    # Save merged overall summary
    output_file = output_dir / "overall_summary_full.json"
    print(f"\nSaving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_overall, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved overall summary")
except FileNotFoundError as e:
    print(f"⚠ Overall summaries not found, skipping: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("📊 Merge Complete!")
print("=" * 70)
print("\nGenerated files:")
print(f"  ✓ comments_full_analysis.csv - {len(merged_df):,} comments")
print(f"  ✓ video_summary_full.csv - {len(merged_video)} videos")
print(f"  ✓ program_summary_full.csv - {len(merged_program)} programs")
print(f"  ✓ overall_summary_full.json - Overall statistics")
print()
print("These files contain all analysis results (KNU + KoELECTRA + HateScore)")
print("and can be used directly for visualization and reporting.")
print()
