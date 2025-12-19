"""
HateScore-based hate speech analyzer
Analyzes Korean text for hate speech and toxicity using HateScore model
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


class HateScoreAnalyzer:
    """
    HateScore-based hate speech classifier

    Categories: women, LGBTQ+, men, race/nationality, region, religion, age,
                other hate speech, simple insults, neutral
    """

    def __init__(self, model_name='sgunderscore/hatescore-korean-hate-speech', device=None):
        """
        Initialize HateScore analyzer

        Args:
            model_name: Hugging Face model name
            device: Device to run model on (cuda/cpu)
        """
        print(f"Loading HateScore model: {model_name}")

        # Set device (prioritize MPS for Mac, then CUDA, then CPU)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Hate categories (update based on actual model labels)
        self.hate_labels = ['women', 'lgbtq', 'men', 'race', 'region', 'religion',
                           'age', 'other_hate', 'insult', 'neutral']

        # Try to get labels from model config
        if hasattr(self.model.config, 'id2label'):
            self.hate_labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
            print(f"Hate categories: {self.hate_labels}")

    def analyze_batch(self, texts, max_length=128, threshold=0.5):
        """
        Analyze hate speech for a batch of texts

        Args:
            texts: List of texts to analyze
            max_length: Maximum token length
            threshold: Threshold for positive classification

        Returns:
            List of dictionaries with hate speech results
        """
        if not texts:
            return []

        # Filter out empty texts and track indices
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if pd.notna(text) and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        # Initialize results with default values
        results = [{
            'is_hate': False,
            'hate_score': 0.0,
            'hate_categories': [],
            **{f'prob_{label}': 0.0 for label in self.hate_labels}
        } for _ in range(len(texts))]

        if not valid_texts:
            return results

        try:
            # Tokenize batch
            inputs = self.tokenizer(
                valid_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()  # Multi-label classification

            # Process results
            for i, valid_idx in enumerate(valid_indices):
                # Find None category index
                none_idx = None
                for j, label in enumerate(self.hate_labels):
                    if label == 'None' or 'none' in label.lower():
                        none_idx = j
                        break
                
                # Get None probability
                none_prob = float(probs[i][none_idx]) if none_idx is not None else 0.0
                
                # Calculate hate score (max prob among hate categories, excluding None)
                non_none_indices = [j for j, label in enumerate(self.hate_labels)
                                   if label != 'None' and 'none' not in label.lower()]
                hate_score = float(np.max(probs[i][non_none_indices])) if non_none_indices else 0.0

                if none_prob >= threshold:
                    is_hate = False
                elif hate_score > threshold:
                    is_hate = True
                else:
                    is_hate = False

                # Get hate categories above threshold (only if is_hate)
                hate_categories = []
                if is_hate:
                    hate_categories = [self.hate_labels[j] for j in range(len(self.hate_labels))
                                      if probs[i][j] > threshold and self.hate_labels[j] != 'None'
                                      and 'none' not in self.hate_labels[j].lower()]

                result = {
                    'is_hate': is_hate,
                    'hate_score': hate_score,
                    'hate_categories': ','.join(hate_categories) if hate_categories else '',
                }

                # Add individual probabilities
                for j, label in enumerate(self.hate_labels):
                    result[f'prob_{label}'] = float(probs[i][j])

                results[valid_idx] = result

            return results

        except Exception as e:
            print(f"Error analyzing batch: {e}")
            return results

    def analyze_dataframe(self, df, text_column='Comment Content Clean',
                         prefix='hate_', batch_size=64, max_length=128, threshold=0.5):
        """
        Analyze hate speech for all texts in DataFrame

        Args:
            df: Input DataFrame
            text_column: Column containing text to analyze
            prefix: Prefix for output columns
            batch_size: Batch size for processing
            max_length: Maximum token length
            threshold: Threshold for hate classification

        Returns:
            DataFrame with hate speech analysis columns added
        """
        print(f"\nAnalyzing {len(df)} comments with HateScore...")
        print(f"Batch size: {batch_size}, Max length: {max_length}, Threshold: {threshold}")

        results = []

        # Process in batches for efficiency
        texts = df[text_column].fillna('').tolist()

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+batch_size]

            # Process entire batch at once
            batch_results = self.analyze_batch(batch_texts, max_length=max_length, threshold=threshold)
            results.extend(batch_results)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Add prefix to column names
        results_df.columns = [f'{prefix}{col}' for col in results_df.columns]

        # Combine with original DataFrame
        df_result = pd.concat([df, results_df], axis=1)

        print(f"✅ Hate speech analysis complete")

        return df_result

    def get_video_summary(self, df, video_id_col='Video ID', prefix='hate_'):
        """
        Get hate speech summary for each video

        Args:
            df: DataFrame with hate speech analysis results
            video_id_col: Column name for video IDs
            prefix: Prefix of hate columns

        Returns:
            DataFrame with video-level statistics
        """
        print("\nCalculating video-level statistics...")

        is_hate_col = f'{prefix}is_hate'
        hate_score_col = f'{prefix}hate_score'

        # Group by video
        video_groups = df.groupby(video_id_col)

        summaries = []
        for video_id, group in tqdm(video_groups, desc="Processing videos"):
            total_comments = len(group)
            hate_count = group[is_hate_col].sum()

            summary = {
                'video_id': video_id,
                'total_comments': total_comments,
                'hate_count': int(hate_count),
                'hate_ratio': hate_count / total_comments,
                'avg_hate_score': group[hate_score_col].mean(),
                'max_hate_score': group[hate_score_col].max(),
            }

            # Add average probabilities for each category
            for label in self.hate_labels:
                prob_col = f'{prefix}prob_{label}'
                if prob_col in group.columns:
                    summary[f'avg_prob_{label}'] = group[prob_col].mean()

            # Add program and video title if available
            if 'program' in group.columns:
                summary['program'] = group['program'].iloc[0]
            if 'Video Title' in group.columns:
                summary['video_title'] = group['Video Title'].iloc[0]

            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        print(f"✓ Processed {len(summary_df)} videos")

        return summary_df

    def get_program_summary(self, video_summary_df, program_col='program'):
        """
        Get hate speech summary for each program

        Args:
            video_summary_df: DataFrame with video-level statistics
            program_col: Column name for program names

        Returns:
            DataFrame with program-level statistics
        """
        print("\nCalculating program-level statistics...")

        if program_col not in video_summary_df.columns:
            print(f"Warning: '{program_col}' column not found")
            return None

        program_groups = video_summary_df.groupby(program_col)

        summaries = []
        for program, group in tqdm(program_groups, desc="Processing programs"):
            summary = {
                'program': program,
                'video_count': len(group),
                'total_comments': group['total_comments'].sum(),
                'total_hate_count': int(group['hate_count'].sum()),
                'avg_hate_ratio': group['hate_ratio'].mean(),
                'avg_hate_score': group['avg_hate_score'].mean(),
            }

            # Add average probabilities for each category
            for label in self.hate_labels:
                prob_col = f'avg_prob_{label}'
                if prob_col in group.columns:
                    summary[prob_col] = group[prob_col].mean()

            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        print(f"✓ Processed {len(summary_df)} programs")

        return summary_df

    def get_overall_summary(self, df, prefix='hate_'):
        """
        Get overall hate speech statistics

        Args:
            df: DataFrame with hate speech analysis results
            prefix: Prefix of hate columns

        Returns:
            Dictionary with overall statistics
        """
        print("\nCalculating overall statistics...")

        is_hate_col = f'{prefix}is_hate'
        hate_score_col = f'{prefix}hate_score'

        total_comments = len(df)
        hate_count = df[is_hate_col].sum()

        summary = {
            'total_comments': total_comments,
            'total_videos': df['Video ID'].nunique() if 'Video ID' in df.columns else None,
            'total_programs': df['program'].nunique() if 'program' in df.columns else None,
            'hate_count': int(hate_count),
            'hate_ratio': float(hate_count / total_comments),
            'avg_hate_score': float(df[hate_score_col].mean()),
            'max_hate_score': float(df[hate_score_col].max()),
        }

        # Add average probabilities for each category
        for label in self.hate_labels:
            prob_col = f'{prefix}prob_{label}'
            if prob_col in df.columns:
                summary[f'avg_prob_{label}'] = float(df[prob_col].mean())

        print("✓ Overall statistics calculated")

        return summary


def main():
    """
    Main function for running hate speech analysis
    """
    print("=" * 70)
    print("HateScore Hate Speech Analysis")
    print("=" * 70)
    print()

    # Create output directories if they don't exist
    Path("output/results").mkdir(parents=True, exist_ok=True)

    # File paths
    input_file = 'src/merged_comments_preprocessed.csv'
    output_file = 'output/results/comments_with_hatescore.csv'
    video_summary_file = 'output/results/video_summary_hatescore.csv'
    program_summary_file = 'output/results/program_summary_hatescore.csv'

    # Load preprocessed data
    print(f"Loading data from: {input_file}")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"✓ Loaded {len(df):,} comments")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    print()

    # Initialize analyzer
    analyzer = HateScoreAnalyzer()
    print()

    # Analyze all comments
    df = analyzer.analyze_dataframe(
        df,
        text_column='Comment Content Clean',
        prefix='hate_',
        batch_size=64,
        max_length=128,
        threshold=0.5
    )

    # Get video-level summary
    video_summary = analyzer.get_video_summary(df, video_id_col='Video ID', prefix='hate_')

    # Get program-level summary
    program_summary = analyzer.get_program_summary(video_summary, program_col='program')

    # Get overall summary
    overall_summary = analyzer.get_overall_summary(df, prefix='hate_')

    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    # Save comment-level results
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ Saved comment-level results to: {output_file}")

    # Save video summary
    video_summary.to_csv(video_summary_file, index=False, encoding='utf-8')
    print(f"✓ Saved video summary to: {video_summary_file}")

    # Save program summary
    if program_summary is not None:
        program_summary.to_csv(program_summary_file, index=False, encoding='utf-8')
        print(f"✓ Saved program summary to: {program_summary_file}")

    # Save overall summary as JSON
    import json
    overall_summary_file = 'output/results/overall_summary_hatescore.json'
    with open(overall_summary_file, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved overall summary to: {overall_summary_file}")

    # Print overall statistics
    print("\n" + "=" * 70)
    print("📊 Overall Hate Speech Analysis Results (HateScore)")
    print("=" * 70)
    print(f"Total Comments: {overall_summary['total_comments']:,}")
    print(f"Total Videos: {overall_summary['total_videos']}")
    print(f"Total Programs: {overall_summary['total_programs']}")
    print()
    print(f"Hate Comments: {overall_summary['hate_count']:,} ({overall_summary['hate_ratio']:.1%})")
    print(f"Average Hate Score: {overall_summary['avg_hate_score']:.3f}")
    print(f"Maximum Hate Score: {overall_summary['max_hate_score']:.3f}")
    print()
    print("✅ Analysis complete!")


if __name__ == '__main__':
    main()