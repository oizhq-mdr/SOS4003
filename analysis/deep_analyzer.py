"""
Deep learning-based sentiment analyzer using KoELECTRA
Analyzes Korean text using transformer-based emotion classification
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


class KoELECTRAAnalyzer:
    """
    KoELECTRA-based emotion classifier

    Emotions: 기쁨(joy), 슬픔(sadness), 분노(anger), 공포(fear), 놀람(surprise), 혐오(disgust), 중립(neutral)
    """

    def __init__(self, model_name='Jinuuuu/KoELECTRA_fine_tunning_emotion', device=None):
        """
        Initialize KoELECTRA analyzer

        Args:
            model_name: Hugging Face model name
            device: Device to run model on (cuda/cpu)
        """
        print(f"Loading KoELECTRA model: {model_name}")

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

        # Emotion labels (update based on actual model labels)
        # Common emotion labels for Korean sentiment models
        self.emotion_labels = ['기쁨', '슬픔', '분노', '공포', '놀람', '혐오', '중립']

        # Try to get labels from model config
        if hasattr(self.model.config, 'id2label'):
            self.emotion_labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
            print(f"Emotion labels: {self.emotion_labels}")

    def analyze_batch(self, texts, max_length=64):
        """
        Analyze emotions for a batch of texts (more efficient)

        Args:
            texts: List of texts to analyze
            max_length: Maximum token length

        Returns:
            List of dictionaries with emotion results
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
            'emotion': 'neutral',
            'confidence': 0.0,
            **{f'prob_{label}': 0.0 for label in self.emotion_labels}
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
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # Process results
            for i, valid_idx in enumerate(valid_indices):
                pred_idx = np.argmax(probs[i])
                emotion = self.emotion_labels[pred_idx]
                confidence = float(probs[i][pred_idx])

                result = {
                    'emotion': emotion,
                    'confidence': confidence,
                }

                # Add individual probabilities
                for j, label in enumerate(self.emotion_labels):
                    result[f'prob_{label}'] = float(probs[i][j])

                results[valid_idx] = result

            return results

        except Exception as e:
            print(f"Error analyzing batch: {e}")
            return results

    def analyze_dataframe(self, df, text_column='Comment Content Clean',
                         prefix='koelectra_', batch_size=64, max_length=64):
        """
        Analyze emotions for all texts in DataFrame

        Args:
            df: Input DataFrame
            text_column: Column containing text to analyze
            prefix: Prefix for output columns
            batch_size: Batch size for processing
            max_length: Maximum token length

        Returns:
            DataFrame with emotion analysis columns added
        """
        print(f"\nAnalyzing {len(df)} comments with KoELECTRA...")
        print(f"Batch size: {batch_size}, Max length: {max_length}")

        results = []

        # Process in batches for efficiency
        texts = df[text_column].fillna('').tolist()

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+batch_size]

            # Process entire batch at once
            batch_results = self.analyze_batch(batch_texts, max_length=max_length)
            results.extend(batch_results)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Add prefix to column names
        results_df.columns = [f'{prefix}{col}' for col in results_df.columns]

        # Combine with original DataFrame
        df_result = pd.concat([df, results_df], axis=1)

        print(f"✅ Emotion analysis complete")

        return df_result

    def get_video_summary(self, df, video_id_col='Video ID', prefix='koelectra_'):
        """
        Get emotion summary for each video

        Args:
            df: DataFrame with emotion analysis results
            video_id_col: Column name for video IDs
            prefix: Prefix of emotion columns

        Returns:
            DataFrame with video-level statistics
        """
        print("\nCalculating video-level statistics...")

        emotion_col = f'{prefix}emotion'
        confidence_col = f'{prefix}confidence'

        # Group by video
        video_groups = df.groupby(video_id_col)

        summaries = []
        for video_id, group in tqdm(video_groups, desc="Processing videos"):
            # Count emotions
            emotion_counts = group[emotion_col].value_counts()
            total_comments = len(group)

            summary = {
                'video_id': video_id,
                'total_comments': total_comments,
                'avg_confidence': group[confidence_col].mean(),
            }

            # Add counts and ratios for each emotion
            for emotion in self.emotion_labels:
                count = emotion_counts.get(emotion, 0)
                summary[f'{emotion}_count'] = count
                summary[f'{emotion}_ratio'] = count / total_comments

            # Add average probabilities
            for emotion in self.emotion_labels:
                prob_col = f'{prefix}prob_{emotion}'
                if prob_col in group.columns:
                    summary[f'avg_prob_{emotion}'] = group[prob_col].mean()

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
        Get emotion summary for each program

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
                'avg_confidence': group['avg_confidence'].mean(),
            }

            # Add average ratios for each emotion
            for emotion in self.emotion_labels:
                ratio_col = f'{emotion}_ratio'
                if ratio_col in group.columns:
                    summary[f'avg_{emotion}_ratio'] = group[ratio_col].mean()

            # Add average probabilities
            for emotion in self.emotion_labels:
                prob_col = f'avg_prob_{emotion}'
                if prob_col in group.columns:
                    summary[prob_col] = group[prob_col].mean()

            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        print(f"✓ Processed {len(summary_df)} programs")

        return summary_df

    def get_overall_summary(self, df, prefix='koelectra_'):
        """
        Get overall emotion statistics

        Args:
            df: DataFrame with emotion analysis results
            prefix: Prefix of emotion columns

        Returns:
            Dictionary with overall statistics
        """
        print("\nCalculating overall statistics...")

        emotion_col = f'{prefix}emotion'
        confidence_col = f'{prefix}confidence'

        emotion_counts = df[emotion_col].value_counts()
        total_comments = len(df)

        summary = {
            'total_comments': total_comments,
            'total_videos': df['Video ID'].nunique() if 'Video ID' in df.columns else None,
            'total_programs': df['program'].nunique() if 'program' in df.columns else None,
            'avg_confidence': float(df[confidence_col].mean()),
        }

        # Add counts and ratios for each emotion
        for emotion in self.emotion_labels:
            count = emotion_counts.get(emotion, 0)
            summary[f'{emotion}_count'] = int(count)
            summary[f'{emotion}_ratio'] = float(count / total_comments)

        # Add average probabilities
        for emotion in self.emotion_labels:
            prob_col = f'{prefix}prob_{emotion}'
            if prob_col in df.columns:
                summary[f'avg_prob_{emotion}'] = float(df[prob_col].mean())

        print("✓ Overall statistics calculated")

        return summary


def main():
    """
    Main function for running deep learning-based sentiment analysis
    """
    print("=" * 70)
    print("KoELECTRA Deep Learning-based Emotion Analysis")
    print("=" * 70)
    print()

    # File paths
    input_file = 'src/merged_comments_preprocessed.csv'
    output_file = 'output/results/comments_with_koelectra_emotion.csv'
    video_summary_file = 'output/results/video_summary_koelectra.csv'
    program_summary_file = 'output/results/program_summary_koelectra.csv'

    # Load preprocessed data
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"✓ Loaded {len(df):,} comments")
    print()

    # Initialize analyzer
    analyzer = KoELECTRAAnalyzer()
    print()

    # Analyze all comments
    df = analyzer.analyze_dataframe(
        df,
        text_column='Comment Content Clean',
        prefix='koelectra_',
        batch_size=64,
        max_length=64
    )

    # Get video-level summary
    video_summary = analyzer.get_video_summary(df, video_id_col='Video ID', prefix='koelectra_')

    # Get program-level summary
    program_summary = analyzer.get_program_summary(video_summary, program_col='program')

    # Get overall summary
    overall_summary = analyzer.get_overall_summary(df, prefix='koelectra_')

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
    overall_summary_file = 'output/results/overall_summary_koelectra.json'
    with open(overall_summary_file, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved overall summary to: {overall_summary_file}")

    # Print overall statistics
    print("\n" + "=" * 70)
    print("📊 Overall Emotion Analysis Results (KoELECTRA)")
    print("=" * 70)
    print(f"Total Comments: {overall_summary['total_comments']:,}")
    print(f"Total Videos: {overall_summary['total_videos']}")
    print(f"Total Programs: {overall_summary['total_programs']}")
    print(f"Average Confidence: {overall_summary['avg_confidence']:.1%}")
    print()
    print("Emotion Distribution:")

    for emotion in analyzer.emotion_labels:
        count = overall_summary.get(f'{emotion}_count', 0)
        ratio = overall_summary.get(f'{emotion}_ratio', 0)
        print(f"  {emotion}: {count:,} ({ratio:.1%})")

    print()
    print("✅ Analysis complete!")


if __name__ == '__main__':
    main()
