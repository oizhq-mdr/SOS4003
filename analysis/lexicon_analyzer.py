"""
Lexicon-based sentiment analyzer using KNU Sentiment Lexicon
Analyzes Korean text using morpheme-based sentiment scoring
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import ast


class KNULexiconAnalyzer:
    """
    KNU Sentiment Lexicon-based analyzer
    """

    def __init__(self, lexicon_path='lexicons/SentiWord_Dict.txt'):
        """
        Initialize lexicon analyzer

        Args:
            lexicon_path: Path to KNU sentiment lexicon file
        """
        print("Loading KNU Sentiment Lexicon...")
        self.lexicon = self._load_lexicon(lexicon_path)
        print(f"✓ Loaded {len(self.lexicon)} sentiment words")

    def _load_lexicon(self, lexicon_path):
        """
        Load KNU sentiment lexicon from file

        Args:
            lexicon_path: Path to lexicon file

        Returns:
            Dictionary mapping words to sentiment scores
        """
        lexicon = {}

        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) == 2:
                    word, score = parts
                    try:
                        lexicon[word] = float(score)
                    except ValueError:
                        continue

        return lexicon

    def analyze_text(self, pos_tags):
        """
        Analyze sentiment of text using POS-tagged morphemes

        Args:
            pos_tags: List of (morpheme, POS) tuples from Kiwi

        Returns:
            Dictionary with sentiment analysis results
        """
        if not pos_tags or not isinstance(pos_tags, list):
            return {
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 0.0,
                'total_score': 0.0,
                'sentiment': 'neutral',
                'matched_words': 0,
                'total_morphemes': 0
            }

        positive_score = 0.0
        negative_score = 0.0
        matched_words = 0

        # Analyze each morpheme
        for morpheme, pos in pos_tags:
            if morpheme in self.lexicon:
                score = self.lexicon[morpheme]
                matched_words += 1

                if score > 0:
                    positive_score += score
                elif score < 0:
                    negative_score += abs(score)

        # Calculate total and neutral scores
        total_score = positive_score - negative_score
        neutral_score = len(pos_tags) - matched_words

        # Determine overall sentiment
        if total_score > 0:
            sentiment = 'positive'
        elif total_score < 0:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'neutral_score': neutral_score,
            'total_score': total_score,
            'sentiment': sentiment,
            'matched_words': matched_words,
            'total_morphemes': len(pos_tags)
        }

    def analyze_dataframe(self, df, pos_column='POS_Tags', prefix='knu_'):
        """
        Analyze sentiment for all comments in DataFrame

        Args:
            df: Input DataFrame with POS-tagged comments
            pos_column: Column name containing POS tags
            prefix: Prefix for output columns

        Returns:
            DataFrame with sentiment analysis columns added
        """
        print(f"\nAnalyzing {len(df)} comments with KNU Lexicon...")

        # Parse POS_Tags if they are stored as strings
        if isinstance(df[pos_column].iloc[0], str):
            print("Parsing POS tags from strings...")
            tqdm.pandas(desc="Parsing POS tags")
            df[pos_column] = df[pos_column].progress_apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) and x.strip() else []
            )

        # Analyze each comment
        tqdm.pandas(desc="Analyzing sentiment")
        results = df[pos_column].progress_apply(lambda x: self.analyze_text(x))

        # Convert results to DataFrame columns
        results_df = pd.DataFrame(results.tolist())

        # Add prefix to column names
        results_df.columns = [f'{prefix}{col}' for col in results_df.columns]

        # Combine with original DataFrame
        df_result = pd.concat([df, results_df], axis=1)

        print(f"✅ Sentiment analysis complete")

        return df_result

    def get_video_summary(self, df, video_id_col='Video ID', prefix='knu_'):
        """
        Get sentiment summary for each video

        Args:
            df: DataFrame with sentiment analysis results
            video_id_col: Column name for video IDs
            prefix: Prefix of sentiment columns

        Returns:
            DataFrame with video-level statistics
        """
        print("\nCalculating video-level statistics...")

        sentiment_col = f'{prefix}sentiment'

        # Group by video
        video_groups = df.groupby(video_id_col)

        summaries = []
        for video_id, group in tqdm(video_groups, desc="Processing videos"):
            # Count sentiments
            sentiment_counts = group[sentiment_col].value_counts()
            total_comments = len(group)

            summary = {
                'video_id': video_id,
                'total_comments': total_comments,
                'positive_count': sentiment_counts.get('positive', 0),
                'negative_count': sentiment_counts.get('negative', 0),
                'neutral_count': sentiment_counts.get('neutral', 0),
                'positive_ratio': sentiment_counts.get('positive', 0) / total_comments,
                'negative_ratio': sentiment_counts.get('negative', 0) / total_comments,
                'neutral_ratio': sentiment_counts.get('neutral', 0) / total_comments,
                'avg_positive_score': group[f'{prefix}positive_score'].mean(),
                'avg_negative_score': group[f'{prefix}negative_score'].mean(),
                'avg_total_score': group[f'{prefix}total_score'].mean(),
                'std_total_score': group[f'{prefix}total_score'].std(),
                'avg_matched_words': group[f'{prefix}matched_words'].mean(),
            }

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
        Get sentiment summary for each program

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
                'avg_positive_ratio': group['positive_ratio'].mean(),
                'avg_negative_ratio': group['negative_ratio'].mean(),
                'avg_neutral_ratio': group['neutral_ratio'].mean(),
                'avg_total_score': group['avg_total_score'].mean(),
                'std_total_score': group['avg_total_score'].std(),
                'avg_matched_words': group['avg_matched_words'].mean(),
            }

            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        print(f"✓ Processed {len(summary_df)} programs")

        return summary_df

    def get_overall_summary(self, df, prefix='knu_'):
        """
        Get overall sentiment statistics

        Args:
            df: DataFrame with sentiment analysis results
            prefix: Prefix of sentiment columns

        Returns:
            Dictionary with overall statistics
        """
        print("\nCalculating overall statistics...")

        sentiment_col = f'{prefix}sentiment'

        sentiment_counts = df[sentiment_col].value_counts()
        total_comments = len(df)

        summary = {
            'total_comments': total_comments,
            'total_videos': df['Video ID'].nunique() if 'Video ID' in df.columns else None,
            'total_programs': df['program'].nunique() if 'program' in df.columns else None,
            'positive_count': int(sentiment_counts.get('positive', 0)),
            'negative_count': int(sentiment_counts.get('negative', 0)),
            'neutral_count': int(sentiment_counts.get('neutral', 0)),
            'positive_ratio': float(sentiment_counts.get('positive', 0) / total_comments),
            'negative_ratio': float(sentiment_counts.get('negative', 0) / total_comments),
            'neutral_ratio': float(sentiment_counts.get('neutral', 0) / total_comments),
            'avg_positive_score': float(df[f'{prefix}positive_score'].mean()),
            'avg_negative_score': float(df[f'{prefix}negative_score'].mean()),
            'avg_total_score': float(df[f'{prefix}total_score'].mean()),
            'std_total_score': float(df[f'{prefix}total_score'].std()),
            'avg_matched_words': float(df[f'{prefix}matched_words'].mean()),
        }

        print("✓ Overall statistics calculated")

        return summary


def main():
    """
    Main function for running lexicon-based sentiment analysis
    """
    print("=" * 70)
    print("KNU Lexicon-based Sentiment Analysis")
    print("=" * 70)
    print()

    # File paths
    input_file = 'src/merged_comments_preprocessed.csv'
    output_file = 'output/results/comments_with_knu_sentiment.csv'
    video_summary_file = 'output/results/video_summary_knu.csv'
    program_summary_file = 'output/results/program_summary_knu.csv'

    # Load preprocessed data
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"✓ Loaded {len(df):,} comments")
    print()

    # Initialize analyzer
    analyzer = KNULexiconAnalyzer()
    print()

    # Analyze all comments
    df = analyzer.analyze_dataframe(df, pos_column='POS_Tags', prefix='knu_')

    # Get video-level summary
    video_summary = analyzer.get_video_summary(df, video_id_col='Video ID', prefix='knu_')

    # Get program-level summary
    program_summary = analyzer.get_program_summary(video_summary, program_col='program')

    # Get overall summary
    overall_summary = analyzer.get_overall_summary(df, prefix='knu_')

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
    overall_summary_file = 'output/results/overall_summary_knu.json'
    with open(overall_summary_file, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved overall summary to: {overall_summary_file}")

    # Print overall statistics
    print("\n" + "=" * 70)
    print("📊 Overall Sentiment Analysis Results (KNU Lexicon)")
    print("=" * 70)
    print(f"Total Comments: {overall_summary['total_comments']:,}")
    print(f"Total Videos: {overall_summary['total_videos']}")
    print(f"Total Programs: {overall_summary['total_programs']}")
    print()
    print("Sentiment Distribution:")
    print(f"  Positive: {overall_summary['positive_count']:,} ({overall_summary['positive_ratio']:.1%})")
    print(f"  Negative: {overall_summary['negative_count']:,} ({overall_summary['negative_ratio']:.1%})")
    print(f"  Neutral:  {overall_summary['neutral_count']:,} ({overall_summary['neutral_ratio']:.1%})")
    print()
    print("Average Scores:")
    print(f"  Positive Score: {overall_summary['avg_positive_score']:.3f}")
    print(f"  Negative Score: {overall_summary['avg_negative_score']:.3f}")
    print(f"  Total Score: {overall_summary['avg_total_score']:.3f} (±{overall_summary['std_total_score']:.3f})")
    print(f"  Matched Words per Comment: {overall_summary['avg_matched_words']:.1f}")
    print()
    print("✅ Analysis complete!")


if __name__ == '__main__':
    main()
