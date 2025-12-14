"""
Visualization module for sentiment analysis results
Creates charts and graphs for analysis reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set Korean font for matplotlib
plt.rcParams['font.family'] = 'AppleGothic'  # For macOS
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


class SentimentVisualizer:
    """
    Visualizer for sentiment analysis results
    """

    def __init__(self, output_dir='output/figures'):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def plot_sentiment_distribution(self, summary_data, title, filename, sentiment_key='sentiment'):
        """
        Plot sentiment distribution as pie chart

        Args:
            summary_data: Dictionary with sentiment counts
            title: Chart title
            filename: Output filename
            sentiment_key: Key pattern for sentiment data
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract sentiment data
        labels = []
        sizes = []
        colors = []

        color_map = {
            'positive': '#2ecc71', '긍정': '#2ecc71',
            'negative': '#e74c3c', '부정': '#e74c3c',
            'neutral': '#95a5a6', '중립': '#95a5a6',
            'angry': '#e74c3c', 'sad': '#3498db', 'happy': '#f39c12',
            'anxious': '#9b59b6', 'embarrassed': '#e91e63', 'heartache': '#34495e'
        }

        for key, value in summary_data.items():
            if '_count' in key or '_ratio' in key:
                emotion = key.replace('_count', '').replace('_ratio', '')
                if emotion and emotion not in ['hate', 'total', 'video', 'program']:
                    if '_count' in key:
                        labels.append(emotion)
                        sizes.append(value)
                        colors.append(color_map.get(emotion, '#95a5a6'))

        # Create pie chart
        if sizes:
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )

            # Beautify text
            for text in texts:
                text.set_fontsize(12)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)
                autotext.set_weight('bold')

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {filename}")

    def plot_video_comparison(self, video_summary_df, score_column, title, filename,
                              top_n=20, figsize=(14, 8)):
        """
        Plot video-level comparison as bar chart

        Args:
            video_summary_df: DataFrame with video statistics
            score_column: Column name for scores
            title: Chart title
            filename: Output filename
            top_n: Number of top videos to show
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by score and get top N
        df_sorted = video_summary_df.nlargest(top_n, score_column)

        # Create bar chart
        bars = ax.barh(range(len(df_sorted)), df_sorted[score_column])

        # Color bars by value
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_sorted)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Set labels
        ax.set_yticks(range(len(df_sorted)))
        if 'video_title' in df_sorted.columns:
            labels = [f"{row['video_title'][:40]}..." if len(row['video_title']) > 40
                     else row['video_title']
                     for _, row in df_sorted.iterrows()]
        else:
            labels = [f"Video {vid}" for vid in df_sorted['video_id']]

        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(score_column.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {filename}")

    def plot_program_comparison(self, program_summary_df, score_columns, title, filename,
                               figsize=(14, 8)):
        """
        Plot program-level comparison as grouped bar chart

        Args:
            program_summary_df: DataFrame with program statistics
            score_columns: List of column names for scores
            title: Chart title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by first score column
        df_sorted = program_summary_df.sort_values(score_columns[0], ascending=False)

        # Set up bar positions
        x = np.arange(len(df_sorted))
        width = 0.8 / len(score_columns)

        # Create bars for each score column
        for i, col in enumerate(score_columns):
            offset = (i - len(score_columns)/2) * width + width/2
            ax.bar(x + offset, df_sorted[col], width, label=col.replace('_', ' ').title())

        # Set labels
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['program'], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {filename}")

    def plot_emotion_heatmap(self, video_summary_df, emotion_columns, title, filename,
                            figsize=(14, 10)):
        """
        Plot emotion probability heatmap

        Args:
            video_summary_df: DataFrame with video statistics
            emotion_columns: List of emotion probability columns
            title: Chart title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Extract emotion probabilities
        data = video_summary_df[emotion_columns].T

        # Create heatmap
        sns.heatmap(data, cmap='YlOrRd', annot=False, fmt='.2f',
                   cbar_kws={'label': 'Probability'}, ax=ax)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Video Index', fontsize=12)
        ax.set_ylabel('Emotion', fontsize=12)

        # Clean up y-axis labels
        ylabels = [col.replace('avg_prob_', '').replace('_', ' ').title()
                  for col in emotion_columns]
        ax.set_yticklabels(ylabels, rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {filename}")

    def plot_boxplot_comparison(self, video_summary_df, score_column, program_col,
                                title, filename, figsize=(14, 8)):
        """
        Plot boxplot for score distribution across programs

        Args:
            video_summary_df: DataFrame with video statistics
            score_column: Column name for scores
            program_col: Column name for program grouping
            title: Chart title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create boxplot
        programs = sorted(video_summary_df[program_col].unique())
        data_to_plot = [video_summary_df[video_summary_df[program_col] == prog][score_column].values
                       for prog in programs]

        bp = ax.boxplot(data_to_plot, labels=programs, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(programs)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Program', fontsize=12)
        ax.set_ylabel(score_column.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {filename}")


def visualize_knu_results():
    """
    Create visualizations for KNU lexicon analysis results
    """
    print("=" * 70)
    print("Visualizing KNU Lexicon Results")
    print("=" * 70)
    print()

    visualizer = SentimentVisualizer()

    # Load results
    overall_summary = json.load(open('output/results/overall_summary_knu.json', 'r', encoding='utf-8'))
    video_summary = pd.read_csv('output/results/video_summary_knu.csv', encoding='utf-8')
    program_summary = pd.read_csv('output/results/program_summary_knu.csv', encoding='utf-8')

    # 1. Overall sentiment distribution
    visualizer.plot_sentiment_distribution(
        overall_summary,
        'KNU 렉시콘 기반 전체 감정 분포',
        'knu_overall_distribution.png'
    )

    # 2. Top videos by positive ratio
    visualizer.plot_video_comparison(
        video_summary,
        'positive_ratio',
        'KNU 렉시콘: 긍정 비율 상위 20개 동영상',
        'knu_top_positive_videos.png'
    )

    # 3. Top videos by negative ratio
    visualizer.plot_video_comparison(
        video_summary,
        'negative_ratio',
        'KNU 렉시콘: 부정 비율 상위 20개 동영상',
        'knu_top_negative_videos.png'
    )

    # 4. Program comparison
    visualizer.plot_program_comparison(
        program_summary,
        ['avg_positive_ratio', 'avg_negative_ratio', 'avg_neutral_ratio'],
        'KNU 렉시콘: 프로그램별 감정 비율 비교',
        'knu_program_comparison.png'
    )

    # 5. Boxplot: total score distribution by program
    if 'program' in video_summary.columns:
        visualizer.plot_boxplot_comparison(
            video_summary,
            'avg_total_score',
            'program',
            'KNU 렉시콘: 프로그램별 감정 점수 분포',
            'knu_program_score_distribution.png'
        )

    print("\n✅ KNU visualizations complete!")


def visualize_koelectra_results():
    """
    Create visualizations for KoELECTRA analysis results
    """
    print("\n" + "=" * 70)
    print("Visualizing KoELECTRA Results")
    print("=" * 70)
    print()

    visualizer = SentimentVisualizer()

    # Load results
    try:
        overall_summary = json.load(open('output/results/overall_summary_koelectra.json', 'r', encoding='utf-8'))
        video_summary = pd.read_csv('output/results/video_summary_koelectra.csv', encoding='utf-8')
        program_summary = pd.read_csv('output/results/program_summary_koelectra.csv', encoding='utf-8')
    except FileNotFoundError:
        print("KoELECTRA results not found. Skipping visualization.")
        return

    # 1. Overall emotion distribution
    visualizer.plot_sentiment_distribution(
        overall_summary,
        'KoELECTRA 딥러닝 기반 전체 감정 분포',
        'koelectra_overall_distribution.png'
    )

    # 2. Top videos by happiness
    if 'happy_ratio' in video_summary.columns:
        visualizer.plot_video_comparison(
            video_summary,
            'happy_ratio',
            'KoELECTRA: 기쁨 비율 상위 20개 동영상',
            'koelectra_top_happy_videos.png'
        )

    # 3. Top videos by anger
    if 'angry_ratio' in video_summary.columns:
        visualizer.plot_video_comparison(
            video_summary,
            'angry_ratio',
            'KoELECTRA: 분노 비율 상위 20개 동영상',
            'koelectra_top_angry_videos.png'
        )

    # 4. Emotion heatmap (first 50 videos)
    emotion_cols = [col for col in video_summary.columns if col.startswith('avg_prob_')]
    if emotion_cols and len(video_summary) > 0:
        visualizer.plot_emotion_heatmap(
            video_summary.head(50),
            emotion_cols,
            'KoELECTRA: 동영상별 감정 확률 히트맵 (상위 50개)',
            'koelectra_emotion_heatmap.png'
        )

    print("\n✅ KoELECTRA visualizations complete!")


def visualize_hatescore_results():
    """
    Create visualizations for HateScore analysis results
    """
    print("\n" + "=" * 70)
    print("Visualizing HateScore Results")
    print("=" * 70)
    print()

    visualizer = SentimentVisualizer()

    # Load results
    try:
        overall_summary = json.load(open('output/results/overall_summary_hatescore.json', 'r', encoding='utf-8'))
        video_summary = pd.read_csv('output/results/video_summary_hatescore.csv', encoding='utf-8')
        program_summary = pd.read_csv('output/results/program_summary_hatescore.csv', encoding='utf-8')
    except FileNotFoundError:
        print("HateScore results not found. Skipping visualization.")
        return

    # 1. Top videos by hate ratio
    visualizer.plot_video_comparison(
        video_summary,
        'hate_ratio',
        'HateScore: 혐오 표현 비율 상위 20개 동영상',
        'hatescore_top_hate_videos.png'
    )

    # 2. Program comparison
    visualizer.plot_program_comparison(
        program_summary,
        ['avg_hate_ratio'],
        'HateScore: 프로그램별 혐오 표현 비율',
        'hatescore_program_comparison.png'
    )

    # 3. Boxplot: hate score distribution by program
    if 'program' in video_summary.columns:
        visualizer.plot_boxplot_comparison(
            video_summary,
            'avg_hate_score',
            'program',
            'HateScore: 프로그램별 혐오 점수 분포',
            'hatescore_program_distribution.png'
        )

    print("\n✅ HateScore visualizations complete!")


def main():
    """
    Main function to create all visualizations
    """
    print("=" * 70)
    print("Sentiment Analysis Visualization")
    print("=" * 70)
    print()

    # Visualize KNU results
    visualize_knu_results()

    # Visualize KoELECTRA results (if available)
    visualize_koelectra_results()

    # Visualize HateScore results (if available)
    visualize_hatescore_results()

    print("\n" + "=" * 70)
    print("✅ All visualizations complete!")
    print(f"Charts saved to: output/figures/")
    print("=" * 70)


if __name__ == '__main__':
    main()
