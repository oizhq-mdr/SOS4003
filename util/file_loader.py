"""
CSV file loader and merger for YouTube comment data
Loads all CSV files from src directory and merges them into a single DataFrame
"""

import os
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def extract_program_name(filename):
    """
    Extract program name from filename (keep numbers!)

    Examples:
        '나는솔로 1~4.csv' -> '나는솔로 1~4'
        '돌싱글즈3.csv' -> '돌싱글즈3'
        '썸바디2.csv' -> '썸바디2'
        '러브캐처 인 발리.csv' -> '러브캐처 인 발리'

    Args:
        filename: CSV filename

    Returns:
        Program name (filename without .csv extension)
    """
    # Simply remove .csv extension and return
    return filename.replace('.csv', '').strip()


def load_single_csv(csv_path, add_metadata=True):
    """
    Load a single CSV file with optional metadata columns

    Args:
        csv_path: Path to CSV file
        add_metadata: Whether to add program and source_file columns

    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')

        if add_metadata:
            filename = Path(csv_path).name
            program_name = extract_program_name(filename)

            # Add metadata columns
            df['program'] = program_name
            df['source_file'] = filename

        return df

    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def load_all_csv_files(src_dir='src', add_metadata=True):
    """
    Load all CSV files from directory and merge into single DataFrame

    Args:
        src_dir: Directory containing CSV files
        add_metadata: Whether to add program and source_file columns

    Returns:
        Merged DataFrame containing all comments
    """
    src_dir = Path(src_dir)

    if not src_dir.exists():
        print(f"Error: Directory '{src_dir}' does not exist")
        return None

    # Find all CSV files
    csv_files = sorted(src_dir.glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in '{src_dir}'")
        return None

    print(f"Found {len(csv_files)} CSV file(s)")
    print(f"Loading and merging files...")
    print()

    dfs = []
    total_rows = 0

    # Load each file with progress bar
    for csv_file in tqdm(csv_files, desc="Loading files"):
        df = load_single_csv(csv_file, add_metadata)

        if df is not None:
            rows = len(df)
            total_rows += rows
            dfs.append(df)

            if add_metadata:
                program = df['program'].iloc[0]
                print(f"  ✓ {csv_file.name}: {rows:,} rows (프로그램: {program})")

    if not dfs:
        print("No data loaded")
        return None

    # Merge all DataFrames
    print()
    print("Merging all data...")
    merged_df = pd.concat(dfs, ignore_index=True)

    print(f"\n✅ Total: {len(merged_df):,} comments from {len(dfs)} files")

    if add_metadata:
        # Show program statistics
        print(f"\n📊 Programs found:")
        program_counts = merged_df['program'].value_counts()
        for program, count in program_counts.items():
            print(f"  - {program}: {count:,} comments")

    return merged_df


def save_merged_data(df, output_path='output/results/merged_comments.csv'):
    """
    Save merged DataFrame to CSV file

    Args:
        df: DataFrame to save
        output_path: Output file path

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n💾 Saved merged data to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return output_path

    except Exception as e:
        print(f"Error saving file: {e}")
        return None


def get_data_summary(df):
    """
    Get summary statistics of the merged data

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with summary statistics
    """
    # Try to get date range, handle mixed types
    try:
        date_range = (df['Comment Date'].min(), df['Comment Date'].max())
    except:
        date_range = ('N/A', 'N/A')

    summary = {
        'total_comments': len(df),
        'total_programs': df['program'].nunique(),
        'total_videos': df['Video ID'].nunique(),
        'total_authors': df['Comment Author'].nunique(),
        'date_range': date_range,
        'avg_comment_length': df['Comment Content'].str.len().mean(),
        'null_comments': df['Comment Content'].isnull().sum(),
    }

    return summary


def main():
    """
    Main function to run from command line
    Usage: python util/file_loader.py
    """
    print("=" * 60)
    print("CSV File Loader and Merger")
    print("=" * 60)
    print()

    # Load all CSV files
    df = load_all_csv_files('src', add_metadata=True)

    if df is None:
        return

    # Get summary
    print("\n" + "=" * 60)
    print("📈 Data Summary")
    print("=" * 60)
    summary = get_data_summary(df)
    print(f"Total Comments: {summary['total_comments']:,}")
    print(f"Total Programs: {summary['total_programs']}")
    print(f"Total Videos: {summary['total_videos']}")
    print(f"Total Authors: {summary['total_authors']:,}")
    print(f"Null Comments: {summary['null_comments']}")
    print(f"Avg Comment Length: {summary['avg_comment_length']:.1f} characters")

    # Show column info
    print(f"\n📋 Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")

    # Save merged data
    print()
    save_merged_data(df)

    return df


if __name__ == '__main__':
    df = main()
