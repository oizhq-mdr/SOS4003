"""
Run preprocessing on merged comments data
"""

import pandas as pd
import sys
sys.path.append('.')

from util.text_preprocessor import TextPreprocessor


def main():
    # File paths
    input_file = 'src/merged_comments.csv'
    output_file = 'src/merged_comments_preprocessed.csv'

    print("=" * 70)
    print("YouTube Comments Preprocessing")
    print("=" * 70)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"✓ Loaded {len(df):,} comments")
    print()

    # Show data info
    print("Data columns:")
    for col in df.columns:
        print(f"  - {col}")
    print()

    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    print()

    # Preprocess
    df = preprocessor.preprocess_dataframe(
        df,
        text_column='Comment Content',
        output_column='Comment Content Clean',
        remove_emojis=True,
        keep_punctuation=True,
        extract_nouns_col='Nouns',  # Extract nouns for word cloud later
        extract_pos_col='POS_Tags'  # Extract POS tags for lexicon analysis
    )
    print()

    # Show sample results
    print("=" * 70)
    print("Sample Results (first 3 comments)")
    print("=" * 70)
    for idx in range(min(3, len(df))):
        print(f"\n[{idx+1}]")
        print(f"Original:  {df.iloc[idx]['Comment Content']}")
        print(f"Cleaned:   {df.iloc[idx]['Comment Content Clean']}")
        print(f"Nouns:     {', '.join(df.iloc[idx]['Nouns'])}")
        pos_tags = df.iloc[idx]['POS_Tags']
        if len(pos_tags) > 0:
            pos_str = ' '.join([f"{form}/{tag}" for form, tag in pos_tags[:10]])  # Show first 10
            if len(pos_tags) > 10:
                pos_str += " ..."
            print(f"POS Tags:  {pos_str}")
    print()

    # Save preprocessed data
    print("=" * 70)
    print("Saving preprocessed data...")
    df.to_csv(output_file, index=False, encoding='utf-8')

    import os
    file_size = os.path.getsize(output_file) / 1024 / 1024
    print(f"✓ Saved to: {output_file}")
    print(f"  File size: {file_size:.2f} MB")
    print()

    # Show statistics
    print("=" * 70)
    print("Preprocessing Statistics")
    print("=" * 70)
    print(f"Total comments:     {len(df):,}")
    print(f"Programs:           {df['program'].nunique()}")
    print(f"Videos:             {df['Video ID'].nunique()}")
    print(f"Avg original length:    {df['Comment Content'].str.len().mean():.1f} chars")
    print(f"Avg cleaned length:     {df['Comment Content Clean'].str.len().mean():.1f} chars")
    print(f"Avg nouns per comment:  {df['Nouns'].apply(len).mean():.1f}")
    print(f"Avg morphemes per comment: {df['POS_Tags'].apply(len).mean():.1f}")
    print()

    print("✅ Preprocessing complete!")


if __name__ == '__main__':
    main()
