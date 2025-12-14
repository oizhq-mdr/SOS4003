"""
Text preprocessor for Korean YouTube comments
Uses Kiwi for morphological analysis and various cleaning techniques
"""

import re
import emoji
import pandas as pd
from kiwipiepy import Kiwi
from tqdm import tqdm


class TextPreprocessor:
    """
    Korean text preprocessor using Kiwi morphological analyzer
    """

    def __init__(self):
        """Initialize Kiwi analyzer"""
        print("Initializing Kiwi morphological analyzer...")
        self.kiwi = Kiwi()
        print("✓ Kiwi initialized")

    @staticmethod
    def remove_url(text):
        """Remove URLs from text"""
        if pd.isna(text):
            return text
        # Remove http/https URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove www URLs
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        return text

    @staticmethod
    def remove_mention(text):
        """Remove @mentions from text"""
        if pd.isna(text):
            return text
        return re.sub(r'@[a-zA-Z0-9_가-힣]+', '', text)

    @staticmethod
    def remove_emoji(text):
        """Remove emojis from text"""
        if pd.isna(text):
            return text
        return emoji.replace_emoji(text, replace='')

    @staticmethod
    def normalize_repeating_chars(text):
        """
        Normalize repeating characters
        Examples: ㅋㅋㅋㅋㅋ -> ㅋㅋ, ㅠㅠㅠㅠ -> ㅠㅠ, 하하하하 -> 하하
        """
        if pd.isna(text):
            return text

        # Normalize repeating Korean characters (ㅋ, ㅎ, ㅠ, etc.)
        text = re.sub(r'([ㅋㅎㅠㅜㅡㅗㅓㅏ])\1{2,}', r'\1\1', text)

        # Normalize repeating syllables (하하하 -> 하하)
        text = re.sub(r'([가-힣])\1{2,}', r'\1\1', text)

        # Normalize repeating punctuation (!!! -> !!, ??? -> ??)
        text = re.sub(r'([!?.])\1{2,}', r'\1\1', text)

        return text

    @staticmethod
    def remove_special_chars(text, keep_punctuation=True):
        """
        Remove special characters
        Args:
            text: Input text
            keep_punctuation: Whether to keep basic punctuation (.,!?)
        """
        if pd.isna(text):
            return text

        if keep_punctuation:
            # Keep Korean, English, numbers, and basic punctuation
            text = re.sub(r'[^가-힣a-zA-Z0-9\s.,!?\'\"-]', '', text)
        else:
            # Keep only Korean, English, numbers, and spaces
            text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)

        return text

    @staticmethod
    def normalize_whitespace(text):
        """Normalize whitespace (remove extra spaces, tabs, newlines)"""
        if pd.isna(text):
            return text

        # Replace tabs and newlines with space
        text = re.sub(r'[\t\n\r]', ' ', text)

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def clean_text(self, text, remove_emojis=True, keep_punctuation=True):
        """
        Apply all cleaning steps to text

        Args:
            text: Input text
            remove_emojis: Whether to remove emojis
            keep_punctuation: Whether to keep punctuation

        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return text

        # Apply cleaning steps in order
        text = self.remove_url(text)
        text = self.remove_mention(text)

        if remove_emojis:
            text = self.remove_emoji(text)

        text = self.normalize_repeating_chars(text)
        text = self.remove_special_chars(text, keep_punctuation)
        text = self.normalize_whitespace(text)

        return text

    def extract_morphemes(self, text, pos_tags=None):
        """
        Extract morphemes from text using Kiwi

        Args:
            text: Input text
            pos_tags: List of POS tags to extract (e.g., ['NNG', 'NNP'] for nouns)
                     If None, returns all morphemes

        Returns:
            List of morphemes
        """
        if pd.isna(text) or not text.strip():
            return []

        try:
            result = self.kiwi.analyze(text)

            if not result:
                return []

            # Get first result
            tokens = result[0][0]

            if pos_tags:
                # Filter by POS tags
                morphemes = [token.form for token in tokens if token.tag in pos_tags]
            else:
                # Return all morphemes
                morphemes = [token.form for token in tokens]

            return morphemes

        except Exception as e:
            print(f"Error in morpheme extraction: {e}")
            return []

    def extract_nouns(self, text):
        """
        Extract nouns from text (for word cloud, etc.)

        Args:
            text: Input text

        Returns:
            List of nouns
        """
        # NNG: 일반명사, NNP: 고유명사
        return self.extract_morphemes(text, pos_tags=['NNG', 'NNP'])

    def extract_pos_tags(self, text):
        """
        Extract morphemes with POS tags for lexicon-based analysis

        Args:
            text: Input text

        Returns:
            List of (morpheme, pos_tag) tuples
        """
        if pd.isna(text) or not text.strip():
            return []

        try:
            result = self.kiwi.analyze(text)

            if not result:
                return []

            # Get first result
            tokens = result[0][0]

            # Return list of (morpheme, POS) tuples
            pos_pairs = [(token.form, token.tag) for token in tokens]

            return pos_pairs

        except Exception as e:
            print(f"Error in POS tagging: {e}")
            return []

    def preprocess_dataframe(self, df, text_column='Comment Content',
                            output_column='Comment Content Clean',
                            remove_emojis=True, keep_punctuation=True,
                            extract_nouns_col=None,
                            extract_pos_col=None):
        """
        Preprocess text in a DataFrame

        Args:
            df: Input DataFrame
            text_column: Name of column containing text to preprocess
            output_column: Name of column for cleaned text
            remove_emojis: Whether to remove emojis
            keep_punctuation: Whether to keep punctuation
            extract_nouns_col: If specified, create this column with extracted nouns
            extract_pos_col: If specified, create this column with POS tags

        Returns:
            DataFrame with cleaned text
        """
        print(f"Preprocessing {len(df)} comments...")

        # Clean text
        tqdm.pandas(desc="Cleaning text")
        df[output_column] = df[text_column].progress_apply(
            lambda x: self.clean_text(x, remove_emojis, keep_punctuation)
        )

        # Extract nouns if requested
        if extract_nouns_col:
            print("Extracting nouns...")
            tqdm.pandas(desc="Extracting nouns")
            df[extract_nouns_col] = df[output_column].progress_apply(
                lambda x: self.extract_nouns(x)
            )

        # Extract POS tags if requested
        if extract_pos_col:
            print("Extracting POS tags (for lexicon analysis)...")
            tqdm.pandas(desc="POS tagging")
            df[extract_pos_col] = df[output_column].progress_apply(
                lambda x: self.extract_pos_tags(x)
            )

        # Remove empty comments after cleaning
        initial_count = len(df)
        df = df[df[output_column].str.strip() != '']
        removed_count = initial_count - len(df)

        if removed_count > 0:
            print(f"Removed {removed_count} empty comments after cleaning")

        print(f"✅ Preprocessing complete: {len(df)} comments")

        return df


def main():
    """
    Main function for testing
    """
    # Test examples
    test_texts = [
        "ㅋㅋㅋㅋㅋㅋㅋ진짜 웃겨ㅠㅠㅠㅠㅠ",
        "이거 보러 가야지!!!!! 😍😍😍",
        "@user123 님 완전 공감해요ㅎㅎㅎ",
        "https://example.com 여기 링크 확인해보세요",
        "하하하하하 너무 재밌다ㅋㅋㅋㅋ",
    ]

    print("=" * 60)
    print("Text Preprocessor Test")
    print("=" * 60)
    print()

    preprocessor = TextPreprocessor()

    print("Testing text cleaning:")
    print("-" * 60)
    for text in test_texts:
        cleaned = preprocessor.clean_text(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print()

    print("\nTesting noun extraction:")
    print("-" * 60)
    test_text = "나는 솔로 프로그램이 정말 재미있어요. 출연자들이 매력적입니다."
    nouns = preprocessor.extract_nouns(test_text)
    print(f"Text: {test_text}")
    print(f"Nouns: {', '.join(nouns)}")


if __name__ == '__main__':
    main()
