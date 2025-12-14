"""
xlsx to csv converter for YouTube comment data
Converts all .xlsx files in the src directory to .csv format with UTF-8 encoding
"""

import os
import sys
from pathlib import Path
from xlsx2csv import Xlsx2csv
from tqdm import tqdm


def convert_xlsx_to_csv(xlsx_path, csv_path=None, encoding='utf-8'):
    """
    Convert a single xlsx file to csv format

    Args:
        xlsx_path: Path to the xlsx file
        csv_path: Path to save the csv file (optional, auto-generated if not provided)
        encoding: Output encoding (default: utf-8)

    Returns:
        Path to the converted csv file
    """
    xlsx_path = Path(xlsx_path)

    if csv_path is None:
        csv_path = xlsx_path.with_suffix('.csv')
    else:
        csv_path = Path(csv_path)

    try:
        # Convert xlsx to csv
        Xlsx2csv(str(xlsx_path), outputencoding=encoding).convert(str(csv_path))
        return csv_path
    except Exception as e:
        print(f"Error converting {xlsx_path}: {e}")
        return None


def convert_all_xlsx_in_directory(src_dir='src', output_dir=None, encoding='utf-8'):
    """
    Convert all xlsx files in a directory to csv format

    Args:
        src_dir: Source directory containing xlsx files
        output_dir: Output directory for csv files (if None, saves in same location as xlsx)
        encoding: Output encoding (default: utf-8)

    Returns:
        List of paths to converted csv files
    """
    src_dir = Path(src_dir)

    if not src_dir.exists():
        print(f"Error: Directory '{src_dir}' does not exist")
        return []

    # Find all xlsx files
    xlsx_files = list(src_dir.glob('*.xlsx'))

    if not xlsx_files:
        print(f"No xlsx files found in '{src_dir}'")
        return []

    print(f"Found {len(xlsx_files)} xlsx file(s) to convert")

    converted_files = []

    # Convert each file with progress bar
    for xlsx_file in tqdm(xlsx_files, desc="Converting files"):
        if output_dir:
            output_path = Path(output_dir) / xlsx_file.with_suffix('.csv').name
        else:
            output_path = None

        csv_file = convert_xlsx_to_csv(xlsx_file, output_path, encoding)

        if csv_file:
            converted_files.append(csv_file)
            print(f"  ✓ Converted: {xlsx_file.name} → {csv_file.name}")

    print(f"\n✅ Conversion complete: {len(converted_files)}/{len(xlsx_files)} files converted")

    return converted_files


def main():
    """
    Main function to run from command line
    Usage: python util/xlsx_converter.py [src_directory] [output_directory]
    """
    # Get source directory from command line or use default
    src_dir = sys.argv[1] if len(sys.argv) > 1 else 'src'

    # Get output directory if specified
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 50)
    print("XLSX to CSV Converter")
    print("=" * 50)
    print(f"Source directory: {src_dir}")
    print(f"Output directory: {output_dir if output_dir else 'Same as source'}")
    print(f"Encoding: UTF-8")
    print("=" * 50)
    print()

    # Convert all files
    converted_files = convert_all_xlsx_in_directory(src_dir, output_dir)

    if converted_files:
        print("\nConverted files:")
        for csv_file in converted_files:
            print(f"  - {csv_file}")


if __name__ == '__main__':
    main()
