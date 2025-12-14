"""
Analysis report generator
Creates comprehensive markdown and HTML reports from analysis results
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class ReportGenerator:
    """
    Generate comprehensive analysis reports
    """

    def __init__(self, output_dir='output/reports'):
        """
        Initialize report generator

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(self):
        """
        Generate comprehensive markdown report
        """
        report_path = self.output_dir / 'analysis_report.md'

        # Load all summary data
        try:
            knu_overall = json.load(open('output/results/overall_summary_knu.json', encoding='utf-8'))
            knu_video = pd.read_csv('output/results/video_summary_knu.csv', encoding='utf-8')
            knu_program = pd.read_csv('output/results/program_summary_knu.csv', encoding='utf-8')
        except:
            knu_overall = knu_video = knu_program = None

        try:
            koelectra_overall = json.load(open('output/results/overall_summary_koelectra.json', encoding='utf-8'))
            koelectra_video = pd.read_csv('output/results/video_summary_koelectra.csv', encoding='utf-8')
            koelectra_program = pd.read_csv('output/results/program_summary_koelectra.csv', encoding='utf-8')
        except:
            koelectra_overall = koelectra_video = koelectra_program = None

        try:
            hate_overall = json.load(open('output/results/overall_summary_hatescore.json', encoding='utf-8'))
            hate_video = pd.read_csv('output/results/video_summary_hatescore.csv', encoding='utf-8')
            hate_program = pd.read_csv('output/results/program_summary_hatescore.csv', encoding='utf-8')
        except:
            hate_overall = hate_video = hate_program = None

        # Build markdown report
        md = []
        md.append("# 유튜브 댓글 감정 분석 종합 보고서\n")
        md.append(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        md.append("---\n")

        # Table of Contents
        md.append("## 목차\n")
        md.append("1. [분석 개요](#분석-개요)\n")
        md.append("2. [KNU 렉시콘 기반 분석](#knu-렉시콘-기반-분석)\n")
        md.append("3. [KoELECTRA 딥러닝 기반 분석](#koelectra-딥러닝-기반-분석)\n")
        md.append("4. [HateScore 혐오 표현 분석](#hatescore-혐오-표현-분석)\n")
        md.append("5. [프로그램별 비교 분석](#프로그램별-비교-분석)\n")
        md.append("6. [주요 인사이트](#주요-인사이트)\n")
        md.append("\n---\n\n")

        # 1. Analysis Overview
        md.append("## 분석 개요\n\n")
        if knu_overall:
            md.append(f"- **전체 댓글 수**: {knu_overall['total_comments']:,}개\n")
            md.append(f"- **동영상 수**: {knu_overall['total_videos']}개\n")
            md.append(f"- **프로그램 수**: {knu_overall['total_programs']}개\n")
        md.append("\n### 분석 방법론\n\n")
        md.append("1. **렉시콘 기반 분석** (KNU 감정 사전)\n")
        md.append("2. **딥러닝 기반 분석** (KoELECTRA 모델)\n")
        md.append("3. **혐오 표현 탐지** (HateScore 모델)\n")
        md.append("\n---\n\n")

        # 2. KNU Lexicon Analysis
        md.append("## KNU 렉시콘 기반 분석\n\n")
        if knu_overall:
            md.append("### 전체 감정 분포\n\n")
            md.append("| 감정 | 개수 | 비율 |\n")
            md.append("|------|------|------|\n")
            md.append(f"| 긍정 | {knu_overall['positive_count']:,} | {knu_overall['positive_ratio']:.1%} |\n")
            md.append(f"| 부정 | {knu_overall['negative_count']:,} | {knu_overall['negative_ratio']:.1%} |\n")
            md.append(f"| 중립 | {knu_overall['neutral_count']:,} | {knu_overall['neutral_ratio']:.1%} |\n")
            md.append("\n")
            md.append("### 평균 감정 점수\n\n")
            md.append(f"- **긍정 점수**: {knu_overall['avg_positive_score']:.3f}\n")
            md.append(f"- **부정 점수**: {knu_overall['avg_negative_score']:.3f}\n")
            md.append(f"- **전체 점수**: {knu_overall['avg_total_score']:.3f} (±{knu_overall['std_total_score']:.3f})\n")
            md.append(f"- **댓글당 매칭 단어 수**: {knu_overall['avg_matched_words']:.1f}개\n")
            md.append("\n")

            if knu_program is not None and len(knu_program) > 0:
                md.append("### 프로그램별 감정 분포 (상위 10개)\n\n")
                md.append("| 프로그램 | 동영상 수 | 댓글 수 | 긍정률 | 부정률 | 중립률 |\n")
                md.append("|----------|-----------|---------|--------|--------|--------|\n")
                top_programs = knu_program.nlargest(10, 'total_comments')
                for _, row in top_programs.iterrows():
                    md.append(f"| {row['program']} | {row['video_count']} | {row['total_comments']:,} | ")
                    md.append(f"{row['avg_positive_ratio']:.1%} | {row['avg_negative_ratio']:.1%} | ")
                    md.append(f"{row['avg_neutral_ratio']:.1%} |\n")
                md.append("\n")

        md.append("\n---\n\n")

        # 3. KoELECTRA Analysis
        md.append("## KoELECTRA 딥러닝 기반 분석\n\n")
        if koelectra_overall:
            md.append("### 전체 감정 분포\n\n")
            md.append(f"**평균 신뢰도**: {koelectra_overall['avg_confidence']:.1%}\n\n")
            md.append("| 감정 | 개수 | 비율 |\n")
            md.append("|------|------|------|\n")

            emotions = ['angry', 'anxious', 'embarrassed', 'happy', 'heartache', 'sad']
            emotion_kr = {
                'angry': '분노',
                'anxious': '불안',
                'embarrassed': '당혹',
                'happy': '기쁨',
                'heartache': '마음아픔',
                'sad': '슬픔'
            }

            for emotion in emotions:
                count = koelectra_overall.get(f'{emotion}_count', 0)
                ratio = koelectra_overall.get(f'{emotion}_ratio', 0)
                kr_name = emotion_kr.get(emotion, emotion)
                md.append(f"| {kr_name} | {count:,} | {ratio:.1%} |\n")
            md.append("\n")

            if koelectra_program is not None and len(koelectra_program) > 0:
                md.append("### 프로그램별 주요 감정 비율 (상위 10개)\n\n")
                md.append("| 프로그램 | 동영상 수 | 댓글 수 | 기쁨 | 분노 | 슬픔 |\n")
                md.append("|----------|-----------|---------|------|------|------|\n")
                top_programs = koelectra_program.nlargest(10, 'total_comments')
                for _, row in top_programs.iterrows():
                    md.append(f"| {row['program']} | {row['video_count']} | {row['total_comments']:,} | ")
                    happy = row.get('avg_happy_ratio', 0)
                    angry = row.get('avg_angry_ratio', 0)
                    sad = row.get('avg_sad_ratio', 0)
                    md.append(f"{happy:.1%} | {angry:.1%} | {sad:.1%} |\n")
                md.append("\n")

        md.append("\n---\n\n")

        # 4. HateScore Analysis
        md.append("## HateScore 혐오 표현 분석\n\n")
        if hate_overall:
            md.append("### 전체 혐오 표현 통계\n\n")
            md.append(f"- **혐오 댓글 수**: {hate_overall['hate_count']:,}개\n")
            md.append(f"- **혐오 댓글 비율**: {hate_overall['hate_ratio']:.1%}\n")
            md.append(f"- **평균 혐오 점수**: {hate_overall['avg_hate_score']:.3f}\n")
            md.append(f"- **최대 혐오 점수**: {hate_overall['max_hate_score']:.3f}\n")
            md.append("\n")

            if hate_program is not None and len(hate_program) > 0:
                md.append("### 프로그램별 혐오 표현 비율 (상위 10개)\n\n")
                md.append("| 프로그램 | 동영상 수 | 댓글 수 | 혐오 댓글 수 | 혐오 비율 |\n")
                md.append("|----------|-----------|---------|--------------|----------|\n")
                top_programs = hate_program.nlargest(10, 'avg_hate_ratio')
                for _, row in top_programs.iterrows():
                    md.append(f"| {row['program']} | {row['video_count']} | {row['total_comments']:,} | ")
                    md.append(f"{row['total_hate_count']:,} | {row['avg_hate_ratio']:.1%} |\n")
                md.append("\n")

        md.append("\n---\n\n")

        # 5. Program Comparison
        md.append("## 프로그램별 비교 분석\n\n")
        md.append("### 종합 비교표\n\n")
        if knu_program is not None and koelectra_program is not None:
            # Merge program summaries
            comparison = knu_program.merge(
                koelectra_program[['program', 'avg_confidence']],
                on='program',
                how='left'
            )

            if hate_program is not None:
                comparison = comparison.merge(
                    hate_program[['program', 'avg_hate_ratio']],
                    on='program',
                    how='left'
                )

            md.append("| 프로그램 | 댓글 수 | KNU 긍정률 | KNU 부정률 | KoELECTRA 신뢰도 | 혐오 비율 |\n")
            md.append("|----------|---------|------------|------------|------------------|----------|\n")

            for _, row in comparison.nlargest(15, 'total_comments').iterrows():
                md.append(f"| {row['program']} | {row['total_comments']:,} | ")
                md.append(f"{row['avg_positive_ratio']:.1%} | {row['avg_negative_ratio']:.1%} | ")
                md.append(f"{row.get('avg_confidence', 0):.1%} | ")
                md.append(f"{row.get('avg_hate_ratio', 0):.1%} |\n")
            md.append("\n")

        md.append("\n---\n\n")

        # 6. Key Insights
        md.append("## 주요 인사이트\n\n")
        md.append("### 렉시콘 기반 분석 인사이트\n\n")
        if knu_overall:
            total_comments = knu_overall['total_comments']
            positive_ratio = knu_overall['positive_ratio']
            negative_ratio = knu_overall['negative_ratio']
            neutral_ratio = knu_overall['neutral_ratio']

            md.append(f"1. 전체 댓글의 **{neutral_ratio:.1%}**가 중립적이며, 이는 ")
            md.append(f"댓글당 평균 {knu_overall['avg_matched_words']:.1f}개의 감정 단어만 매칭되었기 때문입니다.\n\n")

            md.append(f"2. 긍정 댓글({positive_ratio:.1%})이 부정 댓글({negative_ratio:.1%})보다 ")
            md.append(f"약 **{positive_ratio/negative_ratio:.1f}배** 많습니다.\n\n")

        md.append("### 딥러닝 기반 분석 인사이트\n\n")
        if koelectra_overall:
            md.append(f"1. 평균 모델 신뢰도는 **{koelectra_overall['avg_confidence']:.1%}**입니다.\n\n")

            # Find dominant emotion
            emotions = ['angry', 'anxious', 'embarrassed', 'happy', 'heartache', 'sad']
            emotion_ratios = {e: koelectra_overall.get(f'{e}_ratio', 0) for e in emotions}
            dominant = max(emotion_ratios, key=emotion_ratios.get)
            md.append(f"2. 가장 많이 나타난 감정은 **{emotion_kr.get(dominant, dominant)}**")
            md.append(f"({emotion_ratios[dominant]:.1%})입니다.\n\n")

        md.append("### 혐오 표현 분석 인사이트\n\n")
        if hate_overall:
            hate_ratio = hate_overall['hate_ratio']
            md.append(f"1. 전체 댓글 중 **{hate_ratio:.1%}**에서 혐오 표현이 감지되었습니다.\n\n")

            if hate_ratio < 0.05:
                md.append("2. 혐오 표현 비율이 5% 미만으로, 전반적으로 건전한 댓글 문화를 보입니다.\n\n")
            elif hate_ratio < 0.10:
                md.append("2. 혐오 표현 비율이 5-10% 수준으로, 적절한 모니터링이 필요합니다.\n\n")
            else:
                md.append("2. 혐오 표현 비율이 10% 이상으로, 적극적인 관리가 필요합니다.\n\n")

        md.append("\n---\n\n")
        md.append("## 시각화 차트\n\n")
        md.append("생성된 차트는 `output/figures/` 디렉토리에서 확인하실 수 있습니다.\n\n")
        md.append("- KNU 렉시콘 분석 차트\n")
        md.append("- KoELECTRA 감정 분석 차트\n")
        md.append("- HateScore 혐오 표현 분석 차트\n")
        md.append("- 프로그램별 비교 차트\n")
        md.append("\n---\n\n")
        md.append(f"**보고서 생성 완료**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(''.join(md))

        print(f"✓ Markdown report saved to: {report_path}")
        return report_path


def main():
    """
    Main function to generate reports
    """
    print("=" * 70)
    print("Generating Analysis Reports")
    print("=" * 70)
    print()

    generator = ReportGenerator()

    # Generate markdown report
    report_path = generator.generate_markdown_report()

    print("\n" + "=" * 70)
    print("✅ Report generation complete!")
    print(f"Report saved to: {report_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
