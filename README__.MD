# 유튜브 댓글 감정 분석 프로젝트

## 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [데이터 소스](#데이터-소스)
3. [프로젝트 구조](#프로젝트-구조)
4. [요구사항](#요구사항)
5. [분석 계획서](#분석-계획서)

---

## 프로젝트 개요

본 프로젝트는 **유튜브 연애 클립 댓글에 대한 종합적인 감정 분석**을 수행합니다.

### 핵심 목표
- 개별 동영상, 프로그램, 전체 수준의 감정 분석 수행
- 렉시콘 기반 및 딥러닝 기반 분석 결과 비교
- 보고에 첨부하기 위한 댓글의 감정 패턴 시각화

---

## 데이터 소스

### 입력 데이터
- **위치**: `/src` 디렉토리
- **형식**: `.xlsx` 또는 `.csv` 파일
- **내용**: 유튜브 연애 클립의 댓글 데이터

### 데이터 변환
- **도구**: [xlsx2csv](https://github.com/dilshod/xlsx2csv)
- **변환 대상**: `/src` 디렉토리 내의 모든 `.xlsx` 파일
- **출력 형식**: UTF-8 인코딩 CSV 파일

---

## 프로젝트 구조

```
SOS4003_2/
├── src/              # 분석 대상 데이터 (댓글 xlsx/csv 파일)
├── util/             # 공통 유틸리티 함수 (전처리, 파일 변환 등)
├── analysis/         # 감정 분석 스크립트 (렉시콘, 딥러닝)
├── visualization/    # 시각화 및 통계 분석 스크립트
├── output/           # 분석 결과 출력 디렉토리
│   ├── results/      # CSV/JSON 형식 결과
│   ├── figures/      # 시각화 차트 이미지
│   └── reports/      # 최종 분석 보고서
└── CLAUDE.MD         # 본 문서
```

---

## 요구사항

### 1. 분석 범위
분석 결과는 다음 **세 가지 수준 모두**에서 누락 없이 출력되어야 합니다:
- ✅ **개별 동영상별** 분석
- ✅ **프로그램별** 분석
- ✅ **전체 데이터** 분석

### 2. 분석 방법론

#### 2.1 렉시콘 기반 분석
- **KNU 감정 사전**: 한국어 감정 어휘 기반 긍정/부정/중립 분석
- **HateScore**: 혐오 표현 및 공격성 탐지

#### 2.2 딥러닝 기반 분석
- **임베딩 모델**: [KcELECTRA](https://huggingface.co/beomi/KcELECTRA-base-v2022)
- **감정 분류기**: [KoELECTRA](https://huggingface.co/Jinuuuu/KoELECTRA_fine_tunning_emotion) (다중 감정 분류)

### 3. 출력 요구사항
- ✅ 분석 결과의 **시각화**
- ✅ 동영상별 **편차 분석**
- ✅ 정량적/정성적 인사이트 도출

---

# 분석 계획서

## 1. 분석 목표

### 1.1 주요 목표
유튜브 연애 클립 댓글에 대한 종합적인 감정 분석을 수행하여, 동영상별/프로그램별/전체별 감정 패턴과 편차를 파악합니다.

### 1.2 세부 목표
- 렉시콘 기반 분석과 딥러닝 기반 분석의 결과 비교 및 검증
- 동영상 간 감정 편차의 통계적 분석
- 혐오 표현 및 부정적 댓글 패턴 식별
- 데이터 기반 인사이트 도출 및 시각화

---

## 2. 데이터 전처리

### 2.1 파일 변환
- **도구**: [xlsx2csv](https://github.com/dilshod/xlsx2csv)
- **입력**: `/src` 디렉토리의 `.xlsx` 파일
- **출력**: CSV 형식 파일 (UTF-8 인코딩)
- **스크립트 위치**: `/util/xlsx_converter.py`

### 2.2 데이터 정제
- 중복 댓글 제거
- 결측치 처리
- 텍스트 전처리:
  - URL 제거
  - 멘션(@) 처리
  - 이모지 제거
  - 특수문자 정규화
  - 반복 문자 정규화 (ㅋㅋㅋㅋ → ㅋㅋ)
  - 공백 정리
- **전처리 도구**: [Kiwi](https://github.com/bab2min/kiwipiepy) (빠르고 정확한 한국어 형태소 분석기)
- **스크립트 위치**: `/util/text_preprocessor.py`

---

## 3. 감정 분석 방법론

### 3.1 렉시콘 기반 분석

#### KNU 감정 사전
- **GitHub**: [park1200656/KnuSentiLex](https://github.com/park1200656/KnuSentiLex)
- 한국어 감정 어휘 사전 활용
- 긍정/부정/중립 점수 산출
- 감정 강도 측정

#### HateScore
- **GitHub**: [sgunderscore/hatescore-korean-hate-speech](https://github.com/sgunderscore/hatescore-korean-hate-speech)
- 혐오 표현 탐지
- 공격성 수준 측정
- 유해 댓글 필터링

**스크립트 위치**: `/analysis/lexicon_analyzer.py`

### 3.2 딥러닝 기반 분석

#### 임베딩 모델: KcELECTRA
- 모델: [beomi/KcELECTRA-base-v2022](https://huggingface.co/beomi/KcELECTRA-base-v2022)
- 한국어 댓글 특화 사전학습 모델
- 문맥 기반 의미 표현 생성

#### 감정 분류기: KoELECTRA
- 모델: [Jinuuuu/KoELECTRA_fine_tunning_emotion](https://huggingface.co/Jinuuuu/KoELECTRA_fine_tunning_emotion)
- 다중 감정 분류 (기쁨, 슬픔, 분노, 공포, 놀람, 혐오 등)
- 확률 기반 감정 레이블링

**스크립트 위치**: `/analysis/deep_analyzer.py`

---

## 4. 분석 수준 및 결과물

### 4.1 개별 동영상별 분석
- 각 동영상의 댓글 감정 분포
- 렉시콘/딥러닝 기반 감정 점수
- 긍정률, 부정률, 중립률
- 혐오 표현 비율
- 대표 감정 키워드

### 4.2 프로그램별 분석
- 동일 프로그램 내 동영상들의 감정 패턴 비교
- 프로그램별 평균 감정 점수
- 프로그램별 댓글 특성 분석
- 프로그램 간 감정 차이 비교

### 4.3 전체 데이터 분석
- 전체 댓글의 감정 분포 통계
- 전반적인 감정 트렌드
- 이상치 탐지 및 분석

---

## 5. 시각화 계획

### 5.1 감정 분포 시각화
- **파이 차트**: 감정 카테고리 비율
- **막대 그래프**: 동영상별/프로그램별 감정 점수 비교
- **히트맵**: 동영상 간 감정 유사도 매트릭스

### 5.2 편차 분석 시각화
- **박스플롯**: 동영상별 감정 점수 분산
- **바이올린 플롯**: 감정 분포의 밀도
- **산점도**: 렉시콘 vs 딥러닝 기반 점수 상관관계

### 5.3 시계열 분석
- 댓글 시간대별 감정 변화 (데이터에 시간 정보가 있는 경우)
- 트렌드 라인 및 패턴 식별

### 5.4 워드클라우드
- 감정별 주요 키워드 시각화
- 긍정/부정 댓글의 특징적 단어 추출

**스크립트 위치**: `/visualization/visualizer.py`

---

## 6. 통계 분석

### 6.1 기술 통계
- 평균, 중앙값, 표준편차 계산
- 감정 점수 분포 분석
- 동영상별 편차 정량화

### 6.2 이상치 탐지
- Z-score 기반 이상 동영상 탐지
- 특이 패턴을 보이는 댓글 집합 식별
- 이상치에 대한 질적 분석

**스크립트 위치**: `/visualization/stats_analyzer.py`

---

## 7. 구현 계획

### 7.1 디렉토리별 모듈 구성

#### `/util` - 공통 유틸리티
- `xlsx_converter.py`: xlsx → csv 변환
- `text_preprocessor.py`: 텍스트 전처리 함수
- `file_loader.py`: 데이터 로드 및 관리

#### `/analysis` - 감정 분석
- `lexicon_analyzer.py`: KNU, HateScore 기반 분석
- `deep_analyzer.py`: KcELECTRA + KoELECTRA 분석
- `sentiment_aggregator.py`: 분석 결과 통합

#### `/visualization` - 시각화 및 통계
- `visualizer.py`: 차트 및 그래프 생성
- `stats_analyzer.py`: 통계 분석 함수
- `report_generator.py`: 최종 보고서 생성

### 7.2 분석 파이프라인

```
1. 데이터 준비
   └─ xlsx → csv 변환 (util/xlsx_converter.py)
   └─ 데이터 로드 (util/file_loader.py)

2. 전처리
   └─ 텍스트 정제 (util/text_preprocessor.py)

3. 감정 분석
   ├─ 렉시콘 기반 분석 (analysis/lexicon_analyzer.py)
   └─ 딥러닝 기반 분석 (analysis/deep_analyzer.py)

4. 결과 통합
   └─ 분석 결과 병합 (analysis/sentiment_aggregator.py)

5. 시각화 및 통계
   ├─ 차트 생성 (visualization/visualizer.py)
   └─ 통계 분석 (visualization/stats_analyzer.py)

6. 보고서 작성
   └─ 최종 리포트 생성 (visualization/report_generator.py)
```

### 7.3 출력 형식

#### `/output/results`
- `comments_with_sentiment.csv`: 각 댓글별 감정 점수 및 레이블
- `video_summary.json`: 동영상별 통계
- `program_summary.json`: 프로그램별 통계
- `overall_summary.json`: 전체 통계

#### `/output/figures`
- `emotion_distribution_*.png`: 감정 분포 차트
- `video_comparison_*.png`: 동영상별 비교 그래프
- `heatmap_*.png`: 히트맵 시각화
- `wordcloud_*.png`: 워드클라우드

#### `/output/reports`
- `analysis_report.md`: 마크다운 형식 종합 분석 리포트
- `analysis_report.html`: HTML 형식 리포트 (선택사항)

---

## 8. 기술 스택

### 8.1 필수 라이브러리

```python
# 데이터 처리
pandas>=1.5.0
numpy>=1.23.0

# 파일 변환
xlsx2csv

# 텍스트 전처리
kiwipiepy
emoji

# 딥러닝
transformers>=4.30.0
torch>=2.0.0

# 시각화
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
wordcloud>=1.9.0

# 통계 분석
scipy>=1.10.0
```

### 8.2 개발 환경
- Python 3.8 이상
- GPU 권장 (딥러닝 모델 추론 속도 향상)
- 메모리: 최소 8GB RAM

---

## 9. 예상 결과물

### 9.1 정량적 결과
- 전체 댓글 수 및 동영상/프로그램 통계
- 동영상별/프로그램별 평균 감정 점수
- 긍정/부정/중립 비율
- 혐오 표현 비율
- 감정 점수 분산 및 편차

### 9.2 정성적 인사이트
- 특정 프로그램의 감정 패턴 특징
- 이상 동영상에 대한 해석
- 렉시콘 vs 딥러닝 분석 결과 비교 고찰
- 시청자 반응 트렌드 분석

---

## 10. 마일스톤

1. ✅ **환경 설정 및 데이터 변환**
   - 필요 라이브러리 설치
   - xlsx → csv 변환 스크립트 작성

2. ⬜ **전처리 및 렉시콘 기반 분석**
   - 텍스트 전처리 구현
   - KNU, HateScore 기반 분석 수행

3. ⬜ **딥러닝 기반 분석**
   - KcELECTRA 모델 로드
   - 감정 분류 실행

4. ⬜ **결과 통합 및 시각화**
   - 분석 결과 병합
   - 차트 및 그래프 생성

5. ⬜ **통계 분석 및 보고서 작성**
   - 편차 분석 및 통계 검정
   - 최종 분석 보고서 작성
