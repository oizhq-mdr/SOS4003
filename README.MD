# 유튜브 연애 프로그램 댓글 감성 분석 프로젝트

## 프로젝트 개요
한국 유튜브 연애 프로그램 36개의 동영상 342개에 달린 총 156,655개의 댓글을 대상으로 감성 분석, 감정 분류, 혐오 표현 분석을 수행한 프로젝트입니다.

# 1. 전처리

## 1.1. 사용 도구
전처리에는 **Kiwipiepy**([GitHub](https://github.com/bab2min/kiwipiepy))를 사용하였습니다. Kiwipiepy는 한국어 형태소 분석을 위한 고성능 라이브러리로, 품사 태깅 및 형태소 추출 기능을 제공합니다.

## 1.2. 전처리 과정
- **URL 제거**: 댓글 내 http/https URL 및 www URL 제거
- **멘션 제거**: @사용자명 형태의 멘션 제거
- **이모지 제거**: emoji 라이브러리를 활용한 이모지 제거
- **반복 문자 정규화**: 'ㅋㅋㅋㅋㅋ → ㅋㅋ', '하하하하 → 하하' 등 과도한 반복 문자 정규화
- **특수 문자 처리**: 한글, 영문, 숫자 및 기본 구두점만 유지
- **공백 정규화**: 연속된 공백, 탭, 개행 문자를 단일 공백으로 변환
- **형태소 분석**: Kiwipiepy를 활용하여 품사 태깅 수행 (일반명사: NNG, 고유명사: NNP 등)

전처리 코드: `util/text_preprocessor.py`

---

# 2. 분석 방법론

본 연구에서는 (1) **감성어 사전 기반 분석**, (2) **BERT 기반 감정 분류**, (3) **HateScore 기반 혐오 표현 분석**을 수행하였습니다.

- **(1)** 렉시콘 기반 분석을 통해 댓글의 일반적인 긍정/부정 여부를 탐색
- **(2)** 딥러닝 기반 감정 분류로 6가지 세분화된 감정 차원 분석
- **(3)** 악플에 초점을 맞춰 10가지 혐오 표현 카테고리 구분

---

## 2.1. KNU 감성어 사전 기반 분석

### 방법론
**KNU 한국어 감성사전**(Korean Sentiment Lexicon)을 활용한 렉시콘 기반 감성 분석입니다. 형태소별로 사전에 정의된 긍정/부정 점수를 합산하여 텍스트의 전체 감성을 계산합니다.

### 사전 정보
- **사전 파일**: `lexicons/SentiWord_Dict.txt`
- **출처**: [GitHub](https://github.com/park1200656/KnuSentiLex)
- **구조**: 형태소와 감성 점수(긍정: 양수, 부정: 음수)의 쌍으로 구성
- **매칭 방식**: Kiwipiepy로 추출한 형태소를 사전과 매칭하여 점수 산출

### 코드 및 결과
- **분석 코드**: `analysis/lexicon_analyzer.py`, `analysis/01_KNU_Lexicon_Analysis.ipynb`
- **결과 파일**: `analysis/01_KNU_Lexicon_Analysis.ipynb`, `output/reports/knu_sentiment_dashboard.html`

---

## 2.2. KoELECTRA 기반 감정 분류

### 방법론
**KoELECTRA**(Korean ELECTRA) 모델을 한국어 감정 분류 데이터셋으로 파인튜닝한 모델을 사용하였습니다. ELECTRA는 BERT의 효율적인 대안으로 제안된 Transformer 기반 사전학습 모델입니다.

### 모델 정보
- **모델명**: `Jinuuuu/KoELECTRA_fine_tunning_emotion`
- **출처**: [Hugging Face](https://huggingface.co/Jinuuuu/KoELECTRA_fine_tunning_emotion)
- **아키텍처**: KoELECTRA + Sequence Classification Head
- **입력**: 최대 64 토큰
- **출력**: 6가지 감정 라벨에 대한 확률 분포

### 감정 분류 카테고리 (6가지)
1. **기쁨(happy)**: 즐거움, 행복, 만족
2. **슬픔(sad)**: 슬픔, 우울, 실망
3. **분노(angry)**: 화남, 짜증, 적대감
4. **불안(anxious)**: 불안, 걱정, 두려움
5. **당황(embarrassed)**: 당황, 놀람, 혼란
6. **마음아픔(heartache)**: 연민, 애틋함, 안타까움

### 코드 및 결과
- **분석 코드**: `analysis/deep_analyzer.py`, `analysis/02_KoELECTRA_Emotion_Analysis.ipynb`
- **결과 파일**: `analysis/02_KoELECTRA_Emotion_Analysis.ipynb`, `output/reports/koelectra_emotion_dashboard.html`

---

## 2.3. HateScore 기반 혐오 표현 분석

### 방법론
**HateScore** 모델은 한국어 혐오 표현 탐지를 위해 개발된 딥러닝 기반 다중 레이블 분류 모델입니다. BERT 계열 모델을 한국어 혐오 표현 데이터셋으로 파인튜닝하여 특정 대상에 대한 혐오 표현을 세분화하여 분류합니다.

### 모델 정보
- **모델명**: `sgunderscore/hatescore-korean-hate-speech`
- **출처**: [Hugging Face](https://huggingface.co/sgunderscore/hatescore-korean-hate-speech)
- **분류 방식**: Multi-label Classification (하나의 댓글이 여러 혐오 카테고리에 동시 속할 수 있음)
- **입력**: 최대 128 토큰
- **출력**: 10가지 카테고리별 확률값

### 혐오 표현 카테고리 (10가지)
1. **None**: 정상 댓글 (혐오 표현 없음)
2. **여성/가족**: 여성 및 가족에 대한 혐오
3. **남성**: 남성에 대한 혐오
4. **성소수자(LGBTQ+)**: 성소수자에 대한 혐오
5. **인종/국적**: 특정 인종이나 국적에 대한 혐오
6. **지역**: 특정 지역에 대한 혐오
7. **종교**: 특정 종교에 대한 혐오
8. **연령**: 특정 연령대에 대한 혐오
9. **기타 혐오**: 위 카테고리에 속하지 않는 기타 혐오 표현
10. **단순 악플**: 혐오 대상이 명확하지 않은 일반 악성 댓글

### 코드 및 결과
- **분석 코드**: `analysis/hatescore_analyzer.py`, `analysis/03_HateScore_Analysis.ipynb`
- **결과 파일**: `analysis/03_HateScore_Analysis.ipynb`, `output/results/hatescore/`