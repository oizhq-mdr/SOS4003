import nbformat
import os

notebook_path = '/Users/jaewoolee/Yonsei/SOS4003_2/analysis/02_KoELECTRA_Emotion_Analysis.ipynb'
merged_data_path = '../src/merged_comments.csv'

def update_notebook():
    if not os.path.exists(notebook_path):
        print(f"Notebook not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Find the index to insert new cells (before "2. 전체 감정 분포 시각화")
    insert_idx = -1
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '## 2. 전체 감정 분포 시각화' in cell.source:
            insert_idx = i
            break
    
    if insert_idx == -1:
        # If section not found, append to the end
        insert_idx = len(nb.cells)

    new_cells = []

    # 1. Data Preparation Code Cell
    source_prep = f"""# =========================
# 추가 데이터 로드 및 전처리 (날짜 정보 병합)
# =========================
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 댓글 원본 데이터에서 업로드 날짜 추출
merged_df = pd.read_csv('{merged_data_path}', encoding='utf-8', on_bad_lines='skip')
# Video ID 기준 중복 제거 및 날짜 추출
video_meta = merged_df[['Video ID', 'Video Upload Date']].drop_duplicates(subset=['Video ID'])
video_meta['Video Upload Date'] = pd.to_datetime(video_meta['Video Upload Date'], errors='coerce')

# 기존 video_df에 날짜 병합
# video_df는 이전 셀에서 로드되었다고 가정
if 'video_df' in locals():
    video_df = pd.merge(video_df, video_meta, left_on='video_id', right_on='Video ID', how='left')
    print(f"✓ 날짜 병합 완료: {{len(video_df)}}개 동영상")
else:
    print("⚠ video_df가 로드되지 않았습니다. 이전 셀을 실행해주세요.")
"""
    new_cells.append(nbformat.v4.new_code_cell(source_prep))

    # 2. Markdown Header
    source_md = """## 2. 인터랙티브 감정 분석 (프로그램별 & 시계열)
- **목표**: 개별 프로그램별 감정 라벨 분포 확인 및 시간 흐름에 따른 감정 변화 시각화
- **도구**: Plotly (인터랙티브 그래프)"""
    new_cells.append(nbformat.v4.new_markdown_cell(source_md))

    # 3. Interactive Bar Chart (Program Emotion Distribution)
    source_bar = """# 1. 프로그램별 감정 라벨 분포 (인터랙티브 Stacked Bar Chart)
# 데이터 준비 (Melt)
emotion_cols = [col for col in program_df.columns if col.endswith('_ratio') and 'avg' in col]
# 컬럼명 정리 (avg_angry_ratio -> angry)
rename_map = {col: col.replace('avg_', '').replace('_ratio', '') for col in emotion_cols}

program_melted = program_df.melt(
    id_vars=['program'], 
    value_vars=emotion_cols,
    var_name='Emotion', 
    value_name='Ratio'
)
program_melted['Emotion'] = program_melted['Emotion'].map(rename_map)

# 색상 매핑 정의 (일관성을 위해)
color_map = {
    'happy': '#FFD700',       # 금색
    'sad': '#4169E1',         # 파랑
    'angry': '#DC143C',       # 빨강
    'anxious': '#8B008B',     # 보라
    'embarrassed': '#FF8C00', # 오렌지
    'heartache': '#2F4F4F'    # 진한 회색
}

fig = px.bar(
    program_melted, 
    x='program', 
    y='Ratio', 
    color='Emotion',
    title='프로그램별 감정 분포 비교 (Stacked Bar)',
    color_discrete_map=color_map,
    hover_data={'Ratio': ':.1%'}
)

fig.update_layout(
    xaxis_title='프로그램',
    yaxis_title='감정 비율',
    legend_title_text='감정',
    barmode='stack',
    width=1000,
    height=600
)
fig.show()"""
    new_cells.append(nbformat.v4.new_code_cell(source_bar))

    # 4. Interactive Time Series Plot (All Emotions)
    source_scatter = """# 2. 동영상 업로드 일자별 감정 추이 (인터랙티브 Scatter Plot)
# 데이터 준비
ts_df = video_df.dropna(subset=['Video Upload Date']).sort_values('Video Upload Date')

# 각 감정별 확률/비율을 시계열로 표현하기 위해 Melt
video_emotion_cols = [col for col in video_df.columns if col.endswith('_ratio') and not col.startswith('avg')]
# 컬럼명 예: angry_ratio -> angry
video_rename_map = {col: col.replace('_ratio', '') for col in video_emotion_cols}

ts_melted = ts_df.melt(
    id_vars=['Video Upload Date', 'video_title', 'program'], 
    value_vars=video_emotion_cols,
    var_name='Emotion', 
    value_name='Score'
)
ts_melted['Emotion'] = ts_melted['Emotion'].map(video_rename_map)

fig2 = px.scatter(
    ts_melted, 
    x='Video Upload Date', 
    y='Score', 
    color='Emotion',
    hover_data=['video_title', 'program'],
    title='시간 흐름에 따른 감정 종류별 점수 분포',
    color_discrete_map=color_map,
    opacity=0.6
)

fig2.update_traces(marker=dict(size=6))

fig2.update_layout(
    xaxis_title='동영상 업로드 일자',
    yaxis_title='감정 점수 (비율)',
    legend_title_text='감정',
    width=1000,
    height=600
)
fig2.show()"""
    new_cells.append(nbformat.v4.new_code_cell(source_scatter))

    # Insert cells
    for i, cell in enumerate(new_cells):
        nb.cells.insert(insert_idx + i, cell)

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Successfully inserted {len(new_cells)} cells into {notebook_path}")

if __name__ == "__main__":
    update_notebook()
