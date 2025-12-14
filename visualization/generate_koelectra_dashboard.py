
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from pathlib import Path

def generate_dashboard():
    print("=" * 70)
    print("Generating HTML Dashboard for KoELECTRA Emotion Analysis (with Trendlines)")
    print("=" * 70)

    # 1. Load Data
    try:
        # Load summaries
        program_path = 'output/results/bert_based_classifier/program_summary_koelectra.csv'
        video_path = 'output/results/bert_based_classifier/video_summary_koelectra.csv'
        program_df = pd.read_csv(program_path, encoding='utf-8')
        video_df = pd.read_csv(video_path, encoding='utf-8')
        print(f"✓ Loaded program summary from {program_path}")
        print(f"✓ Loaded video summary from {video_path}")

        # Load metadata for dates
        meta_path = 'src/merged_comments.csv'
        meta_df = pd.read_csv(meta_path, usecols=['Video ID', 'Video Upload Date'], encoding='utf-8', on_bad_lines='skip')
        print(f"✓ Loaded metadata from {meta_path}")

        # Merge dates
        video_dates = meta_df.drop_duplicates(subset=['Video ID']).rename(columns={
            'Video ID': 'video_id',
            'Video Upload Date': 'upload_date'
        })
        video_dates['upload_date'] = pd.to_datetime(video_dates['upload_date'], errors='coerce')
        
        video_df = video_df.merge(video_dates, on='video_id', how='left')
        print("✓ Merged upload dates")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Setup Plotly & Color Map
    pio.templates.default = "plotly_white"
    
    color_map = {
        'happy': '#FFD700',       # Gold
        'sad': '#4169E1',         # Royal Blue
        'angry': '#DC143C',       # Crimson
        'anxious': '#8B008B',     # Dark Magenta
        'embarrassed': '#FF8C00', # Dark Orange
        'heartache': '#2F4F4F'    # Dark Slate Gray
    }

    figures = []

    # ---------------------------------------------------------
    # Plot 1: Program-wise Emotion Distribution (Stacked Bar)
    # ---------------------------------------------------------
    # Melt data
    emotion_cols = [col for col in program_df.columns if col.endswith('_ratio') and 'avg' in col]
    # Clean column names (avg_angry_ratio -> angry)
    rename_map = {col: col.replace('avg_', '').replace('_ratio', '') for col in emotion_cols}

    program_melted = program_df.melt(
        id_vars=['program'], 
        value_vars=emotion_cols,
        var_name='Emotion', 
        value_name='Ratio'
    )
    program_melted['Emotion'] = program_melted['Emotion'].map(rename_map)

    fig1 = px.bar(
        program_melted, 
        x='program', 
        y='Ratio', 
        color='Emotion',
        title='<b>프로그램별 감정 라벨 분포</b> (Interactive Stacked Bar)',
        color_discrete_map=color_map,
        hover_data={'Ratio': ':.1%'},
        labels={'program': '프로그램', 'Ratio': '감정 비율', 'Emotion': '감정'},
        height=600
    )
    fig1.update_layout(xaxis_tickangle=-45, barmode='stack')
    figures.append(fig1)

    # ---------------------------------------------------------
    # Plot 2: Overall Time-Series Emotion Trends (Scatter with Trendline)
    # ---------------------------------------------------------
    ts_df = video_df.dropna(subset=['upload_date']).sort_values('upload_date')

    # Melt for time series
    video_emotion_cols = [col for col in video_df.columns if col.endswith('_ratio') and not col.startswith('avg')]
    video_rename_map = {col: col.replace('_ratio', '') for col in video_emotion_cols}

    ts_melted = ts_df.melt(
        id_vars=['upload_date', 'video_title', 'program'], 
        value_vars=video_emotion_cols,
        var_name='Emotion', 
        value_name='Score'
    )
    ts_melted['Emotion'] = ts_melted['Emotion'].map(video_rename_map)

    fig2 = px.scatter(
        ts_melted, 
        x='upload_date', 
        y='Score', 
        color='Emotion',
        hover_data=['video_title', 'program'],
        title='<b>전체 동영상 업로드 일자별 감정 추이</b> (All Emotions with Trendlines)',
        color_discrete_map=color_map,
        opacity=0.6,
        labels={'upload_date': '업로드 일자', 'Score': '감정 점수 (비율)', 'Emotion': '감정'},
        height=600
    )
    fig2.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
    fig2.update_xaxes(tickformat="%Y-%m-%d")

    # Add Trendlines for each emotion
    unique_emotions = ts_melted['Emotion'].unique()
    for emotion in unique_emotions:
        e_data = ts_melted[ts_melted['Emotion'] == emotion].sort_values('upload_date')
        if len(e_data) > 1:
            # Rolling average to smooth the line
            window = max(5, int(len(e_data) * 0.05)) # Dynamic window size
            e_data['trend'] = e_data['Score'].rolling(window=window, min_periods=1).mean()
            
            fig2.add_trace(
                go.Scatter(
                    x=e_data['upload_date'],
                    y=e_data['trend'],
                    mode='lines',
                    name=f'{emotion} Trend',
                    line=dict(color=color_map.get(emotion, 'black'), width=2),
                    opacity=0.9,
                    hoverinfo='skip', # Skip hover for trendlines to avoid clutter
                    legendgroup=emotion # Group legend if possible (Scattergl doesn't always group well with Scatter)
                )
            )

    figures.append(fig2)

    # ---------------------------------------------------------
    # Plot 3: Individual Program Plots (Time Series with Trendline)
    # ---------------------------------------------------------
    programs = sorted(video_df['program'].unique())
    program_figures = []

    for prog in programs:
        # Filter for specific program
        prog_data = ts_melted[ts_melted['program'] == prog]
        
        if len(prog_data) < 10: # Skip if very little data
            continue

        fig_p = px.scatter(
            prog_data, 
            x='upload_date', 
            y='Score', 
            color='Emotion',
            hover_data=['video_title'],
            title=f"<b>{prog}</b>: 감정 변화 추이",
            color_discrete_map=color_map,
            opacity=0.7,
            labels={'upload_date': '업로드 일자', 'Score': '감정 점수', 'Emotion': '감정'},
            height=500
        )
        fig_p.update_xaxes(tickformat="%Y-%m-%d")

        # Add Trendlines for each emotion in this program
        possible_emotions = prog_data['Emotion'].unique()
        for emotion in possible_emotions:
            e_data = prog_data[prog_data['Emotion'] == emotion].sort_values('upload_date')
            if len(e_data) > 1:
                # Smaller window for individual programs
                window = max(2, int(len(e_data) * 0.2))
                e_data['trend'] = e_data['Score'].rolling(window=window, min_periods=1).mean()
                
                fig_p.add_trace(
                    go.Scatter(
                        x=e_data['upload_date'],
                        y=e_data['trend'],
                        mode='lines',
                        name=f'{emotion} Trend',
                        line=dict(color=color_map.get(emotion, 'black'), width=2),
                        opacity=0.9,
                        showlegend=False, # Hide legend for individual plots to keep it clean
                        hoverinfo='skip'
                    )
                )

        program_figures.append(fig_p)

    # 4. Save to HTML
    output_dir = Path('output/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'koelectra_emotion_dashboard.html'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('<html><head><title>KoELECTRA Emotion Analysis Dashboard</title></head><body>')
        f.write('<h1 style="text-align: center; margin-top: 20px;">KoELECTRA 감정 분석 대시보드</h1>')
        f.write('<p style="text-align: center;">6가지 세부 감정(기쁨, 슬픔, 분노, 불안, 당황, 상처)의 분포와 추이를 확인하세요.</p>')
        
        f.write('<h2 style="text-align: center; margin-top: 40px;">1. 전체 프로그램 비교 및 추이</h2>')
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('<hr>')

        f.write('<h2 style="text-align: center; margin-top: 40px;">2. 프로그램별 상세 감정 추이</h2>')
        if not program_figures:
            f.write('<p style="text-align: center;">데이터가 충분한 프로그램이 없습니다.</p>')
        
        for fig in program_figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('<hr>')
            
        f.write('</body></html>')

    print(f"\n✅ Dashboard saved to: {output_file}")
    print(f"   Includes {len(program_figures)} individual program plots.")
    print(f"   Open this file in your web browser to view interactive plots.")

if __name__ == "__main__":
    generate_dashboard()
