
import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path

def generate_dashboard():
    print("=" * 70)
    print("生成 HTML Dashboard for KNU Sentiment Analysis")
    print("=" * 70)

    # 1. Load Data
    try:
        # Load video summary
        video_path = 'output/results/knu_lexicon/video_summary_knu.csv'
        video_df = pd.read_csv(video_path, encoding='utf-8')
        print(f"✓ Loaded video summary from {video_path}")

        # Load metadata for dates
        meta_path = 'src/merged_comments.csv'
        meta_df = pd.read_csv(meta_path, usecols=['Video ID', 'Video Upload Date'], encoding='utf-8')
        print(f"✓ Loaded metadata from {meta_path}")

        # Merge dates
        video_dates = meta_df.drop_duplicates(subset=['Video ID']).rename(columns={
            'Video ID': 'video_id',
            'Video Upload Date': 'upload_date'
        })
        video_dates['upload_date'] = pd.to_datetime(video_dates['upload_date'])
        
        video_df = video_df.merge(video_dates, on='video_id', how='left')
        print("✓ Merged upload dates")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Setup Plotly
    pio.templates.default = "plotly_white"
    
    # 3. Create Plots
    figures = []
    
    # Plot 1: Dot Plot
    fig1 = px.strip(
        video_df, 
        x="program", 
        y="avg_total_score", 
        color="program", 
        hover_data=["video_title", "total_comments", "upload_date"],
        title="<b>프로그램별 동영상 감정 점수 분포</b> (Interactive Dot Plot)",
        labels={"avg_total_score": "평균 감정 점수 (Sentiment Score)", "program": "프로그램"},
        height=600
    )
    fig1.update_layout(xaxis_tickangle=-45, showlegend=False)
    fig1.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    figures.append(fig1)

    import plotly.graph_objects as go

    # Plot 2: Time Series
    plot_df = video_df.dropna(subset=['upload_date']).sort_values('upload_date')
    
    # Calculate global trend (rolling average with window=10)
    plot_df['trend'] = plot_df['avg_total_score'].rolling(window=10, min_periods=1).mean()

    fig2 = px.scatter(
        plot_df, 
        x="upload_date", 
        y="avg_total_score", 
        color="program", 
        size="total_comments", 
        hover_data=["video_title"],
        title="<b>시간 흐름에 따른 동영상 감정 추이</b> (Time Series with Trendline)",
        labels={
            "upload_date": "업로드 일자 (Upload Date)", 
            "avg_total_score": "평균 감정 점수",
            "total_comments": "댓글 수 (Circle Size)",
            "program": "프로그램"
        },
        height=600
    )
    
    # Add trendline trace
    fig2.add_trace(
        go.Scatter(
            x=plot_df['upload_date'],
            y=plot_df['trend'],
            mode='lines',
            name='전체 추이 (Moving Avg, n=10)',
            line=dict(color='black', width=3, dash='solid')
        )
    )

    fig2.update_xaxes(tickformat="%Y-%m-%d")
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="중립 (0)")
    figures.append(fig2)

    # Plot 3: Individual Program Plots
    programs = sorted(video_df['program'].unique())
    program_figures = []

    for prog in programs:
        prog_df = video_df[video_df['program'] == prog].dropna(subset=['upload_date']).sort_values('upload_date')
        
        if len(prog_df) < 3: # Skip programs with too few videos
            continue
            
        # Calculate trend (rolling avg)
        # Use smaller window for individual programs as they have fewer data points
        window_size = max(3, int(len(prog_df) * 0.2)) 
        prog_df['trend'] = prog_df['avg_total_score'].rolling(window=window_size, min_periods=1).mean()

        fig_p = px.scatter(
            prog_df, 
            x="upload_date", 
            y="avg_total_score", 
            size="total_comments", 
            hover_data=["video_title"],
            title=f"<b>{prog}</b>: 감정 점수 변화 추이",
            labels={
                "upload_date": "업로드 일자", 
                "avg_total_score": "평균 감정 점수",
                "total_comments": "댓글 수"
            },
            height=500
        )

        fig_p.add_trace(
            go.Scatter(
                x=prog_df['upload_date'],
                y=prog_df['trend'],
                mode='lines',
                name=f'추세선 (MA={window_size})',
                line=dict(color='red', width=2, dash='solid')
            )
        )
        
        fig_p.update_xaxes(tickformat="%Y-%m-%d")
        fig_p.add_hline(y=0, line_dash="dash", line_color="gray")
        program_figures.append(fig_p)

    # 4. Save to HTML
    output_dir = Path('output/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'knu_sentiment_dashboard.html'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('<html><head><title>KNU Sentiment Analysis Dashboard</title></head><body>')
        f.write('<h1 style="text-align: center; margin-top: 20px;">KNU 감정 사전 분석 대시보드</h1>')
        f.write('<p style="text-align: center;">인터랙티브 그래프를 통해 데이터를 탐색해보세요.</p>')
        
        f.write('<h2 style="text-align: center; margin-top: 40px;">1. 전체 프로그램 비교</h2>')
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('<hr>')

        f.write('<h2 style="text-align: center; margin-top: 40px;">2. 프로그램별 상세 감정 추이</h2>')
        for fig in program_figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('<hr>')
            
        f.write('</body></html>')

    print(f"\n✅ Dashboard saved to: {output_file}")
    print(f"   Now includes {len(program_figures)} individual program plots.")
    print(f"   Open this file in your web browser to view interactive plots on a large screen.")

if __name__ == "__main__":
    generate_dashboard()
