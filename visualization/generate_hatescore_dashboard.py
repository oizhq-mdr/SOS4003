
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from pathlib import Path

def generate_dashboard():
    print("=" * 70)
    print("Generating HTML Dashboard for HateScore Hate Speech Analysis (Corrected)")
    print("=" * 70)

    # 1. Load Data
    try:
        # Load corrected summaries (재분류된 데이터)
        # Note: 노트북에서 재계산한 video_df와 program_df를 사용해야 합니다
        # 만약 CSV가 업데이트되지 않았다면, 노트북에서 먼저 실행하세요
        program_path = 'output/results/hatescore/program_summary_hatescore.csv'
        video_path = 'output/results/hatescore/video_summary_hatescore.csv'
        program_df = pd.read_csv(program_path, encoding='utf-8')
        video_df = pd.read_csv(video_path, encoding='utf-8')
        print(f"✓ Loaded program summary from {program_path}")
        print(f"✓ Loaded video summary from {video_path}")
        print("⚠️  Note: 이 데이터는 노트북에서 재분류된 결과여야 합니다")

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
    
    # 혐오 카테고리별 색상 매핑
    category_colors = {
        '여성/가족': '#DC143C',
        '단순 악플': '#FF4500',
        '남성': '#FF6347',
        '성소수자': '#8B0000',
        '인종/국적': '#B22222',
        '연령': '#CD5C5C',
        '지역': '#F08080',
        '종교': '#E9967A',
        '기타 혐오': '#FF7F50'
    }

    figures = []

    # ---------------------------------------------------------
    # Plot 1: Program-wise Hate Category Distribution (Stacked Bar)
    # ---------------------------------------------------------
    # Melt data
    prob_cols = [col for col in program_df.columns if col.startswith('avg_prob_') and col != 'avg_prob_None']
    
    program_melted = program_df.melt(
        id_vars=['program'], 
        value_vars=prob_cols,
        var_name='Category', 
        value_name='Probability'
    )
    program_melted['Category'] = program_melted['Category'].str.replace('avg_prob_', '')

    fig1 = px.bar(
        program_melted, 
        x='program', 
        y='Probability', 
        color='Category',
        title='<b>프로그램별 혐오 카테고리 분포</b> (Interactive Stacked Bar)',
        color_discrete_map=category_colors,
        hover_data={'Probability': ':.3f'},
        labels={'program': '프로그램', 'Probability': '평균 확률', 'Category': '혐오 카테고리'},
        height=600
    )
    fig1.update_layout(
        xaxis_tickangle=-45, 
        barmode='stack',
        legend_title_text='혐오 카테고리',
        width=1200
    )
    figures.append(fig1)

    # ---------------------------------------------------------
    # Plot 2: Overall Time-Series Hate Ratio Trends
    # ---------------------------------------------------------
    ts_df = video_df.dropna(subset=['upload_date']).sort_values('upload_date')

    if len(ts_df) > 0:
        fig2 = px.scatter(
            ts_df, 
            x='upload_date', 
            y='hate_ratio', 
            color='program', 
            size='total_comments',
            hover_data=['video_title', 'avg_hate_score'],
            title='<b>시간 흐름에 따른 혐오 표현 비율 추이</b>',
            labels={
                'upload_date': '업로드 일자',
                'hate_ratio': '혐오 표현 비율',
                'total_comments': '댓글 수',
                'program': '프로그램'
            },
            height=600
        )
        
        # 이동평균선 추가
        ts_df['trend'] = ts_df['hate_ratio'].rolling(window=10, min_periods=1).mean()
        fig2.add_trace(
            go.Scatter(
                x=ts_df['upload_date'],
                y=ts_df['trend'],
                mode='lines',
                name='전체 추세 (MA=10)',
                line=dict(color='black', width=3, dash='solid')
            )
        )
        
        fig2.update_xaxes(tickformat="%Y-%m-%d")
        fig2.add_hline(y=0.05, line_dash="dash", line_color="orange", annotation_text="주의 (5%)")
        fig2.add_hline(y=0.1, line_dash="dash", line_color="red", annotation_text="위험 (10%)")
        figures.append(fig2)

        # 혐오 점수 추이
        fig3 = px.scatter(
            ts_df, 
            x='upload_date', 
            y='avg_hate_score', 
            color='program', 
            size='total_comments',
            hover_data=['video_title', 'hate_ratio'],
            title='<b>시간 흐름에 따른 혐오 점수 추이</b>',
            labels={
                'upload_date': '업로드 일자',
                'avg_hate_score': '평균 혐오 점수',
                'total_comments': '댓글 수',
                'program': '프로그램'
            },
            height=600
        )
        
        # 이동평균선 추가
        ts_df['trend_score'] = ts_df['avg_hate_score'].rolling(window=10, min_periods=1).mean()
        fig3.add_trace(
            go.Scatter(
                x=ts_df['upload_date'],
                y=ts_df['trend_score'],
                mode='lines',
                name='전체 추세 (MA=10)',
                line=dict(color='black', width=3, dash='solid')
            )
        )
        
        fig3.update_xaxes(tickformat="%Y-%m-%d")
        figures.append(fig3)

    # ---------------------------------------------------------
    # Plot 4: Dot Plot - Video Hate Score Distribution
    # ---------------------------------------------------------
    fig4 = px.strip(
        video_df, 
        x="program", 
        y="avg_hate_score", 
        color="program", 
        hover_data=["video_title", "total_comments", "hate_ratio"],
        title="<b>프로그램별 동영상 혐오 점수 분포</b> (Interactive Dot Plot)",
        labels={
            "avg_hate_score": "평균 혐오 점수 (Hate Score)", 
            "program": "프로그램"
        },
        height=600
    )
    fig4.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        title_font_size=20,
        width=1200
    )
    fig4.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    figures.append(fig4)

    # ---------------------------------------------------------
    # Plot 5: Individual Program Plots (Time Series with Trendline)
    # ---------------------------------------------------------
    programs = sorted(video_df['program'].unique())
    program_figures = []

    for prog in programs:
        prog_df = video_df[video_df['program'] == prog].dropna(subset=['upload_date']).sort_values('upload_date')
        
        if len(prog_df) < 3:  # Skip if very little data
            continue
        
        # 이동평균 계산
        window_size = max(3, int(len(prog_df) * 0.2))
        prog_df['trend'] = prog_df['hate_ratio'].rolling(window=window_size, min_periods=1).mean()
        
        fig_p = px.scatter(
            prog_df, 
            x='upload_date', 
            y='hate_ratio', 
            size='total_comments', 
            hover_data=['video_title', 'avg_hate_score'],
            title=f"<b>{prog}</b>: 혐오 표현 비율 변화 추이",
            labels={
                'upload_date': '업로드 일자', 
                'hate_ratio': '혐오 표현 비율',
                'total_comments': '댓글 수'
            },
            height=500
        )
        
        # 추세선 추가
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
        fig_p.add_hline(y=0.05, line_dash="dash", line_color="orange")
        fig_p.add_hline(y=0.1, line_dash="dash", line_color="red")
        program_figures.append(fig_p)

    # ---------------------------------------------------------
    # Plot 6: Individual Hate Category Plots (각 카테고리별 개별 플롯)
    # ---------------------------------------------------------
    # 혐오 카테고리별 색상 매핑
    HATE_CATEGORY_COLORS = {
        'None': '#90EE90',
        '여성/가족': '#DC143C',
        '단순 악플': '#FF4500',
        '남성': '#FF6347',
        '성소수자': '#8B0000',
        '인종/국적': '#B22222',
        '연령': '#CD5C5C',
        '지역': '#F08080',
        '종교': '#E9967A',
        '기타 혐오': '#FF7F50'
    }
    
    hate_categories_list = [cat for cat in HATE_CATEGORY_COLORS.keys() if cat != 'None']
    category_figures = []  # 각 카테고리별 플롯들을 저장
    
    print("\nGenerating individual category plots...")
    for category in hate_categories_list:
        prob_col = f'avg_prob_{category}'
        category_color = HATE_CATEGORY_COLORS.get(category, '#808080')
        category_plots = []  # 이 카테고리의 모든 플롯
        
        # 1. 프로그램별 분포
        if prob_col in program_df.columns:
            top_20_programs = program_df.nlargest(20, 'total_comments').sort_values(prob_col, ascending=True)
            
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=top_20_programs[prob_col],
                y=top_20_programs['program'],
                orientation='h',
                marker=dict(
                    color=category_color,
                    opacity=0.8,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=[f'{val:.4f}' for val in top_20_programs[prob_col]],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>확률: %{x:.4f}<br>총 댓글: %{customdata[0]:,}개<br>혐오 비율: %{customdata[1]:.2%}<extra></extra>',
                customdata=top_20_programs[['total_comments', 'avg_hate_ratio']].values
            ))
            fig1.update_layout(
                title=f'<b>[{category}] 프로그램별 평균 확률</b> (상위 20개 프로그램)',
                xaxis_title='평균 확률',
                yaxis_title='프로그램',
                height=600,
                width=1000,
                hovermode='closest',
                plot_bgcolor='white'
            )
            category_plots.append(('프로그램별 분포', fig1))
        
        # 2. 동영상별 분포
        if prob_col in video_df.columns:
            top_30_videos = video_df.nlargest(30, 'total_comments')
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=top_30_videos['hate_ratio'],
                y=top_30_videos[prob_col],
                mode='markers',
                marker=dict(
                    size=top_30_videos['total_comments'] / 10,
                    color=category_color,
                    opacity=0.7,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=top_30_videos['video_title'],
                hovertemplate='<b>%{text}</b><br>프로그램: %{customdata[0]}<br>' + category + ' 확률: %{y:.4f}<br>혐오 비율: %{x:.2%}<br>혐오 점수: %{customdata[1]:.3f}<extra></extra>',
                customdata=top_30_videos[['program', 'avg_hate_score']].values
            ))
            fig2.update_layout(
                title=f'<b>[{category}] 동영상별 확률 분포</b> (상위 30개 동영상)',
                xaxis_title='혐오 표현 비율',
                yaxis_title=f'{category} 확률',
                height=600,
                width=1000,
                hovermode='closest',
                plot_bgcolor='white'
            )
            category_plots.append(('동영상별 분포', fig2))
        
        # 3. 시간에 따른 추이
        if 'upload_date' in video_df.columns and prob_col in video_df.columns:
            ts_df = video_df.dropna(subset=['upload_date', prob_col]).sort_values('upload_date')
            
            if len(ts_df) > 0:
                fig3 = go.Figure()
                
                # 산점도
                fig3.add_trace(go.Scatter(
                    x=ts_df['upload_date'],
                    y=ts_df[prob_col],
                    mode='markers',
                    marker=dict(
                        size=ts_df['total_comments'] / 20,
                        color=category_color,
                        opacity=0.6,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    text=ts_df['video_title'],
                    name='데이터 포인트',
                    hovertemplate='<b>%{text}</b><br>프로그램: %{customdata[0]}<br>' + category + ' 확률: %{y:.4f}<br>혐오 비율: %{customdata[1]:.2%}<extra></extra>',
                    customdata=ts_df[['program', 'hate_ratio']].values
                ))
                
                # 추세선 추가
                if len(ts_df) > 1:
                    window = max(5, int(len(ts_df) * 0.1))
                    ts_df['trend'] = ts_df[prob_col].rolling(window=window, min_periods=1).mean()
                    
                    fig3.add_trace(go.Scatter(
                        x=ts_df['upload_date'],
                        y=ts_df['trend'],
                        mode='lines',
                        name='추세선',
                        line=dict(color=category_color, width=3, dash='solid'),
                        opacity=0.8,
                        hoverinfo='skip'
                    ))
                
                fig3.update_layout(
                    title=f'<b>[{category}] 시간에 따른 추이</b>',
                    xaxis_title='업로드 일자',
                    yaxis_title=f'{category} 확률',
                    height=600,
                    width=1200,
                    hovermode='closest',
                    plot_bgcolor='white',
                    xaxis=dict(tickformat='%Y-%m-%d')
                )
                category_plots.append(('시계열 추이', fig3))
        
        if category_plots:
            category_figures.append((category, category_plots))
            print(f"  ✓ {category} 카테고리: {len(category_plots)}개 플롯 생성")

    # 4. Save to HTML
    output_dir = Path('output/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'hatescore_dashboard.html'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('<html><head><title>HateScore Analysis Dashboard (Corrected)</title>')
        f.write('<meta charset="UTF-8">')
        f.write('''<style>
            body { 
                font-family: "Nanum Gothic", "Malgun Gothic", sans-serif; 
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }
            .warning-box {
                background: #fff3cd;
                border-left: 5px solid #ffc107;
                padding: 20px;
                margin: 20px auto;
                max-width: 1200px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .warning-box h3 {
                color: #856404;
                margin-top: 0;
            }
            .warning-box p {
                color: #856404;
                line-height: 1.6;
            }
            .warning-box ul {
                color: #856404;
                line-height: 1.8;
            }
            h1, h2, h3 {
                text-align: center;
            }
        </style>''')
        f.write('</head><body>')
        f.write('<h1 style="margin-top: 20px;">🎯 HateScore 혐오 표현 분석 대시보드</h1>')
        f.write('<p style="text-align: center; font-size: 1.1em; color: #666;">(수정된 분류 기준 적용)</p>')
        f.write('<p style="text-align: center;">10가지 혐오 카테고리(여성/가족, 단순 악플, 남성, 성소수자, 인종/국적, 연령, 지역, 종교, 기타 혐오)의 분포와 추이를 확인하세요.</p>')
        
        # 경고 박스 추가
        f.write('''
        <div class="warning-box">
            <h3>⚠️ 분류 기준 변경 안내</h3>
            <p>
                기존 분석에서는 <strong>argmax 기반</strong>으로 분류하여 대부분의 댓글이 혐오로 잘못 분류되었습니다. 
                이 대시보드는 <strong>threshold 기반 분류 (임계값 0.5)</strong>를 적용하여 정확도를 크게 개선했습니다.
            </p>
            <p style="margin-top: 15px;"><strong>새로운 분류 기준:</strong></p>
            <ul>
                <li><strong>None 확률 ≥ 0.5</strong> → 정상 댓글</li>
                <li><strong>혐오 카테고리 확률 > 0.5 AND None < 0.5</strong> → 혐오 댓글</li>
                <li><strong>그 외</strong> → 정상 댓글 (애매한 경우는 정상으로 처리)</li>
            </ul>
            <p style="margin-top: 15px; font-size: 0.9em;">
                💡 이 변경으로 혐오 댓글 비율이 97.4% → 6.1%로 조정되었습니다.
            </p>
        </div>
        ''')
        
        f.write('<h2 style="text-align: center; margin-top: 40px;">1. 전체 프로그램 비교 및 추이</h2>')
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('<hr>')

        f.write('<h2 style="text-align: center; margin-top: 40px;">2. 프로그램별 상세 혐오 표현 추이</h2>')
        if not program_figures:
            f.write('<p style="text-align: center;">데이터가 충분한 프로그램이 없습니다.</p>')
        
        for fig in program_figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('<hr>')
        
        # 3. 각 혐오 카테고리별 개별 플롯
        f.write('<h2 style="text-align: center; margin-top: 40px;">3. 혐오 카테고리별 개별 분석</h2>')
        for category, plots in category_figures:
            f.write(f'<h3 style="text-align: center; margin-top: 30px; color: {HATE_CATEGORY_COLORS.get(category, "#000")};">[{category}] 카테고리 분석</h3>')
            for plot_name, plot_fig in plots:
                f.write(f'<h4 style="text-align: center; margin-top: 20px;">{plot_name}</h4>')
                f.write(plot_fig.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write('<hr>')
            
        f.write('</body></html>')

    print(f"\n✅ Dashboard saved to: {output_file}")
    print(f"   Includes {len(program_figures)} individual program plots.")
    print(f"   Includes {len(category_figures)} categories with individual plots.")
    total_category_plots = sum(len(plots) for _, plots in category_figures)
    print(f"   Total category plots: {total_category_plots}")
    print(f"   Open this file in your web browser to view interactive plots.")

if __name__ == "__main__":
    generate_dashboard()

