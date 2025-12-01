"""
Streamlit 앱: 주가 클러스터링 시각화
섹터별 필터링 및 인터랙티브 시각화
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from collections import Counter

from config import setup_fonts, get_korean_font_path
from data_loader import load_tickers_from_csv, load_tickers_from_fdr, collect_stock_prices, get_stock_name
from data_processor import process_stock_data
from clustering import find_optimal_k, perform_kmeans_clustering
from sector_mapper import load_sector_mapping, get_sector_from_code, get_all_sectors_from_mapping
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="주가 클러스터링 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 폰트 설정 (matplotlib용)
setup_fonts()

# 섹터 분류 키워드 (종목명에 포함된 키워드로 자동 분류)
SECTOR_KEYWORDS = {
    'IT/소프트웨어': ['네이버', 'NAVER', '카카오', '크래프톤', '넷마블', '엔씨소프트', '하이브', '페이', '소프트웨어', '게임', '엔터테인먼트', '스퀘어', 'SK스퀘어'],
    '반도체/전자': ['삼성전자', 'SK하이닉스', 'LG전자', 'SDI', '전기', '이노텍', '반도체', '반도체장비', '디스플레이', '디스플레이', '전자'],
    '자동차': ['현대차', '기아', '모비스', '글로비스', '오토', '타이어', '자동차'],
    '금융': ['금융', '지주', '은행', '생명', '손해보험', '카드', '증권', '투자', '자산운용', '리츠', '캐피탈'],
    '화학/에너지': ['화학', '에너지', 'LG에너지', 'SK이노베이션', 'S-Oil', 'GS칼텍스', '포스코', '석유', '정유', 'LNG', '가스'],
    '건설/중공업': ['건설', '중공업', '조선', '두산', 'HD', '한화에어로', '에어로스페이스', '중공업', '엔진', '발전'],
    '바이오/제약': ['바이오', '제약', '약품', '유한양행', '셀트리온', '삼성바이오', '녹십자', '대웅제약', '종근당', '동화약품'],
    '유통/서비스': ['신세계', '이마트', '롯데쇼핑', 'GS리테일', '퍼시픽', '아모레', '코스맥스', '유통', '백화점', '마트'],
    '통신': ['텔레콤', 'KT', 'LG유플러스', 'SK텔레콤', '통신'],
    '철강/소재': ['POSCO', '포스코', '고려아연', '제일제당', '한진', 'CJ', '한화솔루션', 'LS', '소재', '금속'],
    '운송/물류': ['한진', '현대글로비스', '물류', '운송', '항공', '해운', 'KDB'],
    '전력/가스': ['전력', '가스', '한국전력', '한국가스공사', '도시가스'],
    '섬유/의류': ['한섬', 'LF', '의류', '섬유'],
}

# 정확한 종목명 매칭 (우선순위 높음)
EXACT_MATCH = {
    'IT/소프트웨어': ['NAVER', '카카오', '크래프톤', '넷마블', '엔씨소프트', '하이브', '카카오뱅크', '카카오페이', 'SK스퀘어'],
    '반도체/전자': ['삼성전자', 'SK하이닉스', 'LG전자', '삼성SDI', '삼성전기', 'LG이노텍'],
    '자동차': ['현대차', '기아', '현대모비스'],
    '금융': ['KB금융', '신한지주', '하나금융지주', '우리금융지주', '한국금융지주', '메리츠금융지주', 'BNK금융지주', 'JB금융지주'],
    '화학/에너지': ['LG화학', 'SK이노베이션', 'S-Oil', 'LG에너지솔루션', '포스코퓨처엠'],
    '건설/중공업': ['두산에너빌리티', 'HD현대중공업', 'HD한국조선해양', '현대건설', '한화에어로스페이스'],
    '바이오/제약': ['삼성바이오로직스', '셀트리온', 'SK바이오팜', '한미약품', '유한양행'],
    '유통/서비스': ['아모레퍼시픽', '신세계', '이마트', '롯데쇼핑', 'GS리테일'],
    '통신': ['SK텔레콤', 'KT', 'LG유플러스'],
    '철강/소재': ['POSCO홀딩스', '고려아연', '한국타이어앤테크놀로지'],
}


def get_sector(stock_name, sector_keywords=None, exact_match=None):
    """
    종목명으로부터 섹터 찾기 (개선된 버전)
    
    Parameters:
    -----------
    stock_name : str
        종목명
    sector_keywords : dict
        섹터별 키워드 딕셔너리
    exact_match : dict
        정확한 종목명 매칭 딕셔너리
    
    Returns:
    --------
    sector : str
        섹터명
    """
    if exact_match is None:
        exact_match = EXACT_MATCH
    if sector_keywords is None:
        sector_keywords = SECTOR_KEYWORDS
    
    # 1단계: 정확한 종목명 매칭 (우선순위)
    for sector, stocks in exact_match.items():
        if stock_name in stocks:
            return sector
    
    # 2단계: 키워드 기반 매칭
    for sector, keywords in sector_keywords.items():
        for keyword in keywords:
            if keyword in stock_name:
                return sector
    
    return '기타'


def find_optimal_k_streamlit(movements, max_k=10, min_k=2):
    """Streamlit용 최적 K 탐색 함수 (Plotly 사용)"""
    from sklearn.preprocessing import Normalizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    k_range = range(min_k, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    normalizer = Normalizer()
    normalized_data = normalizer.fit_transform(movements)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, k in enumerate(k_range):
        status_text.text(f"K={k} 테스트 중...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_data)
        
        inertias.append(kmeans.inertia_)
        
        if len(movements) > 100:
            sample_indices = np.random.choice(len(movements), size=min(100, len(movements)), replace=False)
            sample_data = normalized_data[sample_indices]
            sample_labels = labels[sample_indices]
            silhouette_avg = silhouette_score(sample_data, sample_labels)
        else:
            silhouette_avg = silhouette_score(normalized_data, labels)
        silhouette_scores.append(silhouette_avg)
        
        progress_bar.progress((idx + 1) / len(k_range))
    
    results_df = pd.DataFrame({
        'K': list(k_range),
        'Inertia': inertias,
        'Silhouette_Score': silhouette_scores
    })
    
    # 최적 K 결정
    optimal_k_silhouette = int(results_df.loc[results_df['Silhouette_Score'].idxmax(), 'K'])
    
    results_df['Inertia_Change'] = results_df['Inertia'].diff().abs()
    results_df['Inertia_Change_Rate'] = results_df['Inertia_Change'].pct_change().abs()
    
    if len(results_df) > 2:
        mean_change_rate = results_df['Inertia_Change_Rate'].mean()
        elbow_candidates = results_df[results_df['Inertia_Change_Rate'] < mean_change_rate]
        if not elbow_candidates.empty:
            optimal_k_elbow = int(elbow_candidates.iloc[0]['K'])
        else:
            optimal_k_elbow = int(results_df.loc[results_df['Inertia_Change_Rate'].idxmin(), 'K'])
    else:
        optimal_k_elbow = min_k
    
    optimal_k = int(np.round((optimal_k_silhouette + optimal_k_elbow) / 2))
    
    # Plotly 시각화
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method', 'Silhouette Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Elbow Method
    fig.add_trace(
        go.Scatter(
            x=results_df['K'],
            y=results_df['Inertia'],
            mode='lines+markers',
            name='Inertia',
            line=dict(width=3),
            marker=dict(size=10)
        ),
        row=1, col=1
    )
    
    fig.add_vline(
        x=optimal_k_elbow, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Elbow K={optimal_k_elbow}",
        row=1, col=1
    )
    
    # Silhouette Score
    colors_bar = ['red' if k == optimal_k_silhouette else 'steelblue' for k in results_df['K']]
    fig.add_trace(
        go.Bar(
            x=results_df['K'],
            y=results_df['Silhouette_Score'],
            name='Silhouette Score',
            marker_color=colors_bar
        ),
        row=1, col=2
    )
    
    fig.add_vline(
        x=optimal_k_silhouette,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal K={optimal_k_silhouette}",
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="클러스터 개수 (K)", row=1, col=1)
    fig.update_xaxes(title_text="클러스터 개수 (K)", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="최적 K 값 탐색 결과"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    status_text.empty()
    progress_bar.empty()
    
    st.info(f"**Elbow Method 추천 K**: {optimal_k_elbow}  |  **Silhouette Score 최적 K**: {optimal_k_silhouette}  |  **최종 추천 K**: {optimal_k}")
    
    return optimal_k, results_df


@st.cache_data
def load_and_process_data(csv_path, limit, years):
    """데이터 로드 및 처리 (캐싱)"""
    # 종목 리스트 수집
    if csv_path:
        tickers, names = load_tickers_from_csv(csv_path)
        if tickers is None:
            return None, None, None, 0
        total_tickers = len(tickers)
    else:
        tickers, names = load_tickers_from_fdr(limit=limit)
        total_tickers = len(tickers)
    
    if limit:
        tickers = tickers[:limit]
        total_tickers = len(tickers)
    
    # 주가 데이터 수집 (코스피 종목이므로 최소 데이터 포인트를 낮춤)
    stock_data = collect_stock_prices(tickers, years=years, min_data_points=50)
    if stock_data is None:
        return None, None, None, total_tickers
    
    # 로그 누적 수익률 계산
    movements = process_stock_data(stock_data)
    if movements is None or movements.empty:
        return None, None, None, total_tickers
    
    return movements, names, stock_data, total_tickers


def get_stock_code_from_name(company_name, sector_names_dict):
    """
    회사명으로 종목 코드 찾기
    
    Parameters:
    -----------
    company_name : str
        회사명
    sector_names_dict : dict
        {code: name} 형식의 딕셔너리
    
    Returns:
    --------
    str or None
        종목 코드, 없으면 None
    """
    for code, name in sector_names_dict.items():
        if name == company_name:
            return code
    return None


def get_stock_price_chart(stock_code, company_name, years=1):
    """
    종목 코드로 1년치 주가 데이터를 가져와서 그래프로 표시
    
    Parameters:
    -----------
    stock_code : str
        종목 코드 (예: '005930')
    company_name : str
        회사명
    years : float
        가져올 기간 (년)
    
    Returns:
    --------
    plotly.graph_objects.Figure or None
        주가 차트, 실패하면 None
    """
    try:
        from datetime import datetime, timedelta
        
        start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        df = fdr.DataReader(stock_code, start_date, end_date)
        
        if df is None or df.empty:
            return None
        
        # Plotly로 주가 차트 생성
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='종가',
            line=dict(color='blue', width=2),
            hovertemplate='날짜: %{x}<br>종가: %{y:,.0f}원<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{company_name} ({stock_code}) 주가 추이 (최근 {years}년)',
            xaxis_title='날짜',
            yaxis_title='종가 (원)',
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"주가 데이터 가져오기 실패: {e}")
        return None


@st.cache_data
def load_and_process_score_data(csv_path='data/score.csv'):
    """
    score.csv 파일에서 PCA 결과 데이터 로드 및 처리
    
    Parameters:
    -----------
    csv_path : str
        score.csv 파일 경로
    
    Returns:
    --------
    data : pd.DataFrame
        Company_Name, PCA1, PCA2 컬럼을 가진 DataFrame
    pca_data : np.array
        PCA 좌표 (2차원)
    """
    try:
        df = pd.read_csv(csv_path)
        # 컬럼명 정리
        df = df.rename(columns={'0': 'PCA1', '1': 'PCA2'})
        pca_data = df[['PCA1', 'PCA2']].values
        return df, pca_data
    except Exception as e:
        st.error(f"❌ score.csv 로드 실패: {e}")
        return None, None


def perform_clustering_on_pca_data(pca_data, n_clusters=4):
    """
    PCA 데이터에 대해 K-means 클러스터링 수행
    
    Parameters:
    -----------
    pca_data : np.array
        PCA 좌표 (N x 2)
    n_clusters : int
        클러스터 개수
    
    Returns:
    --------
    labels : np.array
        클러스터 레이블
    """
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_data)
    
    return labels


def create_interactive_plot_from_pca(results, selected_sectors=None, selected_clusters=None, search_term=None, n_clusters=4):
    """
    score.csv 데이터용 간단한 Plotly 시각화 (test_clustering_score.py 방식)
    
    Parameters:
    -----------
    results : pd.DataFrame
        결과 데이터 (Name, PCA1, PCA2, Cluster, Sector 포함)
    selected_sectors : list
        선택된 섹터 리스트
    selected_clusters : list
        선택된 클러스터 리스트
    search_term : str
        검색어
    n_clusters : int
        클러스터 개수
    """
    # 필터링
    filtered_results = results.copy()
    
    if selected_sectors and '전체' not in selected_sectors:
        filtered_results = filtered_results[
            filtered_results['Sector'].isin(selected_sectors)
        ]
    
    if selected_clusters:
        filtered_results = filtered_results[
            filtered_results['Cluster'].isin(selected_clusters)
        ]
    
    if search_term:
        # Name 또는 Company_Name 컬럼 사용
        name_col = 'Company_Name' if 'Company_Name' in filtered_results.columns else 'Name'
        filtered_results = filtered_results[
            filtered_results[name_col].str.contains(search_term, case=False, na=False)
        ]
    
    # Plotly로 인터랙티브 시각화 (test_clustering_score.py 방식)
    # Name 또는 Company_Name 컬럼 사용
    name_col = 'Company_Name' if 'Company_Name' in filtered_results.columns else 'Name'
    
    # hover_data에서 Cluster를 명시적으로 포함 (순서 보장을 위해)
    fig = px.scatter(
        filtered_results,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_name=name_col,  # 점에 마우스 올리면 기업명이 제목으로 표시
        hover_data={
            'Sector': True,
            'PCA1': ':.3f',
            'PCA2': ':.3f',
            'Cluster': True,  # 클러스터 값 포함
            name_col: False  # hover_name으로 이미 표시되므로 중복 제거
        },
        title=f'K-means 클러스터링 결과 (K={n_clusters})',
        labels={
            'PCA1': 'PCA 1차원 (X축)', 
            'PCA2': 'PCA 2차원 (Y축)', 
            'Cluster': '클러스터', 
            'Sector': '섹터'
        }
    )
    
    # 커스텀 hover 템플릿으로 상세 정보 표시
    # customdata 구조 확인 결과: [Sector, Cluster, Name]
    # x, y 좌표는 %{x}, %{y}로 직접 참조 가능
    # 클러스터 번호는 customdata[1]에 있음
    fig.update_traces(
        marker=dict(size=10, opacity=0.7, line=dict(width=0.5, color='white')),
        hovertemplate='<b>%{hovertext}</b><br>' +
                      '섹터: %{customdata[0]}<br>' +
                      'PCA1: %{x:.3f}<br>' +
                      'PCA2: %{y:.3f}<br>' +
                      '클러스터: %{customdata[1]}<br>' +
                      '<extra></extra>'
    )
    
    fig.update_layout(
        width=1000,
        height=700,
        font=dict(size=12)
    )
    
    return fig


def create_interactive_plot(results, movements=None, pipeline=None, selected_sectors=None, selected_clusters=None, search_term=None, use_pca_coords=False):
    """
    인터랙티브 Plotly 차트 생성 (주가 데이터용)
    
    Parameters:
    -----------
    results : pd.DataFrame
        결과 데이터 (Code, Name, Cluster, Sector 포함)
    movements : pd.DataFrame or None
        주가 데이터 (use_pca_coords=False일 때 사용)
    pipeline : Pipeline or None
        학습된 파이프라인 (use_pca_coords=False일 때 사용)
    selected_sectors : list
        선택된 섹터 리스트
    selected_clusters : list
        선택된 클러스터 리스트
    search_term : str
        검색어
    use_pca_coords : bool
        True면 results에 이미 PCA 좌표가 있음 (PCA1, PCA2 컬럼)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import Normalizer
    
    # 주가 데이터: PCA 계산 필요
    if movements is None:
        st.error("❌ movements 데이터가 필요합니다.")
        return None, None
    
    normalizer = Normalizer()
    normalized_data = normalizer.fit_transform(movements)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(normalized_data)
    
    results['x'] = reduced_data[:, 0]
    results['y'] = reduced_data[:, 1]
    
    # 필터링
    filtered_results = results.copy()
    
    if selected_sectors and '전체' not in selected_sectors:
        filtered_results = filtered_results[
            filtered_results['Sector'].isin(selected_sectors)
        ]
    
    if selected_clusters:
        filtered_results = filtered_results[
            filtered_results['Cluster'].isin(selected_clusters)
        ]
    
    if search_term:
        name_match = filtered_results['Name'].str.contains(search_term, case=False, na=False)
        # Code 컬럼이 있고 비어있지 않은 경우에만 검색
        if 'Code' in filtered_results.columns:
            # Code 컬럼이 비어있지 않은 행만 검색
            code_not_empty = filtered_results['Code'].astype(str).str.strip() != ''
            code_match = filtered_results['Code'].astype(str).str.contains(search_term, case=False, na=False)
            filtered_results = filtered_results[name_match | (code_not_empty & code_match)]
        else:
            filtered_results = filtered_results[name_match]
    
    # Plotly 시각화
    fig = go.Figure()
    
    # 클러스터별로 색상 지정
    colors = px.colors.qualitative.Set3
    n_clusters = results['Cluster'].nunique()
    
    for cluster_id in sorted(results['Cluster'].unique()):
        cluster_data = filtered_results[filtered_results['Cluster'] == cluster_id]
        
        if len(cluster_data) > 0:
            color = colors[cluster_id % len(colors)]
            
            # 호버 정보 준비
            hover_texts = []
            for idx, row in cluster_data.iterrows():
                hover_text = f"<b>{row['Name']}</b><br>"
                if 'Code' in row and row['Code'] and str(row['Code']).strip():
                    hover_text += f"코드: {row['Code']}<br>"
                hover_text += f"클러스터: {cluster_id}<br>"
                hover_text += f"섹터: {row['Sector']}<br>"
                hover_text += f"X: {row['x']:.3f}<br>"
                hover_text += f"Y: {row['y']:.3f}"
                hover_texts.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                name=f'클러스터 {cluster_id} ({len(cluster_data)}개)',
                text=cluster_data['Name'],
                textposition='middle center',
                hovertext=hover_texts,
                hovertemplate='%{hovertext}<extra></extra>',
                marker=dict(
                    size=15,
                    color=color,
                    line=dict(width=1.5, color='black'),
                    opacity=0.8
                )
            ))
    
    # 레이아웃 설정
    fig.update_layout(
        title={
            'text': f'📊 주가 클러스터링 시각화 (표시: {len(filtered_results)}개 / 전체: {len(results)}개)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': 'darkblue'}
        },
        xaxis_title=f'PCA Component 1 (설명력: {pca.explained_variance_ratio_[0]*100:.1f}%)',
        yaxis_title=f'PCA Component 2 (설명력: {pca.explained_variance_ratio_[1]*100:.1f}%)',
        hovermode='closest',
        height=750,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='white'
    )
    
    # 그리드 추가
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig, pca


def create_wordcloud(sector_counts):
    """
    섹터 빈도수로 워드 클라우드 생성
    
    Parameters:
    -----------
    sector_counts : pd.Series
        섹터별 빈도수 (value_counts 결과)
    
    Returns:
    --------
    wordcloud_image : bytes
        워드 클라우드 이미지 바이트 데이터
    """
    # 섹터별 빈도수를 딕셔너리로 변환
    word_freq = sector_counts.to_dict()
    
    # 한국어 폰트 경로 가져오기
    font_path = get_korean_font_path()
    
    # 워드 클라우드 생성
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path=font_path,  # 한국어 폰트 사용
        colormap='Set3',
        max_words=50,
        relative_scaling=0.5,
        min_font_size=10,
        max_font_size=60
    ).generate_from_frequencies(word_freq)
    
    # 이미지를 바이트로 변환
    img_buffer = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer


def main():
    st.title("📊 주가 클러스터링 분석 대시보드")
    st.markdown("---")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        st.info("📁 데이터 소스: `data/score.csv` (PCA 결과 데이터)")
        
        # 클러스터링 설정
        st.subheader("클러스터링 설정")
        find_optimal = st.checkbox("최적 K 자동 탐색", value=False)
        
        if not find_optimal:
            n_clusters = st.slider("클러스터 개수 (K)", 2, 15, 4)
        else:
            max_k = st.slider("최대 K 값", 5, 15, 10)
            n_clusters = 4  # 초기값
        
        # 분석 실행 버튼
        analyze_button = st.button("🔍 분석 시작", type="primary", use_container_width=True)
    
    # 메인 영역
    if analyze_button:
        with st.spinner("데이터를 로드하고 분석 중입니다..."):
            try:
                # score.csv 데이터 사용
                score_data, pca_data = load_and_process_score_data('data/score.csv')
                if score_data is None or pca_data is None:
                    st.error("❌ score.csv 데이터 로드에 실패했습니다.")
                    return
                
                st.success(f"✅ {len(score_data)}개 기업 데이터 로드 완료")
                
                # 최적 K 탐색 (선택사항)
                if find_optimal:
                    st.subheader("🔍 최적 K 값 탐색")
                    from sklearn.metrics import silhouette_score
                    from sklearn.cluster import KMeans
                    import tqdm
                    
                    k_range = range(2, max_k + 1)
                    silhouette_scores = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, k in enumerate(k_range):
                        status_text.text(f"K={k} 테스트 중...")
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(pca_data)
                        
                        if len(pca_data) > 1000:
                            # 데이터가 많으면 샘플링
                            sample_size = min(1000, len(pca_data))
                            sample_indices = np.random.choice(len(pca_data), size=sample_size, replace=False)
                            sample_data = pca_data[sample_indices]
                            sample_labels = labels[sample_indices]
                            silhouette_avg = silhouette_score(sample_data, sample_labels)
                        else:
                            silhouette_avg = silhouette_score(pca_data, labels)
                        silhouette_scores.append(silhouette_avg)
                        progress_bar.progress((idx + 1) / len(k_range))
                    
                    optimal_k_idx = np.argmax(silhouette_scores)
                    optimal_k = list(k_range)[optimal_k_idx]
                    n_clusters = optimal_k
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"✨ 최적 K 값: {optimal_k}로 설정되었습니다.")
                
                # 클러스터링 수행
                with st.spinner("클러스터링을 수행 중입니다..."):
                    labels = perform_clustering_on_pca_data(pca_data, n_clusters=n_clusters)
                
                # 섹터 정보 로드
                sector_mapping_dict, sector_names_dict = load_sector_mapping('data/kospi_code.csv')
                
                # 결과 정리
                sectors = []
                for name in score_data['Company_Name']:
                    # 이름으로 섹터 찾기
                    sector = '기타'
                    for code, stock_name in sector_names_dict.items():
                        if name == stock_name:
                            sector = get_sector_from_code(code, sector_mapping_dict)
                            break
                    sectors.append(sector)
                
                results = pd.DataFrame({
                    'Name': score_data['Company_Name'],
                    'PCA1': score_data['PCA1'],
                    'PCA2': score_data['PCA2'],
                    'Cluster': labels,
                    'Sector': sectors
                })
                
                # 세션 상태에 저장
                st.session_state['results'] = results
                st.session_state['pca_data'] = pca_data
                st.session_state['n_clusters'] = n_clusters
                st.session_state['sector_names_dict'] = sector_names_dict
                
                st.success(f"✅ 분석 완료! {len(results)}개 종목이 {n_clusters}개 클러스터로 분류되었습니다.")
                
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                st.exception(e)
                return
    
    # 결과가 있으면 시각화
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        st.markdown("---")
        
        # 필터링 섹션
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            st.subheader("🔍 필터링")
            all_sectors = ['전체'] + sorted(results['Sector'].unique().tolist())
            selected_sectors = st.multiselect(
                "섹터 선택",
                all_sectors,
                default=['전체'],
                help="표시할 섹터를 선택하세요"
            )
        
        with col2:
            st.subheader("📊 클러스터")
            all_clusters = sorted(results['Cluster'].unique().tolist())
            selected_clusters = st.multiselect(
                "클러스터 선택",
                all_clusters,
                default=all_clusters,
                help="표시할 클러스터를 선택하세요"
            )
        
        with col3:
            st.subheader("🔎 검색")
            search_term = st.text_input(
                "종목명 검색",
                "",
                help="종목명으로 검색하세요"
            )
        
        # 통계 정보
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("전체 종목 수", len(results))
        with col2:
            st.metric("클러스터 수", results['Cluster'].nunique())
        with col3:
            st.metric("섹터 수", results['Sector'].nunique())
        with col4:
            # 필터링된 결과 개수 계산
            filtered_df = results.copy()
            if selected_sectors and '전체' not in selected_sectors:
                filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]
            if selected_clusters:
                filtered_df = filtered_df[filtered_df['Cluster'].isin(selected_clusters)]
            if search_term:
                # Name 또는 Company_Name 컬럼 사용
                name_col = 'Company_Name' if 'Company_Name' in filtered_df.columns else 'Name'
                name_match = filtered_df[name_col].str.contains(search_term, case=False, na=False)
                # Code 컬럼이 있고 비어있지 않은 경우에만 검색
                if 'Code' in filtered_df.columns:
                    code_not_empty = filtered_df['Code'].astype(str).str.strip() != ''
                    code_match = filtered_df['Code'].astype(str).str.contains(search_term, case=False, na=False)
                    filtered_df = filtered_df[name_match | (code_not_empty & code_match)]
                else:
                    filtered_df = filtered_df[name_match]
            st.metric("표시 중", len(filtered_df))
        
        # 인터랙티브 차트
        st.markdown("---")
        st.subheader("📈 클러스터링 시각화")
        
        # score.csv 데이터: 간단한 시각화 (test_clustering_score.py 방식)
        n_clusters = st.session_state.get('n_clusters', 4)
        fig = create_interactive_plot_from_pca(
            results, 
            selected_sectors, 
            selected_clusters, 
            search_term,
            n_clusters=n_clusters
        )
        
        # 점 클릭 이벤트 처리 (Streamlit 1.31.0+)
        try:
            selected_point = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="pca_chart")
            
            # 선택된 점이 있으면 주가 차트 표시
            if selected_point and 'selection' in selected_point and selected_point['selection'].get('points'):
                point_data = selected_point['selection']['points'][0]
                selected_company = None
                
                # 선택된 점에서 회사명 가져오기
                if 'hovertext' in point_data:
                    selected_company = point_data['hovertext']
                elif 'customdata' in point_data:
                    customdata = point_data['customdata']
                    if isinstance(customdata, list) and len(customdata) > 0:
                        selected_company = customdata[0]
                
                if selected_company:
                    # 회사명으로 종목 코드 찾기
                    sector_names_dict = st.session_state.get('sector_names_dict', {})
                    if not sector_names_dict:
                        _, sector_names_dict = load_sector_mapping('data/kospi_code.csv')
                        st.session_state['sector_names_dict'] = sector_names_dict
                    
                    stock_code = get_stock_code_from_name(selected_company, sector_names_dict)
                    
                    if stock_code:
                        st.markdown("---")
                        st.subheader(f"📈 {selected_company} 주가 추이")
                        
                        with st.spinner("주가 데이터를 가져오는 중..."):
                            price_fig = get_stock_price_chart(stock_code, selected_company, years=1)
                            if price_fig:
                                st.plotly_chart(price_fig, use_container_width=True)
                            else:
                                st.warning(f"⚠️ {selected_company}의 주가 데이터를 가져올 수 없습니다.")
                    else:
                        st.warning(f"⚠️ {selected_company}의 종목 코드를 찾을 수 없습니다.")
        except TypeError:
            # on_select가 지원되지 않는 버전에서는 기본 동작
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 점을 클릭하여 주가 차트를 보려면 Streamlit 1.31.0 이상 버전이 필요합니다.")
        
        # 클러스터별 섹터 분포 (워드 클라우드)
        st.markdown("---")
        st.subheader("📊 클러스터별 섹터 분포")
        
        selected_cluster_for_wordcloud = st.selectbox(
            "워드 클라우드를 볼 클러스터 선택",
            sorted(results['Cluster'].unique().tolist()),
            key="wordcloud_cluster"
        )
        
        cluster_data = results[results['Cluster'] == selected_cluster_for_wordcloud]
        sector_counts = cluster_data['Sector'].value_counts()
        
        if len(sector_counts) > 0:
            col1, col2 = st.columns([2, 1])
            with col1:
                wordcloud_img = create_wordcloud(sector_counts)
                st.image(wordcloud_img, use_container_width=True)
            with col2:
                st.write("**섹터별 개수**")
                st.dataframe(
                    sector_counts.reset_index().rename(columns={'index': '섹터', 'Sector': '개수'}),
                    use_container_width=True,
                    height=400
                )
        else:
            st.info("표시할 섹터 데이터가 없습니다.")
        
        
        # 다운로드 버튼
        st.markdown("---")
        csv = results.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 결과 다운로드 (CSV)",
            data=csv,
            file_name=f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        # 초기 화면
        st.info("👈 왼쪽 사이드바에서 설정을 하고 '분석 시작' 버튼을 클릭하세요.")
        
        st.markdown("""
        ### 📖 사용 방법
        
        1. **데이터 소스**
           - `data/score.csv` 파일의 PCA 결과 데이터를 사용합니다
        
        2. **클러스터링 설정**
           - 최적 K 자동 탐색 또는 수동으로 K 값 지정
        
        3. **분석 시작**
           - 버튼을 클릭하면 데이터를 로드하고 클러스터링을 수행합니다
        
        4. **필터링 및 탐색**
           - 섹터별, 클러스터별로 필터링 가능
           - 종목명으로 검색 가능
           - 인터랙티브 차트에서 점을 클릭하면 해당 기업의 주가 추이 확인
           - 워드 클라우드로 클러스터 내 섹터 분포 확인
        """)


if __name__ == "__main__":
    main()

