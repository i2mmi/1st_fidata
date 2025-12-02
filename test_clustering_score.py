"""
score.csv 파일을 이용한 클러스터링 테스트 스크립트
PCA가 이미 적용된 데이터를 사용하여 K-means 클러스터링만 수행
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score
import tqdm
from sector_mapper import load_sector_mapping, get_sector_from_code
from config import setup_fonts


def load_score_data(csv_path='data/score.csv'):
    """
    score.csv 파일에서 PCA 결과 데이터 로드
    
    Parameters:
    -----------
    csv_path : str
        score.csv 파일 경로
    
    Returns:
    --------
    data : pd.DataFrame
        Company_Name, PCA1(x), PCA2(y) 컬럼을 가진 DataFrame
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ {len(df)}개 기업 데이터 로드 완료")
        return df
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None


def perform_clustering_on_pca(data, n_clusters=5):
    """
    PCA 결과 데이터에 대해 K-means 클러스터링 수행
    
    Parameters:
    -----------
    data : pd.DataFrame
        Company_Name, 0(PCA1), 1(PCA2) 컬럼을 가진 DataFrame
    n_clusters : int
        클러스터 개수
    
    Returns:
    --------
    labels : np.array
        클러스터 레이블
    """
    # PCA 좌표 추출 (컬럼 0과 1)
    X = data[['0', '1']].values
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    print(f"✅ K-means 클러스터링 완료 (K={n_clusters})")
    return labels


def visualize_clustering_results(data, labels, n_clusters):
    """
    클러스터링 결과 시각화
    
    Parameters:
    -----------
    data : pd.DataFrame
        원본 데이터
    labels : np.array
        클러스터 레이블
    n_clusters : int
        클러스터 개수
    """
    # 폰트 설정
    setup_fonts()
    
    # 결과 DataFrame 생성
    results = pd.DataFrame({
        'Company_Name': data['Company_Name'],
        'PCA1': data['0'],
        'PCA2': data['1'],
        'Cluster': labels
    })
    
    # 섹터 정보 로드 및 매핑
    try:
        sector_mapping, names_mapping = load_sector_mapping('data/kospi_code.csv')
        # Company_Name과 매칭하여 섹터 찾기
        sectors = []
        for name in results['Company_Name']:
            sector = '기타'
            # 이름으로 코드 찾기
            for code, stock_name in names_mapping.items():
                if name == stock_name:
                    sector = get_sector_from_code(code, sector_mapping)
                    break
            sectors.append(sector)
        results['Sector'] = sectors
    except Exception as e:
        print(f"⚠️  섹터 정보 로드 실패: {e}")
        results['Sector'] = '기타'
    
    # Plotly로 인터랙티브 시각화 (기업명 hover 표시)
    fig = px.scatter(
        results,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_name='Company_Name',  # 점에 마우스 올리면 기업명이 제목으로 표시
        hover_data={
            'Sector': True,
            'PCA1': ':.3f',
            'PCA2': ':.3f',
            'Cluster': True,
            'Company_Name': False  # hover_name으로 이미 표시되므로 중복 제거
        },
        title=f'K-means 클러스터링 결과 (K={n_clusters})',
        labels={
            'PCA1': 'PCA 1차원 (X축)', 
            'PCA2': 'PCA 2차원 (Y축)', 
            'Cluster': '클러스터', 
            'Sector': '섹터'
        }
    )
    
    # 커스텀 hover 템플릿으로 상세 정보 표시 (기업명, 섹터, 좌표, 클러스터)
    fig.update_traces(
        marker=dict(size=10, opacity=0.7, line=dict(width=0.5, color='white')),
        hovertemplate='<b>%{hovertext}</b><br>' +
                      '섹터: %{customdata[0]}<br>' +
                      'PCA1: %{customdata[1]}<br>' +
                      'PCA2: %{customdata[2]}<br>' +
                      '클러스터: %{customdata[3]}<br>' +
                      '<extra></extra>'
    )
    
    fig.update_layout(
        width=1000,
        height=700,
        font=dict(size=12)
    )
    
    # HTML로 저장
    fig.write_html('data/clustering_score_result.html')
    print("✅ 시각화 결과가 'data/clustering_score_result.html'에 저장되었습니다.")
    print("   브라우저에서 파일을 열어서 확인하세요.")
    
    # matplotlib으로도 간단히 표시
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(results['PCA1'], results['PCA2'], c=results['Cluster'], 
                         cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PCA 1차원 (X축)', fontsize=12)
    plt.ylabel('PCA 2차원 (Y축)', fontsize=12)
    plt.title(f'K-means 클러스터링 결과 (K={n_clusters})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/clustering_score_result.png', dpi=150, bbox_inches='tight')
    print("✅ 정적 이미지가 'data/clustering_score_result.png'에 저장되었습니다.")
    plt.close()
    
    # 클러스터별 통계
    print("\n📊 클러스터별 통계:")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster_id in range(n_clusters):
        count = cluster_counts.get(cluster_id, 0)
        companies = results[results['Cluster'] == cluster_id]['Company_Name'].tolist()
        print(f"\n클러스터 {cluster_id}: {count}개 기업")
        print(f"  대표 기업: {', '.join(companies[:5])}{'...' if len(companies) > 5 else ''}")
    
    return results


def main():
    """메인 함수"""
    print("=" * 60)
    print("📊 score.csv 기반 클러스터링 테스트")
    print("=" * 60)
    
    # 1. 데이터 로드
    data = load_score_data('data/score.csv')
    if data is None:
        return
    
    print(f"\n데이터 정보:")
    print(f"  - 기업 수: {len(data)}")
    print(f"  - PCA 좌표 범위: X=[{data['0'].min():.3f}, {data['0'].max():.3f}], "
          f"Y=[{data['1'].min():.3f}, {data['1'].max():.3f}]")
    
    # 2. 클러스터링 수행 (K=4로 고정)
    print("\n" + "=" * 60)
    print("🎯 클러스터링 수행")
    print("=" * 60)
    
    # K 값을 4로 고정
    user_k = 4
    print(f"\n사용할 K 값: {user_k}")
    
    labels = perform_clustering_on_pca(data, n_clusters=user_k)
    
    # 4. 결과 시각화
    print("\n" + "=" * 60)
    print("📈 결과 시각화")
    print("=" * 60)
    
    results = visualize_clustering_results(data, labels, user_k)
    
    # 5. 결과 저장
    output_file = 'data/clustering_results_score.csv'
    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과가 '{output_file}'에 저장되었습니다.")
    
    print("\n" + "=" * 60)
    print("✨ 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

