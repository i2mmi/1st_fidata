"""
시각화 모듈
- PCA를 이용한 2차원 축소 시각화
- 클러스터링 결과 시각화
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer


def visualize_clustering_results(results, movements, pipeline, n_clusters):
    """
    클러스터링 결과를 2차원 PCA 공간에 시각화
    
    Parameters:
    -----------
    results : DataFrame
        클러스터링 결과 (Code, Name, Cluster 컬럼 포함)
    movements : DataFrame
        원본 데이터 (행: 종목, 열: 날짜)
    pipeline : Pipeline
        학습된 파이프라인 (Normalizer 포함)
    n_clusters : int
        클러스터 개수
    """
    print("\n📈 시각화 생성 중...")
    
    # PCA를 이용한 2차원 축소
    normalizer = Normalizer()
    normalized_data = normalizer.fit_transform(movements)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(normalized_data)
    
    # 결과에 좌표 추가
    results['x'] = reduced_data[:, 0]
    results['y'] = reduced_data[:, 1]
    
    # 단일 플롯 생성
    plt.figure(figsize=(18, 12))
    
    # 색상 팔레트 생성
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    # 클러스터별 시각화
    for cluster_id in range(n_clusters):
        cluster_data = results[results['Cluster'] == cluster_id]
        
        if len(cluster_data) > 0:
            # 산점도
            plt.scatter(
                cluster_data['x'], 
                cluster_data['y'], 
                c=[colors[cluster_id]], 
                s=120,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
                label=f'Cluster {cluster_id} ({len(cluster_data)}개)'
            )
            
            # 종목명 라벨
            for idx, row in cluster_data.iterrows():
                plt.text(
                    row['x'], 
                    row['y'], 
                    row['Name'],
                    fontsize=9,
                    fontweight='bold',
                    alpha=1.0,
                    ha='center',
                    va='center',
                    bbox=dict(
                        boxstyle='round,pad=0.4', 
                        facecolor='white', 
                        alpha=0.85, 
                        edgecolor=colors[cluster_id], 
                        linewidth=1.5
                    )
                )
    
    # 그래프 꾸미기
    plt.title(
        f'KOSPI 종목 로그 누적 수익률 기반 클러스터링 ({len(results)}개 종목)', 
        fontsize=20, 
        fontweight='bold', 
        pad=20
    )
    plt.xlabel('PCA Component 1', fontsize=14)
    plt.ylabel('PCA Component 2', fontsize=14)
    
    # 배경 격자
    plt.grid(True, alpha=0.3, linestyle='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 범례
    plt.legend(fontsize=10, loc='best')
    
    # 설명 텍스트 추가
    explanation_text = (
        "해석 방법:\n"
        "• X축 (PCA Component 1): 첫 번째 주성분 (가장 큰 변동성 방향)\n"
        "• Y축 (PCA Component 2): 두 번째 주성분 (두 번째로 큰 변동성 방향)\n"
        "• 가까운 종목 = 유사한 주가 패턴"
    )
    plt.figtext(0.02, 0.02, explanation_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # PCA 설명력 출력
    explained_variance = pca.explained_variance_ratio_
    print(f"\n📊 PCA 설명력:")
    print(f"   Component 1: {explained_variance[0]*100:.2f}%")
    print(f"   Component 2: {explained_variance[1]*100:.2f}%")
    print(f"   전체 설명력: {sum(explained_variance)*100:.2f}%")
    
    return results


def print_cluster_summary(results):
    """
    클러스터별 통계 출력
    
    Parameters:
    -----------
    results : DataFrame
        클러스터링 결과
    """
    print("\n" + "="*60)
    print("📊 클러스터별 종목 분포")
    print("="*60)
    
    cluster_counts = results['Cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        cluster_stocks = results[results['Cluster'] == cluster_id]['Name'].tolist()
        print(f"\n클러스터 {cluster_id}: {count}개 종목")
        print(f"  {', '.join(cluster_stocks[:15])}" + ("..." if count > 15 else ""))

