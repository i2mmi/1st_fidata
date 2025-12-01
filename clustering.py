"""
클러스터링 모듈
- 최적 K 값 탐색
- K-Means 클러스터링 수행
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import tqdm


def find_optimal_k(movements, max_k=10, min_k=2):
    """
    최적의 클러스터 개수 K를 찾는 함수
    Elbow Method와 Silhouette Score를 사용
    
    Parameters:
    -----------
    movements : DataFrame
        클러스터링할 데이터 (행: 종목, 열: 날짜)
    max_k : int
        테스트할 최대 K 값 (기본값: 10)
    min_k : int
        테스트할 최소 K 값 (기본값: 2)
    
    Returns:
    --------
    optimal_k : int
        최적의 K 값
    results_df : DataFrame
        각 K 값에 대한 평가 지표
    """
    print(f"\n🔍 최적의 K 값 탐색 중... (K 범위: {min_k} ~ {max_k})")
    
    k_range = range(min_k, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    normalizer = Normalizer()
    normalized_data = normalizer.fit_transform(movements)
    
    for k in tqdm.tqdm(k_range, desc="K 값 테스트"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized_data)
        
        # Inertia (왜곡 제곱합) 저장
        inertias.append(kmeans.inertia_)
        
        # Silhouette Score 계산 (데이터가 많으면 샘플링)
        if len(movements) > 100:
            sample_indices = np.random.choice(len(movements), size=min(100, len(movements)), replace=False)
            sample_data = normalized_data[sample_indices]
            sample_labels = labels[sample_indices]
            silhouette_avg = silhouette_score(sample_data, sample_labels)
        else:
            silhouette_avg = silhouette_score(normalized_data, labels)
        silhouette_scores.append(silhouette_avg)
    
    # 결과를 DataFrame으로 정리
    results_df = pd.DataFrame({
        'K': list(k_range),
        'Inertia': inertias,
        'Silhouette_Score': silhouette_scores
    })
    
    # Elbow Method: Inertia의 감소율 계산
    results_df['Inertia_Change'] = results_df['Inertia'].diff().abs()
    results_df['Inertia_Change_Rate'] = results_df['Inertia_Change'].pct_change().abs()
    
    # 최적 K 결정:
    # 1. Silhouette Score가 최대인 K
    optimal_k_silhouette = int(results_df.loc[results_df['Silhouette_Score'].idxmax(), 'K'])
    
    # 2. Elbow Method: Inertia 변화율이 급격히 줄어드는 지점
    if len(results_df) > 2:
        mean_change_rate = results_df['Inertia_Change_Rate'].mean()
        elbow_candidates = results_df[results_df['Inertia_Change_Rate'] < mean_change_rate]
        if not elbow_candidates.empty:
            optimal_k_elbow = int(elbow_candidates.iloc[0]['K'])
        else:
            optimal_k_elbow = int(results_df.loc[results_df['Inertia_Change_Rate'].idxmin(), 'K'])
    else:
        optimal_k_elbow = min_k
    
    # 두 방법의 평균
    optimal_k = int(np.round((optimal_k_silhouette + optimal_k_elbow) / 2))
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Elbow Method 그래프
    ax1 = axes[0]
    ax1.plot(results_df['K'], results_df['Inertia'], marker='o', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k_elbow, color='r', linestyle='--', alpha=0.7, 
                label=f'Elbow 추천 K={optimal_k_elbow}')
    ax1.set_xlabel('클러스터 개수 (K)', fontsize=12)
    ax1.set_ylabel('Inertia (왜곡 제곱합)', fontsize=12)
    ax1.set_title('Elbow Method: 최적 K 탐색', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Silhouette Score 그래프
    ax2 = axes[1]
    bars = ax2.bar(results_df['K'], results_df['Silhouette_Score'], 
                    color=['red' if k == optimal_k_silhouette else 'steelblue' 
                           for k in results_df['K']], alpha=0.7)
    ax2.axvline(x=optimal_k_silhouette, color='r', linestyle='--', alpha=0.7, 
                label=f'Silhouette 최적 K={optimal_k_silhouette}')
    ax2.set_xlabel('클러스터 개수 (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score: 최적 K 탐색', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 K 값 탐색 결과:")
    print(f"   - Elbow Method 추천 K: {optimal_k_elbow}")
    print(f"   - Silhouette Score 최적 K: {optimal_k_silhouette}")
    print(f"   - 최종 추천 K: {optimal_k}")
    print(f"\n   각 K 값별 점수:")
    for _, row in results_df.iterrows():
        print(f"   K={int(row['K']):2d}: Silhouette={row['Silhouette_Score']:.4f}, "
              f"Inertia={row['Inertia']:.2f}")
    
    return optimal_k, results_df


def perform_kmeans_clustering(movements, n_clusters=8):
    """
    K-Means 클러스터링 수행
    
    Parameters:
    -----------
    movements : DataFrame
        클러스터링할 데이터 (행: 종목, 열: 날짜)
    n_clusters : int
        클러스터 개수
    
    Returns:
    --------
    labels : array
        각 종목의 클러스터 레이블
    pipeline : Pipeline
        학습된 파이프라인 (Normalizer + KMeans)
    """
    print(f"\n🔄 K-Means 클러스터링 진행 중... (클러스터 수: {n_clusters})")
    
    normalizer = Normalizer()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pipeline = make_pipeline(normalizer, kmeans)
    
    pipeline.fit(movements)
    labels = pipeline.predict(movements)
    
    return labels, pipeline

