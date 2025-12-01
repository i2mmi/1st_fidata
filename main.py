"""
메인 실행 파일
로그 누적 수익률 기반 K-Means 클러스터링
"""
import pandas as pd
from config import setup_fonts, DEFAULT_LIMIT, DEFAULT_N_CLUSTERS, DEFAULT_MAX_K, DEFAULT_YEARS
from data_loader import load_tickers_from_csv, load_tickers_from_fdr, collect_stock_prices, get_stock_name
from data_processor import process_stock_data
from clustering import find_optimal_k, perform_kmeans_clustering
from visualization import visualize_clustering_results, print_cluster_summary


def run_stock_clustering(csv_path=None, limit=DEFAULT_LIMIT, n_clusters=DEFAULT_N_CLUSTERS, 
                        find_optimal=False, max_k=DEFAULT_MAX_K, years=DEFAULT_YEARS):
    """
    주가 데이터를 기반으로 로그 누적 수익률 클러스터링 수행
    
    Parameters:
    -----------
    csv_path : str, optional
        티커가 들어있는 csv 파일 경로
    limit : int, optional
        테스트할 종목 수 제한 (기본값: 100)
    n_clusters : int, optional
        클러스터 개수 (기본값: 8, find_optimal=True일 때는 무시됨)
    find_optimal : bool, optional
        최적의 K 값을 자동으로 탐색할지 여부 (기본값: False)
    max_k : int, optional
        최적 K 탐색 시 테스트할 최대 K 값 (기본값: 10)
    years : int, optional
        데이터 수집 기간 (년, 기본값: 1)
    
    Returns:
    --------
    results : DataFrame
        클러스터링 결과 (Code, Name, Cluster 컬럼 포함)
    """
    # 폰트 설정
    setup_fonts()
    
    # 1. 종목 리스트 수집
    if csv_path:
        tickers, names = load_tickers_from_csv(csv_path)
        if tickers is None:
            return None
    else:
        tickers, names = load_tickers_from_fdr(limit=limit)
    
    if limit:
        tickers = tickers[:limit]
    
    # 2. 주가 데이터 수집
    stock_data = collect_stock_prices(tickers, years=years, min_data_points=200)
    if stock_data is None:
        return None
    
    # 3. 로그 누적 수익률 계산 및 전처리
    movements = process_stock_data(stock_data)
    if movements is None or movements.empty:
        print("❌ 전처리된 데이터가 없습니다.")
        return None
    
    print(f"\n✅ 전처리 완료: {len(movements)}개 종목으로 클러스터링을 진행합니다.")
    
    # 4. 최적 K 값 탐색 (선택적)
    if find_optimal:
        optimal_k, k_results = find_optimal_k(movements, max_k=max_k, min_k=2)
        n_clusters = optimal_k
        print(f"\n✨ 최적 K 값 {optimal_k}로 클러스터링을 진행합니다.")
    
    # 5. K-Means 클러스터링 수행
    labels, pipeline = perform_kmeans_clustering(movements, n_clusters=n_clusters)
    
    # 6. 종목명 매핑
    current_names = []
    for code in movements.index:
        name = get_stock_name(code, names)
        current_names.append(name)
    
    # 7. 결과 정리
    results = pd.DataFrame({
        'Code': movements.index,
        'Name': current_names,
        'Cluster': labels
    })
    
    # 8. 클러스터별 통계 출력
    print_cluster_summary(results)
    
    # 9. 시각화
    visualize_clustering_results(results, movements, pipeline, n_clusters)
    
    # 10. 결과 저장
    output_file = 'clustering_results.csv'
    results[['Code', 'Name', 'Cluster']].to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과가 '{output_file}'에 저장되었습니다.")
    
    return results


if __name__ == "__main__":
    # 옵션 1: 최적 K 자동 탐색 (추천)
    final_df = run_stock_clustering(
        csv_path='data/data_kospi.csv', 
        limit=50, 
        find_optimal=True,  # 최적 K 자동 탐색
        max_k=10
    )
    
    # 옵션 2: 수동으로 K 지정
    # final_df = run_stock_clustering(
    #     csv_path='data/kospi_code_list_100.csv', 
    #     limit=50, 
    #     n_clusters=5,  # 직접 지정
    #     find_optimal=False
    # )

