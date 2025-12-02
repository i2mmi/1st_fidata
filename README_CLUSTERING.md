# 주가 데이터 클러스터링 프로젝트

## 📁 프로젝트 구조

```
중랑프로젝트/
├── config.py              # 설정 파일 (폰트 설정 등)
├── data_loader.py         # 데이터 수집 모듈
├── data_processor.py      # 데이터 전처리 (로그 누적 수익률 계산)
├── clustering.py          # 클러스터링 모듈 (최적 K 탐색, K-Means)
├── visualization.py       # 시각화 모듈
├── main.py                # 메인 실행 파일
├── CHART_INTERPRETATION.md # 차트 해석 가이드
└── requirements.txt       # 패키지 의존성
```

## 🔧 모듈 설명

### 1. `config.py`
- **역할**: 전역 설정 관리
- **주요 기능**:
  - 한글 폰트 설정 (macOS/Windows/Linux)
  - 기본 상수 정의

### 2. `data_loader.py`
- **역할**: 데이터 수집
- **주요 함수**:
  - `load_tickers_from_csv()`: CSV에서 종목 리스트 읽기
  - `load_tickers_from_fdr()`: FDR에서 자동 수집
  - `collect_stock_prices()`: 주가 데이터 수집

### 3. `data_processor.py`
- **역할**: 데이터 전처리
- **주요 함수**:
  - `calculate_log_cumulative_returns()`: 로그 누적 수익률 계산
  - `process_stock_data()`: 전체 전처리 파이프라인

### 4. `clustering.py`
- **역할**: 클러스터링 수행
- **주요 함수**:
  - `find_optimal_k()`: 최적 K 값 탐색 (Elbow + Silhouette)
  - `perform_kmeans_clustering()`: K-Means 클러스터링 실행

### 5. `visualization.py`
- **역할**: 결과 시각화
- **주요 함수**:
  - `visualize_clustering_results()`: PCA 2차원 시각화
  - `print_cluster_summary()`: 클러스터별 통계 출력

### 6. `main.py`
- **역할**: 전체 프로세스 통합 및 실행
- **주요 함수**:
  - `run_stock_clustering()`: 전체 파이프라인 실행

## 🚀 사용 방법

### 기본 실행
```python
from main import run_stock_clustering

# 최적 K 자동 탐색
results = run_stock_clustering(
    csv_path='kospi_code_list_100.csv',
    limit=50,
    find_optimal=True,
    max_k=10
)

# 수동으로 K 지정
results = run_stock_clustering(
    csv_path='kospi_code_list_100.csv',
    limit=50,
    n_clusters=5,
    find_optimal=False
)
```

### Streamlit 연동 예시
```python
import streamlit as st
from main import run_stock_clustering

st.title("주가 클러스터링 분석")

# 사용자 입력
csv_file = st.file_uploader("CSV 파일 업로드", type=['csv'])
k_value = st.slider("클러스터 개수", 2, 10, 5)
auto_k = st.checkbox("최적 K 자동 탐색")

if csv_file:
    results = run_stock_clustering(
        csv_path=csv_file.name,
        n_clusters=k_value if not auto_k else 5,
        find_optimal=auto_k
    )
    st.dataframe(results)
```

## 📊 출력 파일

- `clustering_results.csv`: 클러스터링 결과 (Code, Name, Cluster)

## 🔍 주요 개념

### 로그 누적 수익률
```
log_returns = log(P_t / P_0)
- P_t: 시점 t의 주가
- P_0: 초기 주가
```

### PCA (Principal Component Analysis)
- 고차원 데이터를 2차원으로 축소
- 정보 손실 최소화하면서 시각화 가능

### K-Means 클러스터링
- 유사한 패턴을 가진 종목들을 그룹화
- 최적 K는 Elbow Method와 Silhouette Score로 결정

## 📖 참고 문서

- `CHART_INTERPRETATION.md`: 차트 해석 방법 상세 가이드

