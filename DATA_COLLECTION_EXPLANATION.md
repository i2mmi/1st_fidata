# 데이터 수집 코드 설명

## 📚 개요

`data_loader.py` 모듈은 주가 데이터를 수집하기 위한 4개의 주요 함수를 제공합니다:
1. CSV 파일에서 종목 리스트 읽기
2. FDR에서 종목 리스트 자동 수집
3. 주가 데이터 수집
4. 종목명 조회

---

## 🔍 함수별 상세 설명

### 1. `load_tickers_from_csv(csv_path)` - CSV 파일에서 종목 읽기

```python
def load_tickers_from_csv(csv_path):
    df_csv = pd.read_csv(csv_path)
    tickers = df_csv['Code'].astype(str).tolist()
    names = {}
    
    if 'Name' in df_csv.columns:
        for idx, row in df_csv.iterrows():
            names[str(row['Code'])] = row['Name']
    
    return tickers, names
```

**역할**: CSV 파일에서 종목 코드와 종목명을 읽어옵니다.

**처리 과정**:
1. `pd.read_csv()`로 CSV 파일 읽기
2. `Code` 컬럼에서 종목 코드 리스트 생성 (문자열로 변환)
3. `Name` 컬럼이 있으면 종목 코드와 이름을 매핑하는 딕셔너리 생성
4. 종목 코드 리스트와 이름 딕셔너리 반환

**입력 예시 (CSV)**:
```csv
Code,Name
005930,삼성전자
000660,SK하이닉스
035420,NAVER
```

**출력 예시**:
- `tickers`: `['005930', '000660', '035420']`
- `names`: `{'005930': '삼성전자', '000660': 'SK하이닉스', '035420': 'NAVER'}`

---

### 2. `load_tickers_from_fdr(limit=100)` - FDR에서 종목 자동 수집

```python
def load_tickers_from_fdr(limit=100):
    df_krx = fdr.StockListing('KOSPI')
    
    names = {}
    for idx, row in df_krx.iterrows():
        names[str(row['Code'])] = row['Name']
    
    tickers = df_krx['Code'].head(limit).astype(str).tolist()
    return tickers, names
```

**역할**: FinanceDataReader(FDR) 라이브러리를 사용해 KOSPI 상장 종목을 자동으로 가져옵니다.

**처리 과정**:
1. `fdr.StockListing('KOSPI')`로 KOSPI 전체 종목 리스트 가져오기
2. 각 종목의 Code와 Name을 매핑 딕셔너리 생성
3. `limit` 파라미터만큼 상위 종목만 선택
4. 종목 코드 리스트와 이름 딕셔너리 반환

**특징**:
- 인터넷 연결 필요
- KOSPI 전체 종목을 한 번에 가져온 후 상위 N개만 선택
- CSV 파일 없이도 바로 사용 가능

---

### 3. `collect_stock_prices(tickers, years=1, min_data_points=200)` - 주가 데이터 수집

```python
def collect_stock_prices(tickers, years=1, min_data_points=200):
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    stock_data = {}
    for ticker in tqdm.tqdm(tickers, desc="데이터 수집"):
        try:
            df = fdr.DataReader(ticker, start_date, end_date)
            
            if len(df) > min_data_points:
                stock_data[ticker] = df['Close']
        except Exception as e:
            continue
    
    return stock_data
```

**역할**: 각 종목 코드에 대해 과거 주가 데이터를 수집합니다.

**처리 과정**:

1. **날짜 계산**
   ```python
   start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
   end_date = datetime.now().strftime('%Y-%m-%d')
   ```
   - 예: `years=1`이면 1년 전부터 오늘까지
   - 결과: `start_date='2024-01-15'`, `end_date='2025-01-15'`

2. **각 종목별로 데이터 수집**
   ```python
   for ticker in tqdm.tqdm(tickers, desc="데이터 수집"):
       df = fdr.DataReader(ticker, start_date, end_date)
   ```
   - `fdr.DataReader()`: FinanceDataReader로 해당 종목의 주가 데이터 다운로드
   - 반환 형태: DataFrame (Date, Open, High, Low, Close, Volume 컬럼)
   - `tqdm`: 진행률 표시 (예: `[████████░░] 80%`)

3. **데이터 품질 검증**
   ```python
   if len(df) > min_data_points:
       stock_data[ticker] = df['Close']
   ```
   - 최소 데이터 포인트 수 확인 (기본값: 200개)
   - 너무 적은 데이터는 제외 (거래 정지, 상장폐지 등)
   - `Close` (종가)만 추출하여 저장

4. **에러 처리**
   ```python
   except Exception as e:
       continue
   ```
   - 데이터 수집 실패 시 해당 종목은 건너뛰고 다음 종목 계속
   - 모든 종목이 실패해도 프로그램은 중단되지 않음

**출력 예시**:
```python
stock_data = {
    '005930': Series([72000, 72500, 73000, ...], index=[Date1, Date2, Date3, ...]),
    '000660': Series([120000, 121000, 119000, ...], index=[Date1, Date2, Date3, ...]),
    ...
}
```

**주의사항**:
- 인터넷 연결 필수
- 종목 수가 많으면 시간이 오래 걸릴 수 있음
- API 제한이 있을 수 있음

---

### 4. `get_stock_name(code, names_dict)` - 종목명 조회

```python
def get_stock_name(code, names_dict):
    if code in names_dict:
        return names_dict[code]
    
    # FDR에서 조회
    try:
        stock_info = fdr.StockListing('KRX')
        stock_info = stock_info[stock_info['Code'] == code]
        if not stock_info.empty:
            return stock_info.iloc[0]['Name']
    except:
        pass
    
    return code
```

**역할**: 종목 코드로부터 종목명을 가져옵니다 (백업 메커니즘 포함).

**처리 과정**:

1. **먼저 메모리의 딕셔너리에서 찾기**
   ```python
   if code in names_dict:
       return names_dict[code]
   ```
   - CSV나 FDR에서 이미 읽은 종목명 딕셔너리에서 먼저 검색
   - 빠르고 API 호출 없음

2. **없으면 FDR에서 실시간 조회**
   ```python
   stock_info = fdr.StockListing('KRX')
   stock_info = stock_info[stock_info['Code'] == code]
   ```
   - KRX 전체 종목 리스트를 가져와서 해당 코드 검색
   - 인터넷 연결 필요

3. **그래도 없으면 종목 코드 반환**
   ```python
   return code
   ```
   - 모든 방법이 실패하면 종목 코드 자체를 반환

**사용 예시**:
```python
names = {'005930': '삼성전자'}
get_stock_name('005930', names)  # → '삼성전자'
get_stock_name('999999', names)  # → '999999' (없는 코드)
```

---

## 🔄 전체 데이터 수집 흐름

```python
# 1. 종목 리스트 가져오기
tickers, names = load_tickers_from_csv('kospi_code_list_100.csv')
# 또는
tickers, names = load_tickers_from_fdr(limit=50)

# 2. 주가 데이터 수집
stock_data = collect_stock_prices(tickers, years=1, min_data_points=200)
# 결과: {종목코드: Close 가격 Series}

# 3. 종목명 조회 (필요시)
name = get_stock_name('005930', names)
```

---

## 📊 실제 데이터 예시

### 입력: 종목 코드 리스트
```python
tickers = ['005930', '000660', '035420']
```

### 처리: 1년간 주가 데이터 수집
```python
stock_data = collect_stock_prices(tickers, years=1)
```

### 출력: 종목별 종가 데이터
```python
stock_data = {
    '005930': 
        Date
        2024-01-15    72000
        2024-01-16    72500
        2024-01-17    73000
        ...
        2025-01-15    85000
        Name: Close, Length: 245,
    
    '000660': 
        Date
        2024-01-15    120000
        2024-01-16    121000
        ...
        Name: Close, Length: 245,
    
    '035420': ...
}
```

---

## ⚙️ 파라미터 설정 가이드

### `years` 파라미터
- **1년 (기본값)**: 최근 1년 데이터
- **0.5년**: 최근 6개월 데이터 (빠른 테스트)
- **3년**: 장기 패턴 분석

### `min_data_points` 파라미터
- **200 (기본값)**: 약 1년치 데이터 (주말 제외)
- **100**: 약 6개월치 데이터
- **50**: 약 3개월치 데이터

---

## ⚠️ 주의사항

1. **인터넷 연결 필수**: FDR은 온라인 API 사용
2. **시간 소요**: 종목 수에 비례하여 시간이 걸림 (100개 종목 ≈ 1-2분)
3. **에러 처리**: 일부 종목 실패해도 계속 진행
4. **데이터 품질**: 최소 데이터 포인트 미달 종목은 자동 제외

---

## 🔧 개선 가능한 부분

1. **캐싱**: 수집한 데이터를 파일로 저장하여 재사용
2. **병렬 처리**: 여러 종목을 동시에 수집
3. **재시도 로직**: 실패 시 자동 재시도
4. **프로그레스 저장**: 중간에 중단되어도 재개 가능

