"""
데이터 수집 모듈
- CSV 파일에서 종목 리스트 읽기
- FinanceDataReader를 통한 주가 데이터 수집
"""
import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import tqdm


def load_tickers_from_csv(csv_path):
    """
    CSV 파일에서 종목 코드와 종목명 읽기
    
    Parameters:
    -----------
    csv_path : str
        CSV 파일 경로 (Code, Name 컬럼 포함)
    
    Returns:
    --------
    tickers : list
        종목 코드 리스트 (문자열 형식)
    names : dict
        종목 코드를 키로, 종목명을 값으로 하는 딕셔너리
    """
    try:
        # Code를 문자열로 읽어서 앞의 0이 제거되지 않도록 함
        df_csv = pd.read_csv(csv_path, dtype={'Code': str})
        # Code를 그대로 사용 (티커는 문자열)
        tickers = df_csv['Code'].tolist()
        names = {}
        
        if 'Name' in df_csv.columns:
            for idx, row in df_csv.iterrows():
                code = str(row['Code'])  # 문자열로 유지
                names[code] = row['Name']
            print(f"✅ CSV 파일에서 {len(tickers)}개의 종목을 읽었습니다. (종목명 포함)")
        else:
            print(f"✅ CSV 파일에서 {len(tickers)}개의 종목을 읽었습니다. (종목명 없음)")
        
        return tickers, names
    except Exception as e:
        print(f"❌ CSV 파일 읽기 실패: {e}")
        return None, None


def load_tickers_from_fdr(limit=100):
    """
    FDR에서 KOSPI 종목 리스트 자동 수집
    
    Parameters:
    -----------
    limit : int
        가져올 종목 수 제한
    
    Returns:
    --------
    tickers : list
        종목 코드 리스트
    names : dict
        종목 코드를 키로, 종목명을 값으로 하는 딕셔너리
    """
    print("📊 KOSPI 상위 종목을 자동으로 가져옵니다.")
    df_krx = fdr.StockListing('KOSPI')
    
    names = {}
    for idx, row in df_krx.iterrows():
        names[str(row['Code'])] = row['Name']
    
    tickers = df_krx['Code'].head(limit).astype(str).tolist()
    return tickers, names


def collect_stock_prices(tickers, years=1, min_data_points=30):
    """
    종목 코드 리스트로부터 주가 데이터 수집
    
    Parameters:
    -----------
    tickers : list
        종목 코드 리스트
    years : int
        수집할 기간 (년)
    min_data_points : int
        최소 데이터 포인트 수 (너무 적은 데이터는 제외, 기본값: 50)
    
    Returns:
    --------
    stock_data : dict
        {종목코드: Series(Close 가격)} 형태의 딕셔너리
    """
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    stock_data = {}
    failed_tickers = []
    
    print(f"\n📈 {len(tickers)}개 종목의 데이터 수집 중... (기간: {start_date} ~ {end_date})")
    
    for ticker in tqdm.tqdm(tickers, desc="데이터 수집"):
        try:
            # Code를 그대로 티커 번호로 사용 (문자열로 유지)
            ticker_str = str(ticker).strip()
            df = fdr.DataReader(ticker_str, start_date, end_date)
            
            if df is None or df.empty:
                failed_tickers.append(ticker)
                continue
            
            # 코스피 종목이므로 최소 데이터 포인트를 낮춤 (너무 적은 데이터만 제외)
            if len(df) >= min_data_points:
                stock_data[ticker_str] = df['Close']
            else:
                failed_tickers.append(ticker)
                
        except Exception as e:
            failed_tickers.append(ticker)
            continue
    
    if not stock_data:
        print("❌ 수집된 데이터가 없습니다.")
        return None
    
    print(f"✅ {len(stock_data)}개 종목의 데이터 수집 완료")
    if failed_tickers:
        print(f"⚠️  {len(failed_tickers)}개 종목 데이터 수집 실패 또는 데이터 부족")
    
    return stock_data


def get_stock_name(code, names_dict):
    """
    종목 코드로부터 종목명 가져오기 (없으면 FDR에서 조회)
    
    Parameters:
    -----------
    code : str
        종목 코드
    names_dict : dict
        종목명 매핑 딕셔너리
    
    Returns:
    --------
    name : str
        종목명
    """
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

