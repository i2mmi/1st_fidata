"""
데이터 전처리 모듈
- 로그 누적 수익률 계산
- 데이터 정제 및 변환
"""
import pandas as pd
import numpy as np


def calculate_log_cumulative_returns(prices_df):
    """
    주가 데이터로부터 로그 누적 수익률 계산
    
    로그 누적 수익률 = log(P_t / P_0)
    - P_t: 시점 t의 주가
    - P_0: 초기 주가 (첫 번째 시점)
    
    Parameters:
    -----------
    prices_df : DataFrame
        주가 데이터프레임 (열: 종목, 행: 날짜)
    
    Returns:
    --------
    log_returns_df : DataFrame
        로그 누적 수익률 데이터프레임 (열: 종목, 행: 날짜)
        형태: [행: 날짜, 열: 종목]
    """
    # NaN이 포함된 종목 제거
    prices_df = prices_df.dropna(axis=1)
    
    if prices_df.empty:
        return None
    
    # 로그 누적 수익률 계산
    log_returns_dict = {}
    
    for ticker in prices_df.columns:
        prices = prices_df[ticker]
        
        # 초기 가격 (첫 번째 값)
        initial_price = prices.iloc[0]
        
        # 로그 누적 수익률: log(P_t / P_0)
        log_cumulative_returns = np.log(prices / initial_price)
        
        log_returns_dict[ticker] = log_cumulative_returns
    
    # DataFrame으로 변환
    log_returns_df = pd.DataFrame(log_returns_dict)
    
    return log_returns_df


def prepare_clustering_data(log_returns_df):
    """
    클러스터링을 위한 데이터 준비
    [행: 종목, 열: 날짜] 형태로 변환
    
    Parameters:
    -----------
    log_returns_df : DataFrame
        로그 누적 수익률 (행: 날짜, 열: 종목)
    
    Returns:
    --------
    movements : DataFrame
        클러스터링 데이터 (행: 종목, 열: 날짜)
    """
    if log_returns_df is None or log_returns_df.empty:
        return None
    
    # Transpose: [행: 종목, 열: 날짜] 형태로 변환
    movements = log_returns_df.T
    
    # 결측치가 있는 종목 제거
    movements = movements.dropna(axis=0)
    
    return movements


def process_stock_data(stock_data_dict):
    """
    주가 딕셔너리를 받아서 로그 누적 수익률로 변환
    
    Parameters:
    -----------
    stock_data_dict : dict
        {종목코드: Series(Close 가격)} 형태
    
    Returns:
    --------
    movements : DataFrame
        클러스터링 데이터 (행: 종목, 열: 날짜)
    """
    if not stock_data_dict:
        return None
    
    # DataFrame으로 변환
    prices_df = pd.DataFrame(stock_data_dict)
    
    # 로그 누적 수익률 계산
    log_returns_df = calculate_log_cumulative_returns(prices_df)
    
    if log_returns_df is None:
        return None
    
    # 클러스터링 데이터 준비
    movements = prepare_clustering_data(log_returns_df)
    
    return movements

