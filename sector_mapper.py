"""
섹터 매핑 모듈
kospi_code.csv 파일에서 Corp_Info를 읽어서 섹터로 사용
"""
import pandas as pd
import os


def load_sector_mapping(csv_path='data/kospi_code.csv'):
    """
    CSV 파일에서 종목코드, 종목명, Corp_Info(섹터)를 읽어서 딕셔너리로 반환
    
    Parameters:
    -----------
    csv_path : str
        CSV 파일 경로 (기본값: 'data/kospi_code.csv')
    
    Returns:
    --------
    sector_mapping : dict
        {종목코드: 섹터(Corp_Info)} 형태의 딕셔너리
    names_mapping : dict
        {종목코드: 종목명} 형태의 딕셔너리
    """
    try:
        # UTF-8 또는 CP949 자동 감지, Code는 문자열로 읽어서 앞의 0 유지
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', dtype={'Code': str})
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp949', dtype={'Code': str})
        
        sector_mapping = {}
        names_mapping = {}
        
        for idx, row in df.iterrows():
            # Code를 그대로 사용 (티커는 문자열로 유지)
            code = str(row['Code']).strip().strip('"')
            name = str(row['Name']).strip().strip('"')
            sector = str(row['Corp_Info']).strip().strip('"')
            
            sector_mapping[code] = sector
            names_mapping[code] = name
        
        print(f"✅ 섹터 정보 로드 완료: {len(sector_mapping)}개 종목")
        return sector_mapping, names_mapping
    except Exception as e:
        print(f"❌ 섹터 정보 로드 실패: {e}")
        return {}, {}


def get_sector_from_code(code, sector_mapping):
    """
    종목 코드로부터 섹터 가져오기 (Corp_Info를 직접 사용)
    
    Parameters:
    -----------
    code : str
        종목 코드 (티커는 문자열로 유지)
    sector_mapping : dict
        {종목코드: 섹터(Corp_Info)} 딕셔너리
    
    Returns:
    --------
    sector : str
        섹터명 (Corp_Info 값)
    """
    # Code를 그대로 사용 (티커는 문자열)
    code = str(code).strip()
    
    if code in sector_mapping:
        return sector_mapping[code]
    
    return '기타'


def get_all_sectors_from_mapping(sector_mapping):
    """
    매핑 딕셔너리에서 모든 섹터 리스트 가져오기
    
    Parameters:
    -----------
    sector_mapping : dict
        {종목코드: 섹터(Corp_Info)} 딕셔너리
    
    Returns:
    --------
    sectors : list
        고유 섹터 리스트
    """
    sectors = set(sector_mapping.values())
    return sorted(list(sectors))

