"""
설정 파일
- 폰트 설정
- 기본 상수 설정
"""
import matplotlib.pyplot as plt
import platform
import os


def setup_fonts():
    """시각화를 위한 한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    elif system == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    else:  # Linux
        plt.rc('font', family='NanumGothic')
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    
    return True


def get_korean_font_path():
    """
    시스템에 맞는 한글 폰트 경로 반환 (WordCloud용)
    
    Returns:
    --------
    font_path : str or None
        폰트 경로, 찾을 수 없으면 None
    """
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        # macOS의 기본 한글 폰트 경로
        possible_paths = [
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',           
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',   
            '/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/AppleGothic.ttc',
            '/Library/Fonts/NanumBarunGothic.ttf',                 
            '/Library/Fonts/NanumGothic.ttf'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
    elif system == 'Windows':
        # Windows의 기본 한글 폰트 경로
        possible_paths = [
            'C:/Windows/Fonts/malgun.ttf',
            'C:/Windows/Fonts/gulim.ttc'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
    else:  # Linux
        # Linux의 한글 폰트 경로
        possible_paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
    
    return None


# 기본 설정 값
DEFAULT_LIMIT = 100
DEFAULT_N_CLUSTERS = 8
DEFAULT_MAX_K = 10
DEFAULT_YEARS = 1  # 데이터 수집 기간 (년)

