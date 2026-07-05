import os
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import ta
import pyupbit
from openai import OpenAI

# ---------------------------------------------------------------------------
# [환경 설정 및 API 키]
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
}

# ---------------------------------------------------------------------------
# [0단계: 업비트 한글 종목명 딕셔너리 생성 (오류 수정 부분)]
# ---------------------------------------------------------------------------
def get_korean_name_dict():
    """업비트 API를 통해 티커와 한글 종목명 딕셔너리 생성"""
    url = "https://api.upbit.com/v1/market/all"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return {item['market']: item['korean_name'] for item in res.json()}
    except Exception as e:
        print(f"⚠️ 종목명 로드 실패: {e}")
    return {}

# ---------------------------------------------------------------------------
# [1단계: 업비트 변동성/거래량 상위 종목 추출]
# ---------------------------------------------------------------------------
def get_volatile_tickers(limit=60):
    """업비트 원화(KRW) 마켓에서 24시간 거래대금 기준 상위 종목 추출"""
    print("🔄 1단계: 업비트 거래대금/변동성 상위 종목 탐색 중...")
    tickers = pyupbit.get_tickers(fiat="KRW")
    
    url = "https://api.upbit.com/v1/ticker"
    res = requests.get(url, headers={"accept": "application/json"}, params={"markets": ",".join(tickers)})
    data = res.json()
    
    # 24시간 거래대금(acc_trade_price_24h) 기준으로 내림차순 정렬
    sorted_data = sorted(data, key=lambda x: x['acc_trade_price_24h'], reverse=True)
    return [item['market'] for item in sorted_data[:limit]]

# ---------------------------------------------------------------------------
# [2단계: 차트 분석 및 상승 잠재력 스코어링 (1주~1달 관점)]
# ---------------------------------------------------------------------------
def analyze_chart_score(ticker):
    """
    일봉(day) 데이터를 기준으로 기술적 지표를 계산하여 '상승 잠재력 점수'를 반환
    """
    # 1주~1달 관점이므로 일봉(day) 사용
    df = pyupbit.get_ohlcv(ticker, interval="day", count=60)
    if df is None or len(df) < 30:
        return -999, 0
        
    # 기술적 지표 계산
    rsi = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.macd_diff(df['close'])
    bb_low = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    
    current_price = df['close'].iloc[-1]
    
    score = 0
    # 1. RSI 반등: 과매도 구간(40 이하)에서 고개를 들었을 때 가점
    if rsi.iloc[-2] < 40 and rsi.iloc[-1] > rsi.iloc[-2]:
        score += 30
    elif rsi.iloc[-1] >= 40 and rsi.iloc[-1] <= 60: # 안정적인 상승세
        score += 15
        
    # 2. MACD 크로스: MACD 히스토그램이 음수에서 양수로 전환 (골든크로스)
    if macd.iloc[-2] <= 0 and macd.iloc[-1] > 0:
        score += 40
    elif macd.iloc[-1] > 0: # 상승 추세 유지
        score += 10
        
    # 3. 볼린저 밴드 지지: 하단 밴드에 근접하여 반등할 가능성
    if current_price <= bb_low.iloc[-1] * 1.05:
        score += 30
        
    return score, current_price

# ---------------------------------------------------------------------------
# [3단계: 뉴스 크롤링 및 AI 분석]
# ---------------------------------------------------------------------------
def get_coin_news(coin_name):
    """ 네이버 뉴스에서 해당 코인 관련 최신 기사 수집 """
    url = f"https://search.naver.com/search.naver?where=news&query={coin_name} 가상화폐"
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    news_titles = [title.text.strip() for title in soup.find_all('a', class_='news_tit')[:3]]
    return "\n".join(news_titles) if news_titles else "최신 뉴스가 없습니다."

def ai_analyze_reason(coin_name, news_text):
    """ GPT를 통해 뉴스를 분석하여 오를 만한 이유 1줄 요약 """
    if not client:
        return "AI 분석 불가 (API 키 없음)"
        
    prompt = f"""
    당신은 가상화폐 전문 분석가입니다. 
    최근 '{coin_name}' 코인에 대한 다음 뉴스를 바탕으로, 앞으로 1주일~1달 내에 상승할 가능성이 있는 핵심 이유(호재)를 분석하세요.
    뉴스:
    {news_text}
    
    반드시 다음 형식으로 1~2줄로 짧고 명확하게 답변하세요. 이유가 명확하지 않다면 기술적 반등이라고 작성하세요.
    [상승 이유]: (여기에 내용 작성)
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return "[상승 이유]: AI 분석 중 오류 발생"

# ---------------------------------------------------------------------------
# [4단계: 텔레그램 알림 발송]
# ---------------------------------------------------------------------------
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ 텔레그램 토큰 미설정")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})

# ---------------------------------------------------------------------------
# [메인 실행]
# ---------------------------------------------------------------------------
def main():
    print("🚀 가상화폐(업비트) 상승 예측 봇 가동 🚀")
    
    # 추가된 부분: 프로그램 시작 시 종목명 사전 불러오기
    korean_names = get_korean_name_dict()
    
    tickers = get_volatile_tickers(limit=60) # 상위 60개 탐색
    
    scored_coins = []
    print("🔍 2단계: 차트 스코어링 진행 중...")
    for ticker in tickers:
        score, price = analyze_chart_score(ticker)
        if score > 0:
            # 수정된 부분: 불러온 사전에서 한글명 찾기
            coin_name = korean_names.get(ticker, ticker)
            scored_coins.append({"ticker": ticker, "name": coin_name, "score": score, "price": price})
        time.sleep(0.1) # 업비트 API 호출 제한 방지
        
    # 점수 높은 순으로 상위 30개 자르기
    scored_coins.sort(key=lambda x: x['score'], reverse=True)
    top_30 = scored_coins[:30]
    
    print(f"📰 3단계: 상위 {len(top_30)}개 종목 뉴스 분석 중...")
    
    # 텔레그램 메시지 길이 제한(4096자)을 우회하기 위해 메시지를 10개씩 분할 전송
    chunk_size = 10
    for i in range(0, len(top_30), chunk_size):
        chunk = top_30[i:i + chunk_size]
        report = f"📈 **업비트 상승 예측 Top {i+1}~{i+len(chunk)}**\n\n"
        
        for rank, coin in enumerate(chunk, start=i+1):
            news = get_coin_news(coin['name'])
            ai_reason = ai_analyze_reason(coin['name'], news)
            
            report += f"**{rank}. {coin['name']}** ({coin['ticker']})\n"
            report += f"▪ 현재가: {coin['price']:,.0f}원 (차트점수: {coin['score']}점)\n"
            report += f"▪ {ai_reason}\n"
            report += "-------------------------\n"
            
        send_telegram_message(report)
        time.sleep(1) # 분할 전송 간 딜레이

if __name__ == "__main__":
    main()
