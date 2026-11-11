"""전략 3종 + 비교 기준(벤치마크).

모든 전략 함수는 '목표 비중' 표를 돌려준다.
  - 행: 매매를 결정한 날짜 (그날 종가까지의 정보만 사용)
  - 열: 종목
  - 값: 전체 자산 대비 비중 (0~1). 행의 합이 1보다 작으면 나머지는 현금.
공매도와 레버리지는 쓰지 않는다 (개인 투자자가 그대로 따라 할 수 있는 형태).

매개변수는 원 논문/저서의 값을 그대로 쓰고, 이 데이터에 맞춰 조정하지 않았다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """각 달의 마지막 거래일."""
    dates = pd.Series(index, index=index)
    return pd.DatetimeIndex(dates.groupby(index.to_period("M")).last().values)


def trend_following(prices: pd.DataFrame, sma_months: int = 10) -> pd.DataFrame:
    """추세추종: 월말 종가가 10개월 이동평균보다 높은 종목만 보유한다.

    근거: Faber (2007) "A Quantitative Approach to Tactical Asset Allocation",
          Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum".
    종목마다 1/N 자리를 주고, 추세가 꺾인 종목의 자리는 현금으로 둔다.
    """
    monthly = prices.loc[month_ends(prices.index)]
    sma = monthly.rolling(sma_months).mean()
    in_uptrend = (monthly > sma) & sma.notna()
    weights = in_uptrend.astype(float) / prices.shape[1]
    return weights.iloc[sma_months - 1:]


def cross_sectional_momentum(prices: pd.DataFrame, lookback_months: int = 12,
                             skip_months: int = 1, top_n: int = 3) -> pd.DataFrame:
    """상대 모멘텀: 최근 12개월(직전 1개월 제외) 수익률 상위 종목을 동일 비중으로 보유한다.

    근거: Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers",
          Moskowitz & Grinblatt (1999) "Do Industries Explain Momentum?".
    원 논문은 상위 매수 + 하위 공매도지만, 여기서는 상위 매수만 한다.
    """
    monthly = prices.loc[month_ends(prices.index)]
    momentum = monthly.shift(skip_months) / monthly.shift(lookback_months) - 1
    valid = momentum.notna().all(axis=1)
    rank = momentum.rank(axis=1, ascending=False, method="first")
    weights = (rank <= top_n).astype(float) / top_n
    return weights[valid]


def rsi(prices: pd.DataFrame, period: int = 2) -> pd.DataFrame:
    """Wilder 방식 RSI (0~100). 낮을수록 최근에 많이 떨어졌다는 뜻."""
    change = prices.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    total = avg_gain + avg_loss
    # 변동이 전혀 없는 구간은 중립값 50, 계산 기간이 모자란 구간은 빈 값으로 둔다.
    return (100 * avg_gain / total).where(total > 0, 50.0).where(total.notna())


def mean_reversion(prices: pd.DataFrame, rsi_period: int = 2, entry_below: float = 10,
                   trend_sma: int = 200, exit_sma: int = 5,
                   max_weight: float = 1 / 3) -> pd.DataFrame:
    """평균회귀: 장기 상승 추세 안에서 단기 급락한 종목을 사서 반등하면 판다.

    근거: 단기 반전 효과 - Jegadeesh (1990), Lehmann (1990).
          매매 규칙 - Connors & Alvarez (2009) "Short Term Trading Strategies That Work".
    매수: 종가 > 200일 이동평균 이고 RSI(2) < 10
    매도: 종가 > 5일 이동평균
    보유 종목끼리 자금을 똑같이 나누되 한 종목은 max_weight를 넘지 않는다.
    """
    entry = ((rsi(prices, rsi_period) < entry_below)
             & (prices > prices.rolling(trend_sma).mean())).to_numpy()
    exit_ = (prices > prices.rolling(exit_sma).mean()).to_numpy()

    held = np.zeros(prices.shape, dtype=bool)
    for t in range(1, len(prices)):
        # 보유 중이면 매도 신호가 날 때까지 유지, 아니면 매수 신호가 날 때 진입
        held[t] = np.where(held[t - 1], ~exit_[t], entry[t])

    count = held.sum(axis=1, keepdims=True)
    per_position = np.minimum(max_weight, 1 / np.maximum(count, 1))
    weights = pd.DataFrame(held * per_position, index=prices.index, columns=prices.columns)

    changed = np.zeros(len(prices), dtype=bool)
    changed[1:] = (held[1:] != held[:-1]).any(axis=1)
    return weights[changed]


def equal_weight_buy_and_hold(prices: pd.DataFrame) -> pd.DataFrame:
    """비교 기준: 전 종목을 동일 비중으로 사서 계속 보유 (매월 말 비중만 다시 맞춤)."""
    dates = month_ends(prices.index).insert(0, prices.index[0]).unique()
    return pd.DataFrame(1.0 / prices.shape[1], index=dates, columns=prices.columns)


def buy_and_hold(prices: pd.DataFrame) -> pd.DataFrame:
    """비교 기준: 첫날 전액 매수 후 그대로 보유 (지수 ETF 한 종목용)."""
    return pd.DataFrame(1.0 / prices.shape[1], index=prices.index[:1], columns=prices.columns)
