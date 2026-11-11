"""성과 지표 계산."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown(equity: pd.Series) -> pd.Series:
    """직전 최고점 대비 하락률 (0 이하)."""
    return equity / equity.cummax() - 1


def yearly_returns(equity: pd.Series) -> pd.Series:
    """연도별 수익률 (첫 해와 마지막 해는 일부 기간일 수 있다)."""
    year_end = equity.groupby(equity.index.year).last()
    previous = year_end.shift(1)
    previous.iloc[0] = equity.iloc[0]
    return year_end / previous - 1


def compute_metrics(equity: pd.Series, weights: pd.DataFrame | None = None,
                    turnover: pd.Series | None = None) -> dict[str, float]:
    """자산 곡선 하나에 대한 요약 지표 (무위험 수익률은 0으로 둔다)."""
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    dd = drawdown(equity)
    underwater = dd < 0
    # 최고점을 회복하지 못한 가장 긴 기간 (거래일 수)
    longest = int(underwater.groupby((~underwater).cumsum()).sum().max())
    by_year = yearly_returns(equity)
    spread = returns.std()
    metrics = {
        "final_value": equity.iloc[-1] / equity.iloc[0],
        "cagr": cagr,
        "volatility": spread * np.sqrt(TRADING_DAYS),
        "sharpe": returns.mean() / spread * np.sqrt(TRADING_DAYS) if spread > 0 else np.nan,
        "max_drawdown": dd.min(),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "longest_drawdown_years": longest / TRADING_DAYS,
        "worst_year": by_year.min(),
        "positive_years": (by_year > 0).mean(),
    }
    if weights is not None:
        metrics["exposure"] = weights.sum(axis=1).mean()
    if turnover is not None:
        metrics["annual_turnover"] = turnover.sum() / years
    return {key: float(value) for key, value in metrics.items()}
