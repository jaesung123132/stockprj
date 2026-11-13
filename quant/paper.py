"""가상 매매(페이퍼 트레이딩).

실제 주문은 내지 않는다. 정해진 시작일에 시작 자금을 넣고 전략을 그대로 따랐다면
지금 계좌가 어떤 모습일지를, 실행할 때마다 최신 가격으로 처음부터 다시 계산한다.
상태 파일을 따로 두지 않으므로 실행이 빠지거나 겹쳐도 결과가 달라지지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest


def confirmed_targets(targets: pd.DataFrame, last_date: pd.Timestamp) -> pd.DataFrame:
    """확정된 신호만 남긴다.

    데이터의 마지막 날에 계산된 신호는 그 달이 끝났는지 아직 알 수 없으므로 뺀다.
    (다음 거래일 가격이 들어오면 확정되고, 그날 종가에 체결된다.)
    """
    return targets.loc[targets.index < last_date]


def paper_targets(targets: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """시작일에는 그때 유효한(직전에 확정된) 신호로 진입하고, 이후에는 새 신호를 따른다."""
    before = targets.loc[targets.index < start]
    after = targets.loc[targets.index >= start]
    if len(after) and after.index[0] == start:
        return after
    if before.empty:
        raise ValueError("시작일 이전에 확정된 신호가 없습니다.")
    entry = before.iloc[[-1]].copy()
    entry.index = pd.DatetimeIndex([start])
    return pd.concat([entry, after])


def run_paper(prices: pd.DataFrame, targets: pd.DataFrame, start: str,
              cost_bps: float) -> tuple[BacktestResult, pd.DataFrame] | None:
    """start 이후 첫 거래일부터 가상 계좌를 굴린다. 아직 그날이 오지 않았으면 None.

    돌려주는 값: (계좌 기록, 따른 신호 표). 계좌 가치는 시작 = 1 기준이다.
    신호 다음 거래일 종가에 체결하므로, 시작일 당일은 전액 현금이다.
    """
    days = prices.index[prices.index >= pd.Timestamp(start)]
    if days.empty:
        return None
    plan = paper_targets(confirmed_targets(targets, prices.index[-1]), days[0])
    return run_backtest(prices.loc[days[0]:], plan, cost_bps=cost_bps, delay=1), plan


def list_trades(result: BacktestResult,
                min_change: float = 0.01) -> list[tuple[pd.Timestamp, list[tuple[str, float, float]]]]:
    """체결 내역: (날짜, [(종목, 이전 비중, 이후 비중), ...]). 비중이 1%p 넘게 바뀐 것만."""
    weights = result.weights
    trades = []
    for pos in np.flatnonzero(result.turnover.to_numpy() > 0):
        before, after = weights.iloc[pos - 1], weights.iloc[pos]
        changes = [(ticker, float(before[ticker]), float(after[ticker]))
                   for ticker in weights.columns
                   if abs(after[ticker] - before[ticker]) >= min_change]
        if changes:
            trades.append((weights.index[pos], changes))
    return trades
