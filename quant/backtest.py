"""백테스트 엔진: 목표 비중 표를 실제 매매로 옮겨 자산 곡선을 계산한다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series      # 일별 자산 가치 (시작 = 1)
    weights: pd.DataFrame  # 일별 실제 보유 비중 (장 마감 기준)
    turnover: pd.Series    # 일별 매매 금액 / 자산 (매수+매도 합)


def run_backtest(prices: pd.DataFrame, targets: pd.DataFrame,
                 cost_bps: float = 10.0, delay: int = 1) -> BacktestResult:
    """목표 비중대로 매매했을 때의 성과를 계산한다.

    prices   : 수정 종가 (빈 값 없음)
    targets  : 전략이 만든 목표 비중 표 (행 = 결정일)
    cost_bps : 매매 금액의 1만 분의 몇을 비용(수수료+세금+슬리피지)으로 낼지. 매수·매도 각각 부과.
    delay    : 결정일로부터 며칠 뒤 종가에 체결하는지.
               1 = 오늘 종가로 신호 계산 -> 다음 거래일 종가에 체결 (보수적, 기본값)
               0 = 신호를 계산한 바로 그 종가에 체결 (낙관적)
    현금에는 이자가 붙지 않는다고 가정한다.
    """
    if prices.isna().any().any():
        raise ValueError("가격 데이터에 빈 값이 있습니다. clean_prices를 먼저 거치세요.")
    if delay < 0:
        raise ValueError("delay는 0 이상이어야 합니다 (미래 정보 사용 방지).")
    targets = targets.reindex(columns=prices.columns).fillna(0.0)
    if (targets.to_numpy() < -1e-12).any() or (targets.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("목표 비중은 0 이상이고 합이 1 이하여야 합니다 (공매도·레버리지 없음).")
    positions = prices.index.get_indexer(targets.index)
    if (positions < 0).any():
        raise ValueError("목표 비중의 날짜가 가격 데이터에 없습니다.")

    n_days, n_assets = prices.shape
    scheduled = {int(p) + delay: row for p, row in zip(positions, targets.to_numpy())
                 if p + delay < n_days}
    daily_returns = prices.pct_change().fillna(0.0).to_numpy()
    cost_rate = cost_bps / 10_000

    asset_value = np.zeros(n_assets)
    cash = 1.0
    equity = np.empty(n_days)
    turnover = np.zeros(n_days)
    weights = np.zeros((n_days, n_assets))

    for t in range(n_days):
        asset_value = asset_value * (1.0 + daily_returns[t])  # 보유분 평가
        total = asset_value.sum() + cash
        target = scheduled.get(t)
        if target is not None:                                # 오늘 종가에 리밸런싱
            traded = np.abs(target * total - asset_value).sum()
            turnover[t] = traded / total
            total -= traded * cost_rate
            asset_value = target * total
            cash = total - asset_value.sum()
        equity[t] = total
        weights[t] = asset_value / total

    return BacktestResult(
        equity=pd.Series(equity, index=prices.index, name="equity"),
        weights=pd.DataFrame(weights, index=prices.index, columns=prices.columns),
        turnover=pd.Series(turnover, index=prices.index, name="turnover"),
    )
