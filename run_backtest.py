"""설정 파일(configs/*.json) 하나에 대해 전략들과 벤치마크를 백테스트한다.

사용법: python run_backtest.py --config configs/us_sectors.json [--refresh]
결과:   results/<이름>/report.md, equity.png, drawdown.png, metrics.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant import report as rp
from quant import strategies as st
from quant.backtest import run_backtest
from quant.data import clean_prices, load_prices
from quant.metrics import compute_metrics


@dataclass
class Entry:
    key: str
    name: str          # 보고서에 쓰는 이름
    chart_label: str   # 차트에 쓰는 이름 (영어)
    prices: pd.DataFrame
    targets: pd.DataFrame
    color: str
    is_strategy: bool = True
    linestyle: str = "-"
    linewidth: float = 1.6


def build_entries(prices: pd.DataFrame, index_prices: pd.DataFrame,
                  benchmark: dict) -> list[Entry]:
    """전략 6개와 비교 기준 2개. 색은 전략마다 고정이다."""
    top_n = max(1, round(prices.shape[1] / 3))  # 상위 1/3 종목에 집중
    return [
        Entry("trend", "추세추종", "Trend following (10-month SMA)", prices,
              st.trend_following(prices), "#2a78d6"),
        Entry("momentum", "모멘텀", f"Momentum (12-1 month, top {top_n})", prices,
              st.cross_sectional_momentum(prices, top_n=top_n), "#eb6834"),
        Entry("dual", "듀얼 모멘텀", "Dual momentum (momentum + trend filter)", prices,
              st.dual_momentum(prices, top_n=top_n), "#eda100"),
        Entry("meanrev", "평균회귀", "Mean reversion (RSI-2 dip buying)", prices,
              st.mean_reversion(prices, max_weight=1 / top_n), "#1baf7a"),
        Entry("lowvol", "저변동성", f"Low volatility (calmest {top_n})", prices,
              st.low_volatility(prices, top_n=top_n), "#e87ba4"),
        Entry("voltarget", "변동성 조절", "Volatility targeting (10%)", prices,
              st.volatility_targeting(prices), "#008300"),
        Entry("equal", "동일비중 보유", "Equal-weight buy & hold", prices,
              st.equal_weight_buy_and_hold(prices), "#52514e", False, "-", 1.2),
        Entry("index", benchmark["label"], benchmark["chart_label"], index_prices,
              st.buy_and_hold(index_prices), "#898781", False, "--", 1.2),
    ]


def evaluate(entries: list[Entry], eval_start: pd.Timestamp, cost_bps: float, delay: int) -> dict:
    """모든 항목을 같은 가정으로 백테스트하고, 같은 시작일 기준으로 맞춘다."""
    results = {}
    for entry in entries:
        result = run_backtest(entry.prices, entry.targets, cost_bps=cost_bps, delay=delay)
        equity = result.equity.loc[eval_start:]
        equity = equity / equity.iloc[0]
        metrics = compute_metrics(equity, result.weights.loc[eval_start:],
                                  result.turnover.loc[eval_start:].iloc[1:])
        results[entry.key] = (equity, metrics)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--refresh", action="store_true", help="가격 데이터를 새로 내려받는다")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tickers = list(config["tickers"])
    index_ticker = config["benchmark"]["ticker"]
    cost_bps = float(config["cost_bps"])

    raw = load_prices(tickers + [index_ticker], config["start"],
                      Path(args.data_dir) / f"{config['name']}.csv", refresh=args.refresh)
    all_prices, notes = clean_prices(raw)
    prices, index_prices = all_prices[tickers], all_prices[[index_ticker]]
    entries = build_entries(prices, index_prices, config["benchmark"])
    strategies = [entry for entry in entries if entry.is_strategy]

    # 모든 전략이 판단을 내릴 수 있게 된 다음 날부터, 같은 기간으로 평가한다.
    last_ready = max(entry.targets.index[0] for entry in strategies)
    eval_start = prices.index[prices.index.get_loc(last_ready) + 1]

    base = evaluate(entries, eval_start, cost_bps, delay=1)
    sensitivity = {(delay, cost): evaluate(strategies, eval_start, cost, delay)
                   for delay in (1, 0) for cost in (0.0, cost_bps, 3 * cost_bps)}
    latest_targets = [(entry.name, f"{entry.targets.index[-1]:%Y-%m-%d}",
                       rp.describe_weights(entry.targets.iloc[-1])) for entry in strategies]

    out_dir = Path(args.results_dir) / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rp.plot_equity(entries, base, out_dir / "equity.png", f"{config['chart_title']}: growth of 1")
    rp.plot_drawdowns(entries, base, out_dir / "drawdown.png",
                      f"{config['chart_title']}: drawdown from previous peak")
    report = rp.build_report(config, entries, base, sensitivity, cost_bps, latest_targets, notes)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    pd.DataFrame({entry.key: base[entry.key][1] for entry in entries}).round(4).to_csv(
        out_dir / "metrics.csv")
    print(report)


if __name__ == "__main__":
    main()
