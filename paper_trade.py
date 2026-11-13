"""가상 매매 현황을 갱신한다 (실제 주문은 내지 않는다).

사용법: python paper_trade.py --config configs/us_sectors.json
결과:   paper/<이름>/status.md (현재 상태), paper/<이름>/log.csv (기준일마다 한 줄),
        paper/<이름>/notice.md (설정에서 notify를 켠 경우: 가장 최근 월말 신호의 매매 안내)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant import strategies as st
from quant.backtest import run_backtest
from quant.data import clean_prices, download_prices
from quant.metrics import drawdown
from quant.paper import list_trades, run_paper
from quant.report import describe_weights, md_table, pct

SMA_MONTHS = 10
# 미국 장 마감(겨울 21:00 UTC, 여름 20:00 UTC) 뒤에만 그날 가격을 종가로 믿는다.
CLOSE_UTC = pd.Timedelta(hours=21, minutes=10)


def money(value: float, unit: str) -> str:
    return f"{value:,.0f}{unit}"


def update_log(path: Path, row: dict) -> None:
    """기준일마다 한 줄. 같은 기준일로 다시 실행하면 그 줄을 덮어쓴다."""
    log = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, dtype={"data_date": str})
        log = pd.concat([old[old["data_date"] != row["data_date"]], log])
    log.to_csv(path, index=False)


def latest_signal_date(targets: pd.DataFrame, last_date: pd.Timestamp) -> pd.Timestamp:
    """매매 안내를 낼 월말 신호의 날짜.

    마지막 날이 그 달의 마지막 평일이고 장이 이미 마감됐으면 그날 신호를 쓴다.
    아니면 이미 확정된 지난달 말 신호를 쓴다.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if last_date == last_date + pd.offsets.BMonthEnd(0) and now >= last_date + CLOSE_UTC:
        return last_date
    return targets.index[targets.index < last_date][-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="paper")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    paper = config["paper"]
    names = config["tickers"]
    tickers = list(names)
    index_ticker = config["benchmark"]["ticker"]
    cost_bps = float(config["cost_bps"])
    capital, unit = float(paper["capital"]), paper["currency"]

    def label(ticker: str) -> str:
        return f"{names[ticker]}({ticker})"

    def listed(found: set[str]) -> str:
        return ", ".join(label(ticker) for ticker in tickers if ticker in found) or "없음"

    # 가상 매매는 항상 최신 가격을 새로 받는다 (캐시를 쓰지 않는다).
    raw = download_prices(tickers + [index_ticker], config["start"])
    all_prices, notes = clean_prices(raw)
    prices, index_prices = all_prices[tickers], all_prices[[index_ticker]]
    last_date = prices.index[-1]
    targets = st.trend_following(prices, sma_months=SMA_MONTHS)
    account_value = capital

    out_dir = Path(args.out_dir) / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# 가상 매매 현황: {config['title']}",
        "",
        "실제 주문은 내지 않습니다. 추세추종 규칙(10개월 이동평균)을 그대로 따랐을 때의 가상 계좌입니다.",
        "",
        f"- 시작 자금: {money(capital, unit)} (설정한 시작일 {paper['start']})",
        f"- 기준일: {last_date:%Y-%m-%d} (장이 열려 있을 때 실행했다면 장중 가격)",
        f"- 체결 가정: 월말 종가로 신호 확정 → 다음 거래일 종가에 체결, 매수·매도 각각 {cost_bps:g}bp",
    ]

    outcome = run_paper(prices, targets, paper["start"], cost_bps)
    if outcome is None:
        lines += ["", f"아직 시작 전입니다. {paper['start']} 이후 첫 거래일의 가격이 들어오면 시작합니다."]
    else:
        result, plan = outcome
        first_day = result.equity.index[0]
        window, index_window = prices.loc[first_day:], index_prices.loc[first_day:]
        hold = run_backtest(window, st.equal_weight_buy_and_hold(window), cost_bps=cost_bps, delay=1)
        index = run_backtest(index_window, st.buy_and_hold(index_window), cost_bps=cost_bps, delay=1)
        account_value = capital * result.equity.iloc[-1]
        weights = result.weights.iloc[-1]

        rows = [[name, money(capital * r.equity.iloc[-1], unit), pct(r.equity.iloc[-1] - 1, 2),
                 pct(drawdown(r.equity).min(), 1)]
                for name, r in [("추세추종 (가상 계좌)", result), ("동일비중 보유", hold),
                                (config["benchmark"]["label"], index)]]
        lines += [f"- 시작일: {first_day:%Y-%m-%d} (이후 {len(result.equity) - 1}거래일 경과)", "",
                  md_table(["", "평가액", "수익률", "최대 낙폭"], rows)]

        lines += ["", "## 현재 보유", ""]
        rows = []
        for ticker in tickers:
            if weights[ticker] > 1e-9:
                amount = account_value * weights[ticker]
                rows.append([label(ticker), pct(weights[ticker]), money(amount, unit),
                             f"{amount / prices[ticker].iloc[-1]:,.1f}주"])
        cash = 1 - weights.sum()
        rows.append(["현금", pct(cash), money(account_value * cash, unit), "-"])
        lines += [md_table(["종목", "비중", "평가액", "수량(약)"], rows), "",
                  f"따르고 있는 신호: {describe_weights(plan.iloc[-1].rename(index=label))}"]
        if len(result.equity) == 1:
            lines += ["", "아직 매수 전입니다. 다음 거래일 종가에 위 신호대로 처음 매수합니다."]

        lines += ["", "## 체결 내역", ""]
        trades = list_trades(result)
        for day, changes in trades:
            detail = ", ".join(f"{label(ticker)} {before:.0%} → {after:.0%}"
                               for ticker, before, after in changes)
            lines.append(f"- {day:%Y-%m-%d}: {detail}")
        if not trades:
            lines.append("- 아직 없음")

        now_held = set(plan.columns[plan.iloc[-1] > 0])
        next_held = set(targets.columns[targets.iloc[-1] > 0])
        lines += ["", "## 이번 달 말 예상 (미확정)", "",
                  "기준일 가격으로 신호를 미리 계산한 것입니다. 실제 신호는 월말 종가로 확정됩니다.", "",
                  f"- 새로 매수: {listed(next_held - now_held)}",
                  f"- 매도: {listed(now_held - next_held)}",
                  f"- 예상 비중: {describe_weights(targets.iloc[-1].rename(index=label))}"]

        update_log(out_dir / "log.csv", {
            "data_date": f"{last_date:%Y-%m-%d}",
            "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "value": round(account_value, 2),
            "return": round(result.equity.iloc[-1] - 1, 6),
            "hold_return": round(hold.equity.iloc[-1] - 1, 6),
            "index_return": round(index.equity.iloc[-1] - 1, 6),
            "invested": round(weights.sum(), 4),
        })

    lines += ["", "## 데이터 점검", ""] + [f"- {note}" for note in notes]
    status = "\n".join(lines) + "\n"
    (out_dir / "status.md").write_text(status, encoding="utf-8")
    print(status)

    if paper.get("notify"):
        # 직접 주문할 사람을 위한 월 1회 매매 안내. 첫 줄(제목)이 같으면 이미 안내한 것이다.
        signal_date = latest_signal_date(targets, last_date)
        monthly = prices.loc[st.month_ends(prices.index)]
        average = monthly.rolling(SMA_MONTHS).mean()
        signal = targets.loc[signal_date]
        previous = targets.loc[targets.index < signal_date].iloc[-1]
        held, was_held = set(signal.index[signal > 0]), set(previous.index[previous > 0])
        rows = [[label(ticker), f"{monthly.loc[signal_date, ticker]:,.2f}",
                 f"{average.loc[signal_date, ticker]:,.2f}", "보유" if ticker in held else "현금"]
                for ticker in tickers]
        notice = [
            f"# {signal_date:%Y-%m} 월말 신호: 매매 안내 ({config['title']})",
            "",
            f"신호 기준일은 {signal_date:%Y-%m-%d} 종가입니다. 다음 거래일에 아래대로 맞추세요. "
            "이 프로그램은 주문을 내지 않으므로 증권사 앱에서 직접 주문해야 합니다.",
            "",
            "## 할 일",
            "",
            f"- 매도 (전량): {listed(was_held - held)}",
            f"- 새로 매수: {listed(held - was_held)}",
            f"- 그대로 보유: {listed(held & was_held)}",
            f"- 보유 대상 종목마다 총 자금의 {1 / len(tickers):.1%}씩 담고, 나머지는 현금으로 둡니다.",
            f"- 가상 계좌(현재 {money(account_value, unit)}) 기준으로는 "
            f"종목당 약 {money(account_value / len(tickers), unit)}입니다.",
            "",
            "처음 시작한다면 '새로 매수'와 '그대로 보유'에 있는 종목을 모두 사면 됩니다.",
            "",
            "## 판단 근거",
            "",
            md_table(["종목", "기준일 종가", "10개월 평균", "판단"], rows),
            "",
            "종가가 10개월 평균보다 높으면 보유, 아니면 현금입니다. "
            "표의 종가가 증권사 앱에서 보는 가격과 크게 다르면 데이터 오류일 수 있으니 주문하지 말고 확인하세요. "
            "(배당을 반영한 가격이라 조금 다를 수는 있습니다.)",
        ]
        (out_dir / "notice.md").write_text("\n".join(notice) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
