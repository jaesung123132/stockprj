"""Markdown 보고서와 차트 생성.

차트 글자는 영어로 쓴다 (실행 환경에 한글 글꼴이 없어도 깨지지 않도록).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .metrics import compute_metrics, drawdown, yearly_returns

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def pct(value: float, digits: int = 1) -> str:
    return "-" if pd.isna(value) else f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    return "-" if pd.isna(value) else f"{value:.{digits}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join([":---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


METRIC_ROWS = [
    ("final_value", "1원이 몇 원이 되었나", lambda v: f"{v:.2f}원"),
    ("cagr", "연평균 수익률 (CAGR)", pct),
    ("volatility", "연 변동성", pct),
    ("sharpe", "샤프 비율", num),
    ("max_drawdown", "최대 낙폭 (MDD)", pct),
    ("calmar", "CAGR ÷ MDD", num),
    ("longest_drawdown_years", "전고점 회복까지 최장 기간", lambda v: f"{v:.1f}년"),
    ("worst_year", "최악의 해 수익률", pct),
    ("positive_years", "수익이 난 해의 비율", lambda v: pct(v, 0)),
    ("exposure", "평균 투자 비중", lambda v: pct(v, 0)),
    ("annual_turnover", "연 회전율 (매수+매도)", lambda v: f"{v:.1f}배"),
]


def describe_weights(row: pd.Series) -> str:
    """목표 비중 한 줄을 '종목 33%, ...' 형태의 글로 바꾼다."""
    held = row[row > 1e-9].sort_values(ascending=False)
    if held.empty:
        return "전액 현금"
    parts = [f"{ticker} {weight:.0%}" for ticker, weight in held.items()]
    cash = 1 - held.sum()
    if cash > 0.005:
        parts.append(f"현금 {cash:.0%}")
    return ", ".join(parts)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_equity(entries, results, path, title: str) -> None:
    """자산 곡선 (로그 축: 같은 높이 차이 = 같은 수익률 차이)."""
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=SURFACE)
    _style(ax)
    low, high = np.inf, 0.0
    for entry in sorted(entries, key=lambda e: -results[e.key][0].iloc[-1]):
        equity = results[entry.key][0]
        ax.plot(equity.index, equity.to_numpy(), color=entry.color,
                linestyle=entry.linestyle, linewidth=entry.linewidth,
                label=f"{entry.chart_label}  ({equity.iloc[-1]:.1f}x)")
        low, high = min(low, equity.min()), max(high, equity.max())
    ax.set_yscale("log")
    ax.set_yticks([2.0 ** k for k in range(-5, 12) if low / 1.3 <= 2.0 ** k <= high * 1.3])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_title(title, loc="left", color=INK, fontsize=13)
    ax.set_ylabel("Growth of 1 (log scale)", color=MUTED)
    legend = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_drawdowns(entries, results, path, title: str) -> None:
    """전략별 낙폭 (직전 최고점 대비 하락률)을 같은 눈금으로 나란히 그린다."""
    fig, axes = plt.subplots(len(entries), 1, figsize=(11, 1.5 * len(entries) + 0.9),
                             sharex=True, sharey=True, facecolor=SURFACE)
    for ax, entry in zip(np.atleast_1d(axes), entries):
        _style(ax)
        dd = drawdown(results[entry.key][0]) * 100
        ax.fill_between(dd.index, dd.to_numpy(), 0, color=entry.color, alpha=0.35, linewidth=0)
        ax.plot(dd.index, dd.to_numpy(), color=entry.color, linewidth=0.8)
        ax.set_title(f"{entry.chart_label}  (max drawdown {dd.min():.0f}%)",
                     loc="left", color=INK, fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _period_rows(entries, results) -> list[list[str]]:
    """10년 단위로 끊어 본 연평균 수익률과 최대 낙폭."""
    index = results[entries[0].key][0].index
    decade_of = np.asarray(index.year) // 10 * 10
    rows = []
    for decade in sorted(set(decade_of)):
        mask = decade_of == decade
        if mask.sum() < 252:
            continue
        dates = index[mask]
        row = [f"{dates[0].year}~{dates[-1].year}"]
        for entry in entries:
            metrics = compute_metrics(results[entry.key][0][mask])
            row.append(f"{pct(metrics['cagr'])} / {pct(metrics['max_drawdown'], 0)}")
        rows.append(row)
    return rows


def build_report(config: dict, entries, base, sensitivity, cost_bps: float,
                 latest_targets: list[tuple[str, str, str]], warnings: list[str]) -> str:
    """보고서 본문(Markdown)을 만든다.

    entries        : 전략들과 비교 기준들 (is_strategy로 구분)
    base           : {key: (자산 곡선, 지표)} - 기본 가정에서의 결과
    sensitivity    : {(delay, cost_bps): {key: (자산 곡선, 지표)}} - 가정을 바꿔 본 결과
    latest_targets : (전략 이름, 기준일, 목표 비중 설명)
    """
    index = base[entries[0].key][0].index
    years = (index[-1] - index[0]).days / 365.25
    names = [entry.name for entry in entries]
    strategies = [entry for entry in entries if entry.is_strategy]
    tickers = ", ".join(f"{ticker}({name})" for ticker, name in config["tickers"].items())

    lines = [
        f"# 백테스트 결과: {config['title']}",
        "",
        f"- 평가 기간: {index[0]:%Y-%m-%d} ~ {index[-1]:%Y-%m-%d} ({years:.1f}년)",
        f"- 대상 종목 {len(config['tickers'])}개: {tickers}",
        f"- 거래 비용: 매수·매도 각각 {cost_bps:g}bp ({cost_bps / 100:.2f}%)",
        "- 체결 가정: 종가로 신호 계산 → 다음 거래일 종가에 체결",
        "- 현금 이자 0%, 세금(양도소득세·배당소득세) 미반영, 공매도·레버리지 없음",
        "- 데이터: Yahoo Finance 수정 종가 (배당·분할 반영)",
    ]
    lines += [f"- **주의**: {note}" for note in config.get("notes", [])]

    lines += ["", "## 1. 요약", ""]
    rows = [[label] + [fmt(base[entry.key][1].get(key, np.nan)) for entry in entries]
            for key, label, fmt in METRIC_ROWS]
    lines += [md_table(["지표"] + names, rows), "",
              "![자산 곡선](equity.png)", "", "![낙폭](drawdown.png)"]

    lines += ["", "## 2. 연도별 수익률", ""]
    yearly = pd.DataFrame({entry.name: yearly_returns(base[entry.key][0]) for entry in entries})
    rows = [[str(year)] + [pct(value) for value in row] for year, row in yearly.iterrows()]
    lines += [md_table(["연도"] + names, rows), "",
              "첫 해와 마지막 해는 일부 기간만 포함합니다."]

    lines += ["", "## 3. 기간별 성과 (연평균 수익률 / 최대 낙폭)", "",
              md_table(["기간"] + names, _period_rows(entries, base))]

    lines += ["", "## 4. 가정을 바꿨을 때: 연평균 수익률 (샤프 비율)", ""]
    rows = []
    for (delay, cost), result in sensitivity.items():
        timing = "다음 날 종가" if delay == 1 else "당일 종가"
        rows.append([f"{timing}, 편도 {cost:g}bp"]
                    + [f"{pct(result[e.key][1]['cagr'])} ({num(result[e.key][1]['sharpe'])})"
                       for e in strategies])
    lines += [md_table(["체결 시점, 비용"] + [e.name for e in strategies], rows)]

    lines += ["", "## 5. 가장 최근 목표 비중 (참고)", ""]
    lines += [f"- **{name}** ({date} 기준): {text}" for name, date, text in latest_targets]

    lines += ["", "## 6. 데이터 점검", ""]
    lines += [f"- {warning}" for warning in warnings] or ["- 이상 징후 없음"]
    return "\n".join(lines) + "\n"
