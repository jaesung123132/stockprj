"""가격 데이터 내려받기(Yahoo Finance), CSV 캐시, 기본 정제."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

# 하루 등락률이 이 값을 넘으면 데이터 오류(잘못된 액면분할 보정 등)를 의심한다.
SUSPICIOUS_DAILY_MOVE = 0.35
# 가격이 며칠 연속 비어 있을 때: SHORT_GAP 이하는 휴장일 차이로 보고 넘어가고,
# MAX_GAP을 넘으면 결과를 믿을 수 없으므로 실행을 중단한다.
SHORT_GAP = 5
MAX_GAP = 20


def download_prices(tickers: list[str], start: str, end: str | None = None,
                    retries: int = 3) -> pd.DataFrame:
    """배당·분할이 반영된 수정 종가를 내려받는다 (행: 날짜, 열: 종목)."""
    import yfinance as yf

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                              progress=False, group_by="column", threads=False)
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            close = close.dropna(how="all")
            missing = [t for t in tickers
                       if t not in close.columns or close[t].dropna().empty]
            if missing:
                raise RuntimeError(f"데이터를 받지 못한 종목: {missing}")
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close.index.name = "date"
            return close[tickers]
        except Exception as error:  # 네트워크 오류나 일시적 차단은 재시도한다
            last_error = error
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"가격 데이터 다운로드 실패: {last_error}")


def load_prices(tickers: list[str], start: str, cache_path: Path,
                refresh: bool = False) -> pd.DataFrame:
    """캐시 CSV가 있으면 그것을 쓰고, 없거나 refresh면 새로 내려받아 저장한다."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if all(t in prices.columns for t in tickers):
            return prices[tickers]
    prices = download_prices(tickers, start)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.round(6).to_csv(cache_path)
    return prices


def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """모든 종목의 데이터가 있는 첫 날부터 잘라내고, 점검 결과를 글로 돌려준다.

    가격이 MAX_GAP 거래일 넘게 연속으로 비어 있는 종목이 있으면 ValueError를 낸다.
    """
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    first_dates = {ticker: prices[ticker].first_valid_index() for ticker in prices.columns}
    latest = max(first_dates, key=first_dates.get)
    prices = prices.loc[first_dates[latest]:]
    notes = [f"공통 시작일 {first_dates[latest]:%Y-%m-%d} (가장 늦게 시작하는 종목: {latest})"]

    for ticker in prices.columns:
        missing = prices[ticker].isna()
        if not missing.any():
            continue
        groups = (~missing).cumsum()
        runs = missing.groupby(groups).sum()
        longest = int(runs.max())
        gap_start = missing.index[(groups == runs.idxmax()) & missing][0]
        if longest > MAX_GAP:
            raise ValueError(f"{ticker}: {gap_start:%Y-%m-%d}부터 {longest}거래일 연속으로 "
                             "가격이 비어 있습니다. 다른 종목이나 데이터로 바꾸세요.")
        if longest > SHORT_GAP:
            notes.append(f"{ticker}: {gap_start:%Y-%m-%d}부터 {longest}거래일 연속으로 "
                         "가격이 비어 있어 직전 값으로 채움")
    prices = prices.ffill()

    if (prices <= 0).any().any():
        bad = prices.columns[(prices <= 0).any()].tolist()
        notes.append(f"0 이하의 가격이 있는 종목: {bad}")
    moves = prices.pct_change().abs()
    for ticker in prices.columns:
        hits = moves.index[moves[ticker] > SUSPICIOUS_DAILY_MOVE]
        if len(hits):
            days = ", ".join(d.strftime("%Y-%m-%d") for d in hits[:5])
            more = " 등" if len(hits) > 5 else ""
            notes.append(f"{ticker}: 하루 {SUSPICIOUS_DAILY_MOVE:.0%} 초과 변동 "
                         f"{len(hits)}회 ({days}{more})")
    return prices, notes
