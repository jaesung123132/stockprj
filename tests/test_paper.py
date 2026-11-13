"""가상 매매가 과거 기록을 바꾸지 않고, 확정된 신호만 따르는지 확인한다."""
import numpy as np
import pandas as pd
import pytest

from quant import strategies as st
from quant.paper import confirmed_targets, list_trades, paper_targets, run_paper


def random_prices(n_days: int = 900, n_assets: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    drift = np.linspace(-0.0002, 0.0008, n_assets)
    returns = rng.normal(drift, 0.012, size=(n_days, n_assets))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=[f"A{i}" for i in range(n_assets)])


def test_signal_on_the_last_day_is_not_confirmed():
    prices = random_prices()
    targets = st.trend_following(prices)
    assert targets.index[-1] == prices.index[-1]
    confirmed = confirmed_targets(targets, prices.index[-1])
    assert len(confirmed) == len(targets) - 1
    assert confirmed.index[-1] < prices.index[-1]


def test_entry_uses_the_signal_in_force_on_the_start_day():
    prices = random_prices()
    targets = confirmed_targets(st.trend_following(prices), prices.index[-1])
    start = prices.index[500]
    previous = targets.loc[targets.index < start]
    plan = paper_targets(targets, start)
    assert plan.index[0] == start
    assert plan.iloc[0].tolist() == previous.iloc[-1].tolist()
    pd.testing.assert_frame_equal(plan.iloc[1:], targets.loc[targets.index > start],
                                  check_freq=False)


def test_account_starts_in_cash_and_buys_on_the_next_day():
    prices = random_prices()
    result, plan = run_paper(prices, st.trend_following(prices), str(prices.index[500].date()), 10)
    assert result.equity.index[0] == prices.index[500]
    assert result.equity.iloc[0] == 1.0
    assert result.weights.iloc[0].sum() == 0
    assert result.weights.iloc[1].sum() == pytest.approx(plan.iloc[0].sum(), abs=1e-3)


def test_new_days_never_change_the_recorded_past():
    prices = random_prices()
    start = str(prices.index[500].date())
    for cut in (540, 560, 585, 640):
        early = prices.iloc[:cut]
        short, _ = run_paper(early, st.trend_following(early), start, 10)
        full, _ = run_paper(prices, st.trend_following(prices), start, 10)
        pd.testing.assert_series_equal(short.equity, full.equity.iloc[:len(short.equity)],
                                       check_freq=False)


def test_nothing_happens_before_the_start_day():
    prices = random_prices()
    assert run_paper(prices, st.trend_following(prices), "2030-01-01", 10) is None


def test_start_on_the_latest_day_is_a_single_cash_row():
    prices = random_prices()
    result, _ = run_paper(prices, st.trend_following(prices), str(prices.index[-1].date()), 10)
    assert result.equity.tolist() == [1.0]


def test_trades_are_listed_with_before_and_after_weights():
    prices = random_prices()
    result, plan = run_paper(prices, st.trend_following(prices), str(prices.index[500].date()), 10)
    trades = list_trades(result)
    first_day, changes = trades[0]
    assert first_day == prices.index[501]
    assert {ticker for ticker, _, _ in changes} == set(plan.columns[plan.iloc[0] > 0])
    assert all(before == 0 and after > 0 for _, before, after in changes)
