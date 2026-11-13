"""가짜 가격 데이터로 엔진과 전략이 의도대로 동작하는지 확인한다."""
import numpy as np
import pandas as pd
import pytest

from quant import strategies as st
from quant.backtest import run_backtest
from quant.metrics import compute_metrics, drawdown


def random_prices(n_days: int = 900, n_assets: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    drift = np.linspace(-0.0002, 0.0008, n_assets)
    returns = rng.normal(drift, 0.012, size=(n_days, n_assets))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=[f"A{i}" for i in range(n_assets)])


def steady_prices(daily_growth: list[float], n_days: int = 400) -> pd.DataFrame:
    """매일 일정한 비율로 오르거나 내리는 가격."""
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    days = np.arange(n_days)[:, None]
    return pd.DataFrame(100 * np.exp(days * np.array(daily_growth)), index=dates,
                        columns=[f"A{i}" for i in range(len(daily_growth))])


def noisy_prices(daily_sigma: list[float], n_days: int = 400, seed: int = 1) -> pd.DataFrame:
    """추세 없이 종목마다 다른 크기로 출렁이는 가격."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    returns = rng.normal(0, daily_sigma, size=(n_days, len(daily_sigma)))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=[f"A{i}" for i in range(len(daily_sigma))])


MONTHLY_STRATEGIES = [st.trend_following, st.cross_sectional_momentum, st.dual_momentum,
                      st.low_volatility, st.volatility_targeting]
ALL_STRATEGIES = MONTHLY_STRATEGIES + [st.mean_reversion, st.equal_weight_buy_and_hold,
                                       st.buy_and_hold]


# ---------- 엔진 ----------

def test_buy_and_hold_equals_price_change():
    prices = random_prices()[["A0"]]
    same_day = run_backtest(prices, st.buy_and_hold(prices), cost_bps=0, delay=0)
    next_day = run_backtest(prices, st.buy_and_hold(prices), cost_bps=0, delay=1)
    assert same_day.equity.iloc[-1] == pytest.approx(prices.iloc[-1, 0] / prices.iloc[0, 0])
    assert next_day.equity.iloc[-1] == pytest.approx(prices.iloc[-1, 0] / prices.iloc[1, 0])


def test_trading_cost_is_charged():
    prices = random_prices()[["A0"]]
    result = run_backtest(prices, st.buy_and_hold(prices), cost_bps=100, delay=0)
    assert result.equity.iloc[0] == pytest.approx(0.99)
    assert result.turnover.iloc[0] == pytest.approx(1.0)


def test_order_is_filled_only_after_the_delay():
    prices = random_prices(n_days=50)[["A0"]]
    targets = pd.DataFrame({"A0": [1.0]}, index=[prices.index[10]])
    result = run_backtest(prices, targets, cost_bps=0, delay=1)
    assert result.weights.iloc[:11].to_numpy().sum() == 0
    assert result.weights.iloc[11, 0] == pytest.approx(1.0)
    assert result.equity.iloc[11] == pytest.approx(1.0)
    assert result.equity.iloc[12] == pytest.approx(prices.iloc[12, 0] / prices.iloc[11, 0])


def test_short_or_leveraged_targets_are_rejected():
    prices = random_prices(n_days=50)
    for bad in (-0.1, 0.5):  # 음수 비중, 합계 300%
        targets = pd.DataFrame(bad, index=prices.index[:1], columns=prices.columns)
        with pytest.raises(ValueError):
            run_backtest(prices, targets)


# ---------- 전략 공통 ----------

@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_weights_are_long_only_and_unleveraged(strategy):
    weights = strategy(random_prices())
    assert len(weights) > 0
    assert weights.to_numpy().min() >= 0
    assert weights.sum(axis=1).max() <= 1 + 1e-9


@pytest.mark.parametrize("strategy", MONTHLY_STRATEGIES)
def test_monthly_strategies_do_not_use_future_data(strategy):
    prices = random_prices()
    full, cut = strategy(prices), strategy(prices.iloc[:600])
    common = cut.index[:-1]  # 잘린 데이터의 마지막 날은 진짜 월말이 아닐 수 있다
    assert len(common) > 5 and common.isin(full.index).all()
    pd.testing.assert_frame_equal(cut.loc[common], full.loc[common], check_freq=False)


def test_mean_reversion_does_not_use_future_data():
    prices = random_prices()
    full, cut = st.mean_reversion(prices), st.mean_reversion(prices.iloc[:600])
    assert len(cut) > 1
    pd.testing.assert_frame_equal(cut, full.loc[:prices.index[599]], check_freq=False)


# ---------- 전략별 ----------

def test_trend_following_holds_risers_and_avoids_fallers():
    weights = st.trend_following(steady_prices([-0.001, 0.001]))
    assert weights.iloc[-1].tolist() == [0.0, 0.5]


def test_momentum_picks_the_strongest_assets():
    weights = st.cross_sectional_momentum(steady_prices([0.0001, 0.0005, 0.001, 0.0003]), top_n=2)
    assert weights.iloc[-1].tolist() == [0.0, 0.5, 0.5, 0.0]


def test_dual_momentum_holds_winners_only_while_they_rise():
    rising = st.dual_momentum(steady_prices([0.0001, 0.0005, 0.001, -0.0003]), top_n=2)
    assert rising.iloc[-1].tolist() == [0.0, 0.5, 0.5, 0.0]
    # 가장 덜 떨어진 종목이 1등이어도, 떨어지고 있으면 사지 않는다.
    falling = st.dual_momentum(steady_prices([-0.001, -0.0005, -0.002]), top_n=1)
    assert falling.iloc[-1].tolist() == [0.0, 0.0, 0.0]


def test_low_volatility_picks_the_calmest_assets():
    weights = st.low_volatility(noisy_prices([0.005, 0.02, 0.01, 0.03]), top_n=2)
    assert weights.iloc[-1].tolist() == [0.5, 0.0, 0.5, 0.0]


def test_volatility_targeting_cuts_exposure_only_in_wild_markets():
    calm = st.volatility_targeting(noisy_prices([0.003]))   # 연 변동성 약 5%
    wild = st.volatility_targeting(noisy_prices([0.03]))    # 연 변동성 약 48%
    assert calm.iloc[-1, 0] == pytest.approx(1.0)
    assert 0.1 < wild.iloc[-1, 0] < 0.4


def test_rsi_extremes():
    always_up = st.rsi(steady_prices([0.001])).iloc[5:]
    always_down = st.rsi(steady_prices([-0.001])).iloc[5:]
    assert np.allclose(always_up, 100)
    assert np.allclose(always_down, 0)


def test_mean_reversion_buys_a_dip_in_an_uptrend_and_sells_the_rebound():
    growth = np.full(330, 0.001)
    growth[300:302] = -0.03  # 300일간 오른 뒤 이틀 연속 급락, 이후 다시 상승
    dates = pd.bdate_range("2015-01-01", periods=len(growth))
    prices = pd.DataFrame({"A0": 100 * np.exp(np.cumsum(growth))}, index=dates)
    weights = st.mean_reversion(prices)
    buys = weights[weights["A0"] > 0]
    assert weights.index[0] == dates[199] and weights.iloc[0, 0] == 0  # 지표가 처음 계산되는 날
    assert buys.index.tolist() == [dates[300]]
    assert buys.iloc[0, 0] == pytest.approx(1 / 3)
    assert weights.iloc[-1, 0] == 0
    assert weights.index[-1] > dates[301]


# ---------- 지표 ----------

def test_metrics_on_known_curve():
    dates = pd.to_datetime(["2020-01-01", "2020-05-01", "2020-09-01", "2021-01-01"])
    equity = pd.Series([1.0, 1.2, 0.9, 2.0], index=dates)
    metrics = compute_metrics(equity)
    assert metrics["cagr"] == pytest.approx(1.0, rel=0.01)
    assert metrics["max_drawdown"] == pytest.approx(-0.25)
    assert drawdown(equity).iloc[-1] == 0
