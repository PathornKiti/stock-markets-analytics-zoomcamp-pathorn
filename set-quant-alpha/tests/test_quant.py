"""Unit tests for the quant layer: run with  `uv run pytest set-quant-alpha/tests`."""

import numpy as np
import pandas as pd
import pytest
from setalpha import evaluate, portfolio, regime


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.01, 300)
    data = {f"S{i}": common * (0.5 + i / 10) + rng.normal(0, 0.015, 300) for i in range(8)}
    return pd.DataFrame(data, index=pd.bdate_range("2024-01-01", periods=300))


def test_mean_variance_respects_constraints(returns):
    cov = portfolio.ledoit_wolf_cov(returns)
    alpha = pd.Series(np.linspace(-0.01, 0.02, 8), index=returns.columns)
    w = portfolio.mean_variance(alpha, cov, max_weight=0.25)
    assert (w >= 0).all() and (w <= 0.25 + 1e-9).all()
    assert w.sum() <= 1 + 1e-6
    assert set(w.index) <= set(alpha.nlargest(5).index)  # money only goes to the best alphas


def test_no_positive_alpha_means_no_position(returns):
    cov = portfolio.ledoit_wolf_cov(returns)
    assert portfolio.mean_variance(pd.Series(-0.01, index=returns.columns), cov).empty


def test_vol_target_scales_down_never_up(returns):
    cov = portfolio.ledoit_wolf_cov(returns)
    w = pd.Series(1 / 8, index=returns.columns)
    vol = portfolio.portfolio_vol(w, cov)
    assert portfolio.portfolio_vol(
        portfolio.vol_target(w, cov, target=vol / 2), cov
    ) == pytest.approx(vol / 2)
    assert portfolio.vol_target(w, cov, target=vol * 2).sum() == pytest.approx(1.0)


def test_grinold_alpha_is_zero_mean_and_scaled():
    scores = pd.Series([0.1, 0.5, 0.9])
    a = portfolio.alpha_from_scores(scores, pd.Series([0.02, 0.02, 0.02]), ic=0.05, horizon=5)
    assert a.mean() == pytest.approx(0.0)
    assert a.iloc[2] == pytest.approx(0.05 * 0.02 * np.sqrt(5) * 1.0)


def test_hmm_filtered_probabilities_are_causal():
    rng = np.random.default_rng(1)
    mkt = pd.Series(
        np.r_[rng.normal(0.001, 0.005, 400), rng.normal(-0.002, 0.025, 200)],
        index=pd.bdate_range("2020-01-01", periods=600),
    )
    obs = regime.observations(mkt)
    model = regime.fit_hmm(obs)
    full = regime.filtered_probs(model, obs)
    partial = regime.filtered_probs(model, obs.iloc[:300])
    np.testing.assert_allclose(
        full.iloc[:300].to_numpy(), partial.to_numpy()
    )  # future data never changes the past
    np.testing.assert_allclose(full.sum(axis=1), 1.0)
    assert full["hmm_p_calm"].iloc[-50:].mean() < 0.1  # the volatile tail is not "calm"


def test_psr_and_dsr():
    rng = np.random.default_rng(2)
    good = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert evaluate.psr(good) > 0.95
    assert evaluate.dsr(good, [0.0, 0.02, 0.05, 0.1]) < evaluate.psr(
        good
    )  # deflation only lowers it


def test_factsheet_metrics():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2022-01-03", periods=500)
    bench = pd.Series(rng.normal(0.0004, 0.01, 500), index=idx)
    caps = evaluate.capture_ratios(bench, bench)
    assert caps["up capture %"] == pytest.approx(100) and caps["down capture %"] == pytest.approx(
        100
    )
    assert evaluate.omega(pd.Series([0.01, -0.01, 0.02, -0.02])) == pytest.approx(1.0)
    stats = evaluate.perf_stats(2 * bench, bench)
    assert stats["beta"] == pytest.approx(2.0) and stats["tracking error %"] > 0
    assert stats["Sharpe 5%"] < stats["Sharpe"] < stats["Sharpe 95%"]
    assert evaluate.max_dd_duration(pd.Series([0.1, -0.05, -0.05, 0.2])) == 2


def test_factor_regression_recovers_loadings():
    from setalpha import factors

    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=1500)
    f = pd.DataFrame(rng.normal(0, 0.01, (1500, 3)), index=idx, columns=["MKT", "SMB", "WML"])
    y = 0.0002 + 0.8 * f["MKT"] + 0.4 * f["SMB"] - 0.2 * f["WML"] + rng.normal(0, 0.002, 1500)
    out = factors.thai_local(y, f)
    assert out["b_MKT"] == pytest.approx(0.8, abs=0.02) and out["b_SMB"] == pytest.approx(
        0.4, abs=0.02
    )
    assert out["b_WML"] == pytest.approx(-0.2, abs=0.02) and out["alpha % p.a."] == pytest.approx(
        5.04, abs=2
    )


def test_monthly_usd_conversion():
    from setalpha import factors

    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    thb = pd.Series(0.0, index=idx)  # flat in THB...
    fx = pd.Series(
        np.where(idx.month == 1, 36.0, 40.0), index=idx
    )  # ...but the baht weakens 36 -> 40 per USD
    usd = factors.to_monthly_usd(thb, fx)
    assert usd.iloc[0] == pytest.approx(36 / 40 - 1)  # a USD investor loses 10% in February


# ----------------------------------------------------------------------------- research layer
def _panel(n_days=300, n_names=60, ic=0.3, seed=7):
    """Synthetic long panel: a score whose rank correlation with forward alpha is about `ic`."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    score = rng.normal(size=(n_days, n_names))
    alpha = ic * score + np.sqrt(1 - ic**2) * rng.normal(size=(n_days, n_names))
    f = pd.DataFrame(
        {
            "Date": np.repeat(dates, n_names),
            "Ticker": np.tile([f"T{i}" for i in range(n_names)], n_days),
            "pred_ens": score.ravel(),
            "fwd_alpha_5": 0.01 * alpha.ravel(),
        }
    )
    for c in ["log_dvol60", "ret_5", "ret_20", "mom_12_1", "idio_vol", "beta_mkt"]:
        f[c] = rng.normal(size=len(f))
    return f


def test_ic_quantiles_and_fama_macbeth():
    from setalpha import signal_eval

    f = _panel()
    ic = signal_eval.daily_ic(f)
    assert ic.mean() == pytest.approx(0.29, abs=0.03)
    s = signal_eval.ic_summary(ic)
    assert s["t (non-overlapping)"] < s["t (naive, overlapping)"]  # fewer obs -> smaller t
    q = signal_eval.quantile_alpha(f)
    assert (
        q.loc["D10", "mean alpha %"] > q.loc["D1", "mean alpha %"] and q.loc["D10-D1", "t (NW)"] > 3
    )
    assert signal_eval.monotonicity(q) > 0.9
    fm, neutral = signal_eval.fama_macbeth(f)
    assert (
        fm.loc["score", "t (NW)"] > 3 and abs(fm.loc["size", "t (NW)"]) < 3
    )  # noise controls do nothing
    assert neutral.mean() == pytest.approx(ic.mean(), abs=0.02)


def test_signal_persistence():
    from setalpha import signal_eval

    f = _panel(n_days=50)
    f["pred_ens"] = f.groupby("Ticker")["pred_ens"].transform("first")  # a constant ranking
    p = signal_eval.persistence(f, lags=(1, 5))
    assert p["rank autocorrelation"].min() == pytest.approx(1.0) and p[
        "top-decile turnover %"
    ].max() == pytest.approx(0.0)


def test_transfer_coefficient_and_fundamental_law():
    from setalpha import signal_eval

    f = _panel(n_days=3)
    f["idio_vol"] = 0.02
    day = f[f["Date"] == f["Date"].iloc[0]]
    w_opt = day.set_index("Ticker")["pred_ens"]
    w_opt = (w_opt - w_opt.mean()) / 100 + 1 / len(w_opt)  # active weights proportional to z
    w = pd.DataFrame([w_opt] * 3, index=f["Date"].unique())
    tc = signal_eval.transfer_coefficient(w, day)
    assert tc.iloc[0] == pytest.approx(1.0)
    law = signal_eval.fundamental_law(0.05, 100, 0.5)
    assert law["expected IR (IC x sqrt BR x TC)"] == pytest.approx(
        0.05 * np.sqrt(100 * 252 / 5) * 0.5
    )


def test_lo_sharpe_and_min_track_record():
    from setalpha import validity

    rng = np.random.default_rng(8)
    e = rng.normal(0.0005, 0.01, 3000)
    ar = pd.Series(np.r_[e[0], e[1:] + 0.5 * e[:-1]])  # positively autocorrelated returns
    iid = pd.Series(e)
    assert validity.lo_sharpe(ar) < ar.mean() / ar.std() * np.sqrt(
        252
    )  # autocorrelation inflates the IID Sharpe
    assert validity.lo_sharpe(iid) == pytest.approx(iid.mean() / iid.std() * np.sqrt(252), rel=0.15)
    assert validity.min_track_record(iid) < validity.min_track_record(
        iid * 0.2 - 0.0003
    )  # weaker edge needs more data
    assert validity.min_track_record(-iid.abs()) == np.inf


def test_multiple_testing_haircut():
    from setalpha import validity

    h = validity.haircut(3.0, years=10, others=[0.5, 0.9], n_trials=10)
    assert h["p bonferroni"] == pytest.approx(min(1, 10 * h["p"]))
    assert h["p holm"] == pytest.approx(h["p bonferroni"])  # the best test: Holm = Bonferroni
    assert 0 < h["haircut SR bonferroni"] < h["SR"]
    assert validity.haircut(3.0, n_trials=1)["haircut t bonferroni"] == pytest.approx(3.0)


def test_pbo_cscv():
    from setalpha import validity

    rng = np.random.default_rng(9)
    noise = pd.DataFrame(rng.normal(0, 0.01, (800, 12)))
    skill = noise.copy()
    skill[0] += 0.003  # one design with a real edge
    assert validity.pbo_cscv(skill, n_blocks=8)["PBO"] < 0.05
    assert (
        0.2 < validity.pbo_cscv(noise, n_blocks=8)["PBO"] < 0.8
    )  # pure noise: selection is a coin flip
    dup = noise.assign(copy=noise[0])
    assert validity.dedupe(dup).shape[1] == 12


def test_risk_and_reporting_tables():
    from setalpha import risk

    r = pd.Series([-0.05, -0.04] + [0.01] * 78, index=pd.bdate_range("2024-01-01", periods=80))
    assert risk.expected_shortfall(r, 0.975) == pytest.approx(
        4.5
    )  # mean of the worst 2.5% (2 of 80 days)
    rng = np.random.default_rng(10)
    idx = pd.bdate_range("2022-01-03", periods=600)
    fac = pd.DataFrame(rng.normal(0, 0.01, (600, 2)), index=idx, columns=["MKT", "SMB"])
    y = fac["MKT"] + rng.normal(0, 0.01, 600)
    dec = risk.risk_decomposition(y, fac)
    assert dec["factor total %"] + dec["specific %"] == pytest.approx(100, abs=1) and dec[
        "MKT %"
    ] == pytest.approx(50, abs=8)
    m = risk.monthly_table(y)
    assert m.loc[2022, "Year"] == pytest.approx(100 * ((1 + y.loc["2022"]).prod() - 1))
    roll = risk.rolling_alpha_ir(2 * fac["MKT"], fac["MKT"])
    assert roll["beta"].iloc[-1] == pytest.approx(2.0) and roll["alpha % p.a."].abs().max() < 1e-8
    curve = pd.DataFrame({"impact drag % p.a.": [0.1, 1.0, 10.0]}, index=[1.0, 10.0, 100.0])
    assert risk.capacity_at(curve, "impact drag % p.a.", 1.0) == pytest.approx(10.0)
    assert np.isnan(risk.capacity_at(curve, "impact drag % p.a.", 50.0))


def test_enhanced_index_keeps_tdex_loadings():
    """Tilts stay fully invested, inside the name cap and within the MKT/SMB/WML band around TDEX's (1, 0, 0)."""
    rng = np.random.default_rng(11)
    n = 40
    alpha = rng.normal(0.002, 0.002, n)
    s2 = np.full(n, 0.04**2)
    B = np.column_stack(
        [rng.normal(1.0, 0.3, n), rng.normal(0.25, 0.3, n), rng.normal(0.1, 0.2, n)]
    )  # small-cap tilt
    w, wt, ok = portfolio.enhanced_index(alpha, s2, B, np.zeros(n), 1.0)
    assert ok and w.sum() + wt == pytest.approx(1.0, abs=1e-6) and w.max() <= 0.05 + 1e-9
    expo = B.T @ w + np.array([1.0, 0.0, 0.0]) * wt - np.array([1.0, 0.0, 0.0])
    assert np.all(np.abs(expo) <= portfolio.LOADING_BAND + 1e-6)
    w_hi_cost, wt_hi, _ = portfolio.enhanced_index(
        alpha, s2, B, np.zeros(n), 1.0, fee=0.05
    )  # costs > every alpha
    assert w_hi_cost.sum() == pytest.approx(0.0, abs=1e-6) and wt_hi == pytest.approx(1.0, abs=1e-6)


def test_paper_ledger_fills_stops_and_marks():
    """Pending targets fill at the next open in 100-share lots; the stop sells at min(open, stop); NAV marks at the close."""
    from setalpha import paper

    days = pd.bdate_range("2026-11-02", periods=2)
    O = pd.DataFrame({"AAA": [10.0, 9.0], "BBB": [20.0, 20.0]}, index=days)  # noqa: E741
    L = pd.DataFrame({"AAA": [9.8, 8.5], "BBB": [19.5, 19.8]}, index=days)  # noqa: E741
    C = pd.DataFrame({"AAA": [10.0, 8.8], "BBB": [20.0, 20.2]}, index=days)
    st = paper.new_state()
    st["pending"] = {"w": {"AAA": 0.5, "BBB": 0.5}, "tdex": 0.0}
    trades = []
    paper.execute(st, days[0], O, C, 1.0, trades, "S")
    assert st["shares"] == {"AAA": 1_000_000, "BBB": 500_000} and st["pending"] is None
    assert st["cash"] == pytest.approx(paper.CAPITAL - 20e6 * (1 + paper.config.FEE_PER_SIDE))
    paper.stops(st, days[1], O, L, 0.08, trades, "S")  # AAA low 8.5 <= 10 * 0.92
    assert "AAA" not in st["shares"] and trades[-1]["price"] == pytest.approx(
        9.0
    )  # open 9.0 < stop 9.2
    nav = paper.mark(st, days[1], C, 1.0)
    assert nav["nav"] == pytest.approx(st["cash"] + 500_000 * 20.2) and nav["tdex_w"] == 0.0


def test_earnings_features_are_point_in_time():
    """The earnings reaction is known only from E+1; the expected-earnings flag uses last year's date + 364 days."""
    from setalpha import events

    dates = pd.bdate_range("2023-01-02", "2024-12-31")
    df = pd.DataFrame([(d, t) for d in dates for t in ["AAA", "BBB"]], columns=["Date", "Ticker"])
    df["beta_mkt"], df["liquid"] = 1.0, True
    px = df[["Date", "Ticker"]].copy()
    px["Close"] = 100.0
    px.loc[(px["Ticker"] == "AAA") & (px["Date"] >= "2023-05-16"), "Close"] = (
        110.0  # +10% on the day after E
    )
    earn = pd.DataFrame(
        {
            "Ticker": ["AAA", "AAA"],
            "Date": pd.to_datetime(["2023-05-15", "2024-05-14"]),
            "eps_est": [1.0, 1.0],
            "eps_rep": [1.2, 1.0],
            "surprise_pct": [20.0, 0.0],
        }
    )
    f = events.earnings_features(df, earn, px).set_index(["Date", "Ticker"])
    a = f.xs("AAA", level="Ticker")
    assert np.isnan(a.loc["2023-05-15", "last_ear"])  # day E: the E+1 return is not known yet
    assert a.loc["2023-05-16", "last_ear"] == pytest.approx(
        0.05
    )  # close of E+1: +10% vs EW market +5%
    assert a.loc["2023-05-17", "last_surprise"] == 20.0
    assert a.loc["2023-05-15", "days_since_earn"] == 0 and a.loc["2023-05-22", "post_earn_5"] == 1.0
    assert a.loc["2024-05-08", "exp_earn_next5"] == 1.0  # 2023-05-15 + 364 d = 2024-05-13
    assert (
        a.loc["2024-05-08", "diag_earn_in_hold"] == 1.0
        and a.loc["2024-04-01", "exp_earn_next5"] == 0.0
    )
    assert "BBB" not in f.index.get_level_values("Ticker")  # no earnings record -> not covered


def test_nvdr_features_lag_one_day():
    from setalpha import events

    dates = pd.bdate_range("2026-11-02", periods=6)
    df = pd.DataFrame({"Date": dates, "Ticker": "PTT.BK", "dvol60": 1e9})
    nvdr = pd.DataFrame({"Date": dates, "Ticker": "PTT.BK", "nvdr_buy": 2e8, "nvdr_sell": 1e8})
    f = events.nvdr_features(df, nvdr).set_index("Date")
    assert dates[0] not in f.index and f.loc[dates[1], "nvdr_int_1"] == pytest.approx(
        0.1
    )  # day-0 flow used on day 1
    assert f.loc[dates[3], "nvdr_int_5"] == pytest.approx(0.3)


def test_paper_score_tilt_matches_integrated_v1():
    """S6 tilt: best name 1.5x, worst 0.5x of 1/n, weights sum to len/n."""
    from setalpha import paper

    score = pd.Series({"A": 0.9, "B": 0.5, "C": 0.7})
    w = paper.tilt_weights(["B", "A", "C"], score, n=3)
    assert list(w.index) == ["A", "C", "B"] and w.sum() == pytest.approx(1.0)
    assert w["A"] / w["B"] == pytest.approx(3.0)  # 1.5 / 0.5


def test_signal_combination_utilities():
    """alpha_portfolio_v1: wide rank IC, Loewdin orthogonalisation, MV signal weights, clustering, ENB."""
    from setalpha import signal_eval

    rng = np.random.default_rng(11)
    y = rng.normal(size=(60, 50))
    x = y + rng.normal(size=(60, 50))
    x[0, :45] = np.nan  # too few names that day -> NaN
    ic = signal_eval.rank_ic_panel(x, y)
    assert np.isnan(ic[0]) and 0.5 < np.nanmean(ic) < 0.85
    a = pd.Series(x[1]).rank()
    assert ic[1] == pytest.approx(a.corr(pd.Series(y[1]).rank()))
    base = rng.normal(size=(200, 1))
    X = np.hstack(
        [
            base + 0.3 * rng.normal(size=(200, 1)),
            base + 0.3 * rng.normal(size=(200, 1)),
            rng.normal(size=(200, 1)),
        ]
    )
    orth = signal_eval.lowdin(X)
    assert np.allclose(orth.T @ orth / 200, np.eye(3), atol=1e-8)  # orthonormal
    assert np.corrcoef(orth[:, 0], X[:, 0])[0, 1] > 0.6  # and still close to the originals
    mu = pd.Series([0.05, 0.05, 0.03], index=list("abc"))
    cov = (
        pd.DataFrame([[1, 0.9, 0], [0.9, 1, 0], [0, 0, 1]], index=list("abc"), columns=list("abc"))
        * 1e-3
    )
    w = signal_eval.mv_signal_weights(mu, cov)
    assert (
        w.sum() == pytest.approx(1.0) and (w >= 0).all() and w["c"] > w["a"]
    )  # redundant pair shares its weight
    lab = signal_eval.cluster_signals(cov / 1e-3, cut=0.5)
    assert lab["a"] == lab["b"] != lab["c"]
    assert signal_eval.effective_bets(np.eye(4))["ENB entropy"] == pytest.approx(4.0)
    assert signal_eval.effective_bets(np.ones((4, 4)))["ENB participation"] == pytest.approx(1.0)
    eq = pd.Series(1 / 3, index=list("abc"))
    assert signal_eval.diversification_ratio(eq, cov) > 1.0


def test_cost_aware_mvo_trades_only_when_alpha_beats_cost():
    from setalpha import portfolio

    S = np.diag([0.004] * 4)
    w0 = np.array([0.25, 0.25, 0.25, 0.25])
    small = np.array([0.0010, 0.0, 0.0, 0.0])  # edge below the round-trip cost
    w, ok = portfolio.cost_aware_mvo(small, S, w0, fee=0.002, risk_aversion=1.0, max_weight=0.4)
    assert ok and np.allclose(w, w0, atol=1e-4)  # no trade
    big = np.array([0.05, 0.0, 0.0, -0.05])
    w, ok = portfolio.cost_aware_mvo(big, S, w0, fee=0.002, risk_aversion=1.0, max_weight=0.4)
    assert ok and w[0] > 0.3 and w[3] < 0.2 and w.sum() == pytest.approx(1.0, abs=1e-6)
    grp = np.array([0, 0, 1, 1])  # group cap binds
    w, ok = portfolio.cost_aware_mvo(
        np.array([0.05, 0.05, 0, 0]),
        S,
        np.zeros(4),
        fee=0.0,
        max_weight=0.6,
        groups=grp,
        group_cap=0.6,
    )
    assert ok and w[:2].sum() <= 0.6 + 1e-6 and w.sum() == pytest.approx(1.0, abs=1e-6)
    E = np.array([[1.0], [1.0], [-1.0], [-1.0]])  # exposure band binds
    w, ok = portfolio.cost_aware_mvo(
        np.array([0.05, 0.05, 0, 0]),
        S,
        np.zeros(4),
        fee=0.0,
        max_weight=0.6,
        exposures=E,
        targets=np.array([0.0]),
        band=0.2,
    )
    assert ok and abs((E[:, 0] * w).sum()) <= 0.2 + 1e-6


def test_stationary_bootstrap_sharpe_difference():
    from setalpha import validity

    rng = np.random.default_rng(12)
    idx = validity.stationary_bootstrap_index(1000, 20, rng)
    assert idx.min() >= 0 and idx.max() < 1000
    runs = np.diff(idx) == 1
    assert 0.9 < runs.mean() < 0.99  # mean block about 20 -> ~95% continuations
    common = rng.normal(0, 0.01, 2500)
    a = pd.Series(common + rng.normal(0.0008, 0.004, 2500))
    b = pd.Series(common + rng.normal(0.0, 0.004, 2500))
    t = validity.sharpe_diff_test(a, b, n_boot=500, seed=1)
    assert t["diff"] > 0 and t["p one-sided (H0 diff <= 0)"] < 0.05 and t["CI 2.5%"] > 0
    same = validity.sharpe_diff_test(b, b + rng.normal(0, 1e-4, 2500), n_boot=300, seed=2)
    assert same["p one-sided (H0 diff <= 0)"] > 0.05
