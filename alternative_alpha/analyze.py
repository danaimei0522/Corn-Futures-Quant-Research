"""Exploratory time-series factor IC study; reads but never modifies frozen inputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent
BT = ROOT / "five_year_backtest"
HIST = ROOT / "long_history_audit"
REGIMES = ROOT / "five_year_trade_research" / "daily_asof_features.csv"
START, END = "2021-10-12", "2026-10-12"
HORIZONS = (5, 10, 20)
FACTORS = [
    "momentum_5d", "momentum_10d", "momentum_20d",
    "volume_change_1d", "volume_change_5d", "relative_volume_20d",
    "oi_change_1d", "price_oi_confirmation_1d",
    "atr14", "atr14_pct", "historical_volatility_20d", "volatility_change_5d",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_corr(x, y):
    z = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    if len(z) < 3 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return float("nan"), len(z)
    return float(z.iloc[:, 0].rank(method="average").corr(z.iloc[:, 1].rank(method="average"))), len(z)


def nw_pvalue(x, y, lags):
    """Normal approximation to a Newey-West t test for rank-on-rank slope."""
    z = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    n = len(z)
    if n < 5 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return float("nan"), float("nan"), n
    xr = z.iloc[:, 0].rank(method="average").to_numpy(dtype=float)
    yr = z.iloc[:, 1].rank(method="average").to_numpy(dtype=float)
    xr = (xr - xr.mean()) / xr.std(ddof=0)
    yr = (yr - yr.mean()) / yr.std(ddof=0)
    X = np.column_stack([np.ones(n), xr])
    inv = np.linalg.pinv(X.T @ X)
    beta = inv @ X.T @ yr
    resid = yr - X @ beta
    g = X * resid[:, None]
    meat = g.T @ g
    L = min(int(lags), n - 1)
    for lag in range(1, L + 1):
        weight = 1 - lag / (L + 1)
        gamma = g[lag:].T @ g[:-lag]
        meat += weight * (gamma + gamma.T)
    cov = inv @ meat @ inv
    se = math.sqrt(max(0.0, float(cov[1, 1])))
    if se == 0 or not np.isfinite(se):
        return float(beta[1]), float("nan"), n
    t = float(beta[1] / se)
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, p, n


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    raw = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def wilder_atr(tr, period=14):
    values = np.asarray(tr, dtype=float)
    out = np.full(len(values), np.nan)
    seed = []
    last = np.nan
    for i, value in enumerate(values):
        if not np.isfinite(value):
            seed, last = [], np.nan
            continue
        if not np.isfinite(last):
            seed.append(value)
            if len(seed) == period:
                last = float(np.mean(seed))
                out[i] = last
        else:
            last = (last * (period - 1) + value) / period
            out[i] = last
    return pd.Series(out, index=tr.index)


def selected_contract_features(s, raw):
    timeline = s.index
    records = []
    symbols = sorted(set(s.contract.astype(str)))
    required_raw = ("open", "high", "low", "close", "volume", "hold")
    for symbol in symbols:
        if symbol not in raw:
            continue
        f = raw[symbol].reindex(timeline)
        valid_hlc = f[["high", "low", "close"]].notna().all(axis=1)
        prev_close = f.close.shift(1)
        tr = pd.concat([f.high - f.low, (f.high - prev_close).abs(), (f.low - prev_close).abs()], axis=1).max(axis=1)
        tr = tr.where(valid_hlc & prev_close.notna())
        atr = wilder_atr(tr, 14)
        vol, oi = f.volume, f.hold
        row = pd.DataFrame(index=timeline)
        row["volume_change_1d"] = vol.div(vol.shift(1)).sub(1).where((vol > 0) & (vol.shift(1) > 0))
        row["volume_change_5d"] = vol.div(vol.shift(5)).sub(1).where((vol > 0) & (vol.shift(5) > 0))
        prior_volume_mean = vol.rolling(20, min_periods=20).mean().shift(1)
        row["relative_volume_20d"] = vol.div(prior_volume_mean).where(prior_volume_mean > 0)
        row["oi_change_1d"] = oi.div(oi.shift(1)).sub(1).where((oi > 0) & (oi.shift(1) > 0))
        row["atr14"] = atr
        row["atr14_pct"] = atr.div(f.close).where(f.close > 0)
        row["volatility_change_5d"] = atr.div(atr.shift(5)).sub(1).where((atr > 0) & (atr.shift(5) > 0))
        chosen = s.contract.astype(str).eq(symbol)
        for dt in timeline[chosen]:
            vals = row.loc[dt].to_dict()
            for col in required_raw:
                vals[f"raw_{col}"] = f.at[dt, col] if col in f else np.nan
            records.append({"date": dt, "contract": symbol, **vals})
    result = pd.DataFrame(records).set_index("date").sort_index().reindex(timeline)
    if result.index.has_duplicates:
        raise ValueError("More than one selected actual contract for a routed date")
    return result


def derive_factors(s, selected):
    daily_index_return = s.index_close.pct_change(fill_method=None)
    feats = selected[["volume_change_1d", "volume_change_5d", "relative_volume_20d", "oi_change_1d", "atr14", "atr14_pct", "volatility_change_5d"]].copy()
    feats["momentum_5d"] = s.index_close.div(s.index_close.shift(5)).sub(1)
    feats["momentum_10d"] = s.index_close.div(s.index_close.shift(10)).sub(1)
    feats["momentum_20d"] = s.index_close.div(s.index_close.shift(20)).sub(1)
    feats["historical_volatility_20d"] = daily_index_return.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    roll = s.roll.fillna(False).astype(bool)
    feats["price_oi_confirmation_1d"] = (np.sign(daily_index_return) * feats.oi_change_1d).where(~roll)
    feats["contract"] = selected.contract
    feats["index_close"] = s.index_close
    feats["source_roll"] = s.roll.fillna(False).astype(bool)
    return feats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    signal_path = BT / "causal_signal_and_contracts.csv"
    missing_path = BT / "missing_dates.csv"
    regime_path = REGIMES
    contract_paths = sorted(HIST.glob("C????_daily.csv"))
    used_paths = [signal_path, missing_path, regime_path, *contract_paths]
    before_hashes = {str(p): sha256(p) for p in used_paths}

    s = pd.read_csv(signal_path, parse_dates=["date"]).sort_values("date").set_index("date")
    s.index = pd.DatetimeIndex(s.index).normalize()
    eval_start, eval_end = pd.Timestamp(START), pd.Timestamp(END)
    timeline = s.index
    if s.index.has_duplicates or s.index.isna().any():
        raise ValueError("Duplicate or invalid signal-index dates")
    if s.index_close.isna().any() or (s.index_close <= 0).any():
        raise ValueError("Missing/nonpositive causal signal index; refusing to fill")

    raw = {}
    for path in contract_paths:
        symbol = path.name[:5]
        frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        frame = frame.loc[~frame.index.duplicated(keep=False)]
        frame = frame.reindex(timeline)
        raw[symbol] = frame

    selected = selected_contract_features(s, raw)
    required_raw = ("open", "high", "low", "close", "volume", "hold")
    missing_required = selected[[f"raw_{f}" for f in required_raw]].isna().any(axis=1)
    missing_selected = int(missing_required.loc[eval_start:eval_end].sum())
    # Price returns use the previously audited causal additive real-contract index.
    feats = derive_factors(s, selected)
    regimes = pd.read_csv(regime_path, parse_dates=["asof"])
    regimes["asof"] = pd.to_datetime(regimes["asof"]).dt.normalize()
    state_by_asof = regimes.drop_duplicates("asof").set_index("asof")[["state", "vol_state"]]
    feats = feats.join(state_by_asof, how="left")
    feats["contract"] = selected.contract

    prefix_checks = []
    for cutoff in ("2022-12-30", "2024-12-31", "2025-12-31"):
        cut = pd.Timestamp(cutoff)
        prefix_s = s.loc[:cut]
        prefix_raw = {k: v.loc[:cut] for k, v in raw.items()}
        prefix_panel = derive_factors(prefix_s, selected_contract_features(prefix_s, prefix_raw))
        compare_cols = [*FACTORS, "contract", "index_close", "source_roll"]
        expected_panel = feats.loc[:cut, compare_cols]
        got_panel = prefix_panel.loc[:, compare_cols]
        if not expected_panel.equals(got_panel):
            numeric = [c for c in FACTORS + ["index_close"] if c in expected_panel]
            for col in numeric:
                if not np.allclose(expected_panel[col].to_numpy(float), got_panel[col].to_numpy(float), equal_nan=True, rtol=0, atol=1e-12):
                    raise AssertionError(f"Prefix invariance failed for {col} at {cutoff}")
            for col in ["contract", "source_roll"]:
                if not expected_panel[col].equals(got_panel[col]):
                    raise AssertionError(f"Prefix invariance failed for {col} at {cutoff}")
        prefix_checks.append({"cutoff": cutoff, "rows": len(prefix_panel), "factor_values_prefix_invariant": True})

    for h in HORIZONS:
        feats[f"future_return_{h}d"] = s.index_close.shift(-h).div(s.index_close).sub(1)
    evaluation = feats.loc[eval_start:eval_end].copy()
    daily_index_return = s.index_close.pct_change(fill_method=None)
    for f in FACTORS:
        evaluation[f] = evaluation[f].replace([np.inf, -np.inf], np.nan)

    missing_audit = pd.read_csv(missing_path)
    if "excluded_by_delivery_month" in missing_audit:
        all_gaps = len(missing_audit)
        delivery_excluded = int(missing_audit.excluded_by_delivery_month.astype(bool).sum())
    else:
        all_gaps, delivery_excluded = len(missing_audit), 0

    ic_rows = []
    overlap_rows = []
    for factor in FACTORS:
        for h in HORIZONS:
            y = evaluation[f"future_return_{h}d"]
            x = evaluation[factor]
            ic, n = rank_corr(x, y)
            lags = max(20, h - 1)
            t, p, _ = nw_pvalue(x, y, lags)
            pair = pd.concat([x, y], axis=1).dropna()
            phase_ics = []
            for phase in range(h):
                part = pair.iloc[phase::h]
                phase_ic, phase_n = rank_corr(part.iloc[:, 0], part.iloc[:, 1])
                if np.isfinite(phase_ic):
                    phase_ics.append({"phase": phase, "n": phase_n, "spearman_ic": phase_ic})
            overlap_rows.append({"factor": factor, "horizon_days": h, "nonoverlap_phase_count": len(phase_ics), "nonoverlap_weighted_mean_ic": float(np.average([x["spearman_ic"] for x in phase_ics], weights=[x["n"] for x in phase_ics])) if phase_ics else np.nan, "nonoverlap_min_ic": min([x["spearman_ic"] for x in phase_ics], default=np.nan), "nonoverlap_max_ic": max([x["spearman_ic"] for x in phase_ics], default=np.nan), "phases": json.dumps(phase_ics, ensure_ascii=False)})
            ic_rows.append({"factor": factor, "horizon_days": h, "n": n, "spearman_ic": ic, "newey_west_lags": lags, "hac_t_stat_normal_approx": t, "hac_p_value_normal_approx": p, "factor_nonmissing": int(x.notna().sum()), "future_return_nonmissing": int(y.notna().sum()), "mean_future_return": float(y.dropna().mean()), "median_future_return": float(y.dropna().median())})
    ic = pd.DataFrame(ic_rows)
    ic["bh_q_value_36_tests"] = bh_adjust(ic.hac_p_value_normal_approx)
    overlap = pd.DataFrame(overlap_rows)

    corr = evaluation[FACTORS].corr(method="spearman", min_periods=30)
    corr_n = evaluation[FACTORS].notna().astype(int).T.dot(evaluation[FACTORS].notna().astype(int))
    corr.to_csv(OUT / "factor_correlation_matrix.csv", encoding="utf-8-sig", float_format="%.8f")
    corr_n.to_csv(OUT / "factor_correlation_sample_sizes.csv", encoding="utf-8-sig")
    ic.to_csv(OUT / "factor_predictive_power.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    overlap.to_csv(OUT / "overlap_robustness.csv", index=False, encoding="utf-8-sig", float_format="%.10g")

    state_rows = []
    for dimension, col in (("trend_state", "state"), ("volatility_state", "vol_state")):
        for state in sorted(evaluation[col].dropna().astype(str).unique()):
            group = evaluation[evaluation[col].astype(str) == state]
            for factor in FACTORS:
                for h in HORIZONS:
                    y = group[f"future_return_{h}d"]
                    icv, n = rank_corr(group[factor], y)
                    state_rows.append({"regime_dimension": dimension, "regime": state, "factor": factor, "horizon_days": h, "n": n, "spearman_ic": icv, "mean_future_return": float(y.dropna().mean()) if y.notna().any() else np.nan, "median_future_return": float(y.dropna().median()) if y.notna().any() else np.nan})
    states = pd.DataFrame(state_rows)
    states.to_csv(OUT / "market_state_factor_ic.csv", index=False, encoding="utf-8-sig", float_format="%.10g")

    # Descriptive contemporaneous price/OI relationship, separate from predictive IC tests.
    relation_rows = []
    p1 = daily_index_return.loc[evaluation.index].where(~evaluation.source_roll)
    oi1 = evaluation.oi_change_1d
    for dimension, col in [("all", None), ("trend_state", "state"), ("volatility_state", "vol_state")]:
        groups = [("all", evaluation.index)] if col is None else [(str(v), evaluation.index[evaluation[col].astype(str).eq(str(v))]) for v in sorted(evaluation[col].dropna().astype(str).unique())]
        for label, idx in groups:
            rval, rn = rank_corr(p1.loc[idx], oi1.loc[idx])
            relation_rows.append({"regime_dimension": dimension, "regime": label, "n": rn, "same_day_price_return_oi_change_spearman": rval})
    pd.DataFrame(relation_rows).to_csv(OUT / "price_oi_relationship.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    quad = evaluation[["oi_change_1d", "future_return_5d", "future_return_10d", "future_return_20d"]].copy()
    quad["same_day_price_return"] = p1
    quad["price_oi_quadrant"] = "flat_or_unavailable"
    valid_quad = quad.same_day_price_return.notna() & quad.oi_change_1d.notna()
    conditions = [
        valid_quad & (quad.same_day_price_return > 0) & (quad.oi_change_1d > 0),
        valid_quad & (quad.same_day_price_return < 0) & (quad.oi_change_1d > 0),
        valid_quad & (quad.same_day_price_return > 0) & (quad.oi_change_1d < 0),
        valid_quad & (quad.same_day_price_return < 0) & (quad.oi_change_1d < 0),
    ]
    quad["price_oi_quadrant"] = np.select(conditions, ["price_up_oi_up", "price_down_oi_up", "price_up_oi_down", "price_down_oi_down"], default="flat_or_unavailable")
    quadrant_rows = []
    for label, group in quad[quad.price_oi_quadrant != "flat_or_unavailable"].groupby("price_oi_quadrant"):
        row = {"price_oi_quadrant": label, "n": len(group)}
        for h in HORIZONS:
            y = group[f"future_return_{h}d"].dropna()
            row[f"mean_future_return_{h}d"] = float(y.mean()) if len(y) else np.nan
            row[f"median_future_return_{h}d"] = float(y.median()) if len(y) else np.nan
            row[f"n_future_return_{h}d"] = len(y)
        quadrant_rows.append(row)
    pd.DataFrame(quadrant_rows).to_csv(OUT / "price_oi_quadrants.csv", index=False, encoding="utf-8-sig", float_format="%.10g")

    # Descriptive yearly stability; no year is used for factor selection.
    year_rows = []
    for year, group in evaluation.groupby(evaluation.index.year):
        for factor in FACTORS:
            for h in HORIZONS:
                icv, n = rank_corr(group[factor], group[f"future_return_{h}d"])
                year_rows.append({"year": int(year), "factor": factor, "horizon_days": h, "n": n, "spearman_ic": icv})
    yearly = pd.DataFrame(year_rows)
    yearly.to_csv(OUT / "yearly_factor_ic.csv", index=False, encoding="utf-8-sig", float_format="%.10g")

    out_features = evaluation.reset_index().rename(columns={"index": "date"})
    out_features.to_csv(OUT / "daily_factor_panel.csv", index=False, encoding="utf-8-sig", float_format="%.10g", date_format="%Y-%m-%d")

    # Figures
    piv = ic.pivot(index="factor", columns="horizon_days", values="spearman_ic").reindex(FACTORS)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    piv.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.axhline(0, color="#555", lw=.8); ax.set_ylabel("Spearman IC"); ax.set_xlabel(""); ax.set_title("Exploratory time-series Rank IC by forward horizon")
    ax.legend(title="Forward sessions"); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(OUT / "factor_ic_by_horizon.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(FACTORS)), FACTORS, rotation=70, ha="right", fontsize=8); ax.set_yticks(range(len(FACTORS)), FACTORS, fontsize=8)
    ax.set_title("Factor Spearman Correlation (pairwise complete observations)"); fig.colorbar(im, ax=ax, label="Spearman correlation"); fig.tight_layout(); fig.savefig(OUT / "factor_correlation_heatmap.png", dpi=170); plt.close(fig)
    stateplot = states[(states.regime_dimension == "trend_state") & states.regime.isin(["range", "transition", "uptrend", "downtrend"])]
    statepivot = stateplot[stateplot.horizon_days == 10].pivot(index="factor", columns="regime", values="spearman_ic").reindex(FACTORS)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(statepivot.to_numpy(), vmin=-.3, vmax=.3, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(statepivot.shape[1]), statepivot.columns, rotation=25); ax.set_yticks(range(len(FACTORS)), FACTORS, fontsize=8)
    ax.set_title("10-session Rank IC by pre-defined market state"); fig.colorbar(im, ax=ax, label="Spearman IC"); fig.tight_layout(); fig.savefig(OUT / "factor_ic_by_market_state.png", dpi=170); plt.close(fig)

    factors_missing = {f: int(evaluation[f].isna().sum()) for f in FACTORS}
    n_eval = len(evaluation)
    state_counts = evaluation.state.value_counts(dropna=False).to_dict()
    vol_state_counts = evaluation.vol_state.value_counts(dropna=False).to_dict()
    end_hashes = {str(p): sha256(p) for p in used_paths}
    if before_hashes != end_hashes:
        raise RuntimeError("Protected input hash changed during analysis")
    lock_check = {"status": "PASS" if before_hashes == end_hashes else "FAIL", "input_file_count": len(before_hashes), "hashes": before_hashes}
    (OUT / "input_hashes.json").write_text(json.dumps(lock_check, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "study": "Alternative Alpha Factor Research — exploratory only",
        "evaluation_start": START, "evaluation_end": END,
        "timeline_rows_including_warmup": len(s), "evaluation_rows": n_eval,
        "evaluation_observed_sessions": int(evaluation.index.nunique()),
        "evaluation_routing_roll_days": int(evaluation.source_roll.sum()),
        "roll_days_excluded_from_same_day_price_oi_relation": int(evaluation.source_roll.sum()),
        "selected_contracts": int(evaluation.contract.nunique()),
        "selected_contract_rows_missing_any_required_ohlcv_oi": missing_selected,
        "raw_contract_missing_date_rows_total": all_gaps,
        "raw_missing_rows_marked_excluded_by_delivery_month": delivery_excluded,
        "no_imputation": True,
        "factor_missing_counts": factors_missing,
        "future_target_nonmissing_counts": {str(h): int(evaluation[f"future_return_{h}d"].notna().sum()) for h in HORIZONS},
        "trend_regime_counts": {str(k): int(v) for k, v in state_counts.items()},
        "volatility_regime_counts": {str(k): int(v) for k, v in vol_state_counts.items()},
        "forward_return_overlaps": {"method": "Newey-West HAC normal-approximate inference with Bartlett bandwidth max(horizon-1,20); also report h-phase nonoverlapping IC sensitivity", "horizons": list(HORIZONS)},
        "lookahead_prefix_checks": prefix_checks,
        "multiple_testing": "Benjamini-Hochberg q-values over the 36 factor-by-horizon HAC p-values; exploratory adjustment only",
        "protected_inputs_unchanged": before_hashes == end_hashes,
        "strategy_v1_0_0_modified": False,
        "historical_backtest_modified": False,
        "forward_system_modified_or_started": False,
        "factor_selection_uses_original_trade_pnl": False,
        "simulation_or_trade_actions": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": n_eval, "selected_contracts": int(evaluation.contract.nunique()), "missing_selected_rows": missing_selected, "best_by_abs_ic_10d": ic.loc[ic.horizon_days == 10].assign(abs_ic=lambda d: d.spearman_ic.abs()).sort_values("abs_ic", ascending=False).head(5)[["factor", "n", "spearman_ic", "hac_p_value_normal_approx", "bh_q_value_36_tests"]].to_dict("records"), "protected_inputs_unchanged": before_hashes == end_hashes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
