"""Stability and chronological walk-forward study of four frozen factor definitions."""
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "alternative_alpha"
OUT = Path(__file__).parent
FACTORS = ["oi_change_1d", "volume_change_5d", "atr14_pct", "momentum_20d"]
HORIZONS = [5, 10, 20]
YEARS = [2022, 2023, 2024, 2025, 2026]
ROLL_WINDOW = 252
ROLL_STEP = 21
HAC_LAGS = 20
Z95 = 1.959963984540054

def rank_vectors(x, y):
    z = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    n = len(z)
    if n < 4 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return None
    xr = z.iloc[:, 0].rank(method="average").to_numpy(float)
    yr = z.iloc[:, 1].rank(method="average").to_numpy(float)
    xr = (xr - xr.mean()) / xr.std(ddof=0)
    yr = (yr - yr.mean()) / yr.std(ddof=0)
    return z, xr, yr

def rank_hac(x, y, lags=HAC_LAGS):
    stuff = rank_vectors(x, y)
    if stuff is None:
        return {"n": int(pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna().shape[0]), "ic": np.nan, "se_hac": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "t_hac": np.nan, "p_hac_normal": np.nan, "n_eff_hac_approx": np.nan}
    z, xr, yr = stuff
    n = len(z)
    # The slope equals Spearman's rank correlation after rank-standardization.
    X = np.column_stack([np.ones(n), xr])
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ yr
    resid = yr - X @ beta
    g = X * resid[:, None]
    meat = g.T @ g
    L = min(int(lags), n - 1)
    for lag in range(1, L + 1):
        weight = 1.0 - lag / (L + 1.0)
        gamma = g[lag:].T @ g[:-lag]
        meat += weight * (gamma + gamma.T)
    cov = bread @ meat @ bread
    se = math.sqrt(max(0.0, float(cov[1, 1])))
    ic = float(beta[1])
    t = ic / se if se > 0 else np.nan
    p = math.erfc(abs(t) / math.sqrt(2.0)) if np.isfinite(t) else np.nan
    lo, hi = max(-1.0, ic - Z95 * se), min(1.0, ic + Z95 * se)
    iid_se2 = max(1e-15, (1.0 - ic * ic) / max(1, n - 2))
    neff = float(np.clip((1.0 - ic * ic) / max(se * se, 1e-15) + 2.0, 1.0, n)) if np.isfinite(se) else np.nan
    return {"n": n, "ic": ic, "se_hac": se, "ci95_low": lo, "ci95_high": hi, "t_hac": t, "p_hac_normal": p, "n_eff_hac_approx": neff, "iid_se": math.sqrt(iid_se2)}

def bh_adjust(values):
    p = np.asarray(values, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid): return q
    order = valid[np.argsort(p[valid])]
    raw = p[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    q[order] = np.minimum(adjusted, 1.0)
    return q

def add_stats(rows, keys, x, y, lags=HAC_LAGS):
    stat = rank_hac(x, y, lags)
    return {**keys, **stat}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel_path = SRC / "daily_factor_panel.csv"
    prior_ic_path = SRC / "factor_predictive_power.csv"
    regime_path = SRC / "market_state_factor_ic.csv"
    hashes_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [panel_path, prior_ic_path, regime_path]}
    d = pd.read_csv(panel_path, parse_dates=["date"]).sort_values("date").set_index("date")
    prior = pd.read_csv(prior_ic_path)
    d.index = pd.DatetimeIndex(d.index).normalize()
    if d.index.has_duplicates: raise ValueError("Duplicate daily panel dates")
    calendar_dates = pd.DatetimeIndex(d.index)
    # Map every signal date to the exact h-th subsequent observed DCE session.
    for h in HORIZONS:
        d[f"target_end_{h}d"] = pd.Series(calendar_dates, index=calendar_dates).shift(-h)
    d["year"] = d.index.year
    all_year_rows = []
    for year in YEARS:
        block = d[d.year == year]
        for factor in FACTORS:
            for h in HORIZONS:
                ycol = f"future_return_{h}d"
                endcol = f"target_end_{h}d"
                # Keep target entirely within the same calendar year; avoids annual test windows crossing into adjacent year.
                sample = block[(block[endcol].notna()) & (block[endcol].dt.year == year)]
                stat = rank_hac(sample[factor], sample[ycol])
                all_year_rows.append({"year": year, "factor": factor, "horizon_days": h, **stat})
    annual = pd.DataFrame(all_year_rows)
    annual["bh_q_value_60_annual_tests"] = bh_adjust(annual.p_hac_normal)
    annual["direction"] = np.where(annual.ic > 0, "positive", np.where(annual.ic < 0, "negative", "zero_or_na"))
    base_sign = {(r.factor, int(r.horizon_days)): np.sign(r.spearman_ic) for r in prior.itertuples() if r.factor in FACTORS}
    annual["full_sample_direction"] = ["positive" if base_sign.get((r.factor, int(r.horizon_days)), 0) > 0 else "negative" if base_sign.get((r.factor, int(r.horizon_days)), 0) < 0 else "zero_or_na" for r in annual.itertuples()]
    annual["same_direction_as_full_sample"] = annual.direction == annual.full_sample_direction
    consistency = annual.groupby(["factor", "horizon_days"], dropna=False).agg(years_with_same_direction_as_full_sample=("same_direction_as_full_sample", "sum"), years_with_positive_ic=("ic", lambda x: int((x > 0).sum())), years_with_negative_ic=("ic", lambda x: int((x < 0).sum())), annual_years_observed=("year", "nunique")).reset_index()
    annual.to_csv(OUT / "annual_ic.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    consistency.to_csv(OUT / "annual_direction_consistency.csv", index=False, encoding="utf-8-sig")

    # Rolling 252-session window, stepped by 21 sessions. Each window uses HAC inference;
    # rolling points are highly dependent and are plotted descriptively only.
    roll_rows = []
    for end_i in range(ROLL_WINDOW - 1, len(d), ROLL_STEP):
        win = d.iloc[end_i - ROLL_WINDOW + 1:end_i + 1]
        end_date = win.index[-1]
        for factor in FACTORS:
            for h in HORIZONS:
                stat = rank_hac(win[factor], win[f"future_return_{h}d"], HAC_LAGS)
                roll_rows.append({"window_end": end_date, "window_start": win.index[0], "window_sessions": ROLL_WINDOW, "step_sessions": ROLL_STEP, "factor": factor, "horizon_days": h, **stat})
    rolling = pd.DataFrame(roll_rows)
    rolling.to_csv(OUT / "rolling_ic.csv", index=False, encoding="utf-8-sig", float_format="%.10g", date_format="%Y-%m-%d")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    colors = {FACTORS[0]: "#4c78a8", FACTORS[1]: "#f58518", FACTORS[2]: "#54a24b", FACTORS[3]: "#e45756"}
    for ax, h in zip(axes, HORIZONS):
        sub = rolling[rolling.horizon_days == h]
        for factor in FACTORS:
            x = sub[sub.factor == factor]
            ax.plot(pd.to_datetime(x.window_end), x.ic, label=factor, color=colors[factor], linewidth=1.2)
        ax.axhline(0, color="#333", linewidth=.8)
        ax.set_ylabel(f"{h}-day IC")
        ax.set_title(f"{ROLL_WINDOW}-session rolling window; step {ROLL_STEP} sessions")
        ax.grid(alpha=.2)
    axes[0].legend(ncol=2, loc="upper left")
    axes[-1].set_xlabel("Window end (signal dates; targets are evaluated retrospectively)")
    fig.suptitle("Factor Stability — Rolling Spearman Rank IC (HAC bandwidth 20)")
    fig.tight_layout()
    fig.savefig(OUT / "rolling_ic.png", dpi=170)
    plt.close(fig)

    # Expanding walk-forward with fixed factor definitions; no fitting/selection.
    # Validation labels are purged at both boundaries so their forward target stays inside the fold.
    folds = [
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_partial", "2026-01-01", "2026-10-12"),
    ]
    wf_rows = []
    for fold, val_start_s, val_end_s in folds:
        val_start, val_end = pd.Timestamp(val_start_s), pd.Timestamp(val_end_s)
        for h in HORIZONS:
            endcol = f"target_end_{h}d"
            train = d[(d.index < val_start) & d[endcol].notna() & (d[endcol] < val_start)]
            valid = d[(d.index >= val_start) & (d.index <= val_end) & d[endcol].notna() & (d[endcol] <= val_end)]
            for factor in FACTORS:
                tr = rank_hac(train[factor], train[f"future_return_{h}d"], HAC_LAGS)
                va = rank_hac(valid[factor], valid[f"future_return_{h}d"], HAC_LAGS)
                wf_rows.append({"fold": fold, "train_start": train.index.min().date().isoformat() if len(train) else None, "train_end_signal": train.index.max().date().isoformat() if len(train) else None, "validation_start": val_start_s, "validation_end": val_end_s, "factor": factor, "horizon_days": h, "train_n": tr["n"], "train_ic": tr["ic"], "train_n_eff_hac_approx": tr["n_eff_hac_approx"], "validation_n": va["n"], "validation_ic": va["ic"], "validation_se_hac": va["se_hac"], "validation_ci95_low": va["ci95_low"], "validation_ci95_high": va["ci95_high"], "validation_n_eff_hac_approx": va["n_eff_hac_approx"], "validation_p_hac_normal": va["p_hac_normal"], "train_validation_same_sign": bool(np.sign(tr["ic"]) == np.sign(va["ic"])) if np.isfinite(tr["ic"]) and np.isfinite(va["ic"]) else None, "training_target_boundary_purged": True, "validation_target_confined_to_fold": True})
    wf = pd.DataFrame(wf_rows)
    wf["bh_q_value_48_validation_tests"] = bh_adjust(wf.validation_p_hac_normal)
    wf.to_csv(OUT / "walk_forward_results.csv", index=False, encoding="utf-8-sig", float_format="%.10g")

    # Fixed, previously-defined market states; conditional IC only descriptive.
    state_rows = []
    for dim, col, allowed in [("trend_state", "state", ["range", "transition", "uptrend", "downtrend"]), ("volatility_state", "vol_state", ["low", "high"])]:
        for state in allowed:
            sub = d[d[col].astype(str) == state]
            for factor in FACTORS:
                for h in HORIZONS:
                    stat = rank_hac(sub[factor], sub[f"future_return_{h}d"], HAC_LAGS)
                    state_rows.append({"regime_dimension": dim, "regime": state, "factor": factor, "horizon_days": h, **stat})
    state_result = pd.DataFrame(state_rows)
    state_result["bh_q_value_72_state_tests"] = bh_adjust(state_result.p_hac_normal)
    state_result.to_csv(OUT / "state_conditioned_ic.csv", index=False, encoding="utf-8-sig", float_format="%.10g")

    hash_after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [panel_path, prior_ic_path, regime_path]}
    if hashes_before != hash_after: raise RuntimeError("Existing factor research input changed")
    summary = {
        "research": "Factor Stability & Walk-Forward Analysis",
        "exploratory_only": True,
        "factors": FACTORS,
        "horizons": HORIZONS,
        "annual_years": YEARS,
        "annual_targets_confined_to_calendar_year": True,
        "annual_multiple_test_family": "4 factors x 3 horizons x 5 years = 60 tests; BH q-values",
        "walk_forward_folds": [{"fold": name, "validation_start": start, "validation_end": end} for name, start, end in folds],
        "walk_forward_validation_multiple_test_family": "4 factors x 3 horizons x 4 folds = 48 validation tests; BH q-values",
        "walk_forward_factor_definitions_frozen_no_fit_or_parameter_tuning": True,
        "purge_embargo": "Training rows whose forward target reaches validation start removed; validation rows whose target extends beyond fold end removed.",
        "rolling_window_sessions": ROLL_WINDOW,
        "rolling_step_sessions": ROLL_STEP,
        "hac_bandwidth_lags": HAC_LAGS,
        "ci": "Approximate 95% normal confidence interval based on Newey-West HAC rank-on-rank regression; n_eff is implied HAC effective-sample approximation.",
        "rolling_points_inference": "Not independent; rolling p-values not adjusted or used for decisions.",
        "existing_factor_research_inputs_unchanged": hashes_before == hash_after,
        "true_independent_out_of_sample": False,
        "original_strategy_or_backtests_modified": False,
        "forward_paper_system_started": False,
        "trading_executed": False,
        "input_hashes": hashes_before,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"annual_rows": len(annual), "rolling_points": rolling.window_end.nunique(), "walk_forward_rows": len(wf), "annual_min_q": float(annual.bh_q_value_60_annual_tests.min()), "walk_forward_min_q": float(wf.bh_q_value_48_validation_tests.min()), "inputs_unchanged": hashes_before == hash_after}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
