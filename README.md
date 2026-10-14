# Dalian Corn Futures Quant Research — Code-Only Portfolio

This repository presents the **research code and method**, not empirical result files. It studies systematic trend-following and candidate predictive relationships in Dalian Commodity Exchange (DCE) corn futures (contract family C), with emphasis on real-contract routing, roll accounting, time-series inference, and disciplined validation.

> **Publication status: READY TO PUBLISH AS CODE-ONLY.** The included project code was produced for this project collaboration and no third-party implementation code is vendored. This repository grants no explicit open-source reuse license; empirical deliverables remain excluded because upstream data redistribution rights are unconfirmed. See [CODE_ONLY_RELEASE_AUDIT.md](CODE_ONLY_RELEASE_AUDIT.md).

> **Validation status:** Historical strategy research and exploratory factor analysis were conducted in the private project workspace. Independent out-of-sample validation has **not** been completed. This public code-only bundle contains no historical performance claims, factor tables, figures, or result data.

## Research questions and methods

- Implement a 20/60-session moving-average trend-following baseline, then separately evaluate the previously specified ADX(14) entry condition and ATR(14) × 3 trailing-stop logic. The copied code preserves the project’s recorded parameters; this repository does not tune them.
- Build a causal additive continuous-price research index from actual contracts and model held-position P&L using the selected real contract, with explicit roll and cost treatment.
- Compute momentum, volume, open-interest, and volatility features from information available in the modeled signal prefix.
- Evaluate rank-based predictive associations with future returns using Newey–West/HAC methods for overlapping horizons, calendar-ordered folds, purging at target boundaries, and declared multiple-testing families.
- Preserve missing-date exclusions and surface survivorship, provider timestamp, and daily-bar execution limitations.

The methods describe the analysis code, not proof of profitability or an independently validated result. Rank IC measures association with a research return target and does not establish executable strategy P&L.

## Project structure

```text
github_public_release/
├── five_year_backtest/       # baseline route, strategy and prefix validation code
├── adx20_experiment/         # ADX comparison code
├── atr3_experiment/          # ATR stop and execution-assumption code
├── alternative_alpha/        # exploratory factor definitions and inference
├── factor_stability/         # annual and chronological Rank IC analysis
├── oi_deep_dive/             # open-interest factor analysis
├── oi_robustness/            # standardization, year and extreme-value checks
├── oi_market_regime/         # prior-known state-conditional analysis
├── scripts/                  # data-free package integrity checks
├── requirements-research.txt # pinned direct dependencies
├── requirements-research-macos-py313.lock.txt # tested macOS/Python 3.13 closure
├── THIRD_PARTY_NOTICES.md    # dependency license review
├── CODE_PROVENANCE.md        # authorship and licensing boundary
├── DATA_REQUIREMENTS.md      # lawful data inputs and source terms
└── REPRODUCING.md            # supported checks and research run limits
```

## Data requirements and privacy

No market data, API response, daily NAV/trade path, empirical summary, research chart, or account/feed snapshot is included. Full analyses require contract-level daily OHLCV and open interest, contract-route/index panels, gap/calendar metadata, and the stage-specific source files described in [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md). Users must obtain those inputs lawfully and may not redistribute them through this project without explicit permission. Some scripts call AKShare for an exchange calendar and require internet access.

A provider API wrapper or open-source software license does not grant rights to redistribute upstream market data. Historical contract-universe completeness is unresolved, survivorship bias remains possible, and source publication timestamps are not established.

## Third-party dependency licenses

See [Third-party dependency and license review](THIRD_PARTY_NOTICES.md).

## Environment and run steps

The pinned top-level environment is Python 3.13.1 with NumPy, pandas, Matplotlib, and AKShare. A macOS/Python 3.13 dependency-closure version snapshot is in `requirements-research-macos-py313.lock.txt`; it has no package hashes and is not a cross-platform lock. Direct and resolved dependency license metadata was reviewed for this environment; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research-macos-py313.lock.txt
python scripts/verify_release.py
```

The verifier checks only package inventory, hashes, common secret patterns, and Python syntax. It does not fetch market data, reproduce empirical results, run a backtest, or execute trades. To run an analysis, first obtain lawful data, place it in the project-relative locations expected by the selected script, back up and hash those inputs, and follow that script’s input/schema assumptions. The code is not a turnkey full-pipeline reproduction.

## Licensing and release status

The selected scripts are bespoke project-stage analysis code; this candidate contains no copied AKShare, NumPy, pandas, or Matplotlib source. However, the local repository has no source history or per-file authorship record, and no project license has been chosen. See [CODE_PROVENANCE.md](CODE_PROVENANCE.md). This code-only publication is for public inspection; no reuse license is granted. If an open-source reuse license is desired, choose and add one separately. Third-party data-derived deliverables remain excluded pending documented redistribution rights.

No independent out-of-sample validation has been completed. The historical 2021–2026 development sample was repeatedly examined, and the forward data system remains fail-closed. No trading is connected to or enabled by this code-only package.
