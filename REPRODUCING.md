# Reproduction and environment

## Data-free package checks

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research-macos-py313.lock.txt
python scripts/verify_release.py
```

The lock snapshot records the resolved dependency versions from the tested macOS/Python 3.13.1 environment; it does not include wheel hashes and is not portable across all operating systems. The standard-library verifier checks release hashes, package contents, likely secrets, and Python syntax. It performs no data fetch, signal calculation, backtest, or empirical result reproduction.

## Running a research stage

The included analysis scripts expect stage-specific contract OHLCV/open-interest data, contract mapping/routing panels, missing-date masks, and (for some scripts) a trading calendar. These inputs are not bundled. The project has no single end-to-end runner. Inspect the input paths and expected columns near the top of each selected script, supply lawful data, and preserve an immutable copy and SHA-256 hashes before running. Some scripts query AKShare for the trading calendar and therefore require network access.

Do not fill missing observations or silently change route rules, cost assumptions, parameter values, or validation folds. The studied 2021–2026 period was reused for development and exploration. Independent out-of-sample validation has not been completed; forward collection remains fail-closed.

Scripts write stage outputs to their own directories when run. No code connects to a brokerage or sends orders. This code-only portfolio provides no empirical figures or result tables; it cannot reproduce historical metrics until the user supplies suitable licensed input data.
