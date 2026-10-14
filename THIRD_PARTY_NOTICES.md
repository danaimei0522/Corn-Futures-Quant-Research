# Third-party dependency notices and license review

## Direct runtime dependencies

The selected scripts import the following top-level packages. They are installed from their package distributions; no third-party package source or binary is vendored in this repository.

| Package | Pinned version | License reported by upstream | Primary license/source reference |
|---|---:|---|---|
| AKShare | 1.19.1 | MIT | [AKShare LICENSE](https://github.com/akfamily/akshare/blob/main/LICENSE) |
| NumPy | 2.5.4 | BSD-3-Clause and licenses for included components | [NumPy license](https://numpy.org/doc/stable/license.html) |
| pandas | 3.0.6 | BSD-3-Clause | [pandas LICENSE](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| Matplotlib | 3.11.2 | Python Software Foundation License / Matplotlib license agreement | [Matplotlib license](https://github.com/matplotlib/matplotlib/blob/main/doc/project/license.rst) |

## Resolved dependency closure snapshot

The top-level packages above resolve to the following **35 installed distributions** in the checked Python 3.13.1 macOS environment. License identifiers/text labels are read from installed package metadata and normalized for display. This is a point-in-time, platform-specific inventory; it is not a complete SBOM for every platform, nor legal advice. All 35 packages reported license metadata in that environment. Dependencies are installed separately with pip, and their own distributions carry their license notices.

| Distribution | Version | Metadata license label |
|---|---:|---|
| akshare | 1.19.1 | MIT License |
| beautifulsoup4 | 4.15.0 | MIT License |
| certifi | 2026.7.22 | Mozilla Public License 2.0 (MPL 2.0) |
| cffi | 2.1.1 | MIT-0 |
| charset-normalizer | 3.5.2 | MIT |
| contourpy | 1.4.0 | BSD-3-Clause |
| curl-cffi | 0.16.3 | MIT |
| cycler | 0.12.1 | BSD License |
| decorator | 5.3.1 | BSD-2-Clause |
| et-xmlfile | 2.0.0 | MIT License |
| fonttools | 4.66.1 | MIT |
| html5lib | 1.1 | MIT License |
| idna | 3.20 | BSD-3-Clause |
| jsonpath | 0.82.2 | MIT License |
| kiwisolver | 1.5.1 | BSD License |
| lxml | 6.1.3 | BSD-3-Clause |
| matplotlib | 3.11.2 | Python Software Foundation License |
| mini-racer | 0.14.1 | ISC |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| openpyxl | 3.1.5 | MIT License |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.6 | BSD License |
| pillow | 12.3.0 | MIT-CMU |
| pycparser | 3.11 | BSD-3-Clause |
| pyparsing | 3.3.3 | MIT |
| python-dateutil | 2.9.0.post0 | Apache Software License / BSD License |
| requests | 2.34.2 | Apache Software License |
| six | 1.17.0 | MIT License |
| soupsieve | 3.0.3 | MIT |
| tabulate | 0.10.0 | MIT |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| typing-extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.8.0 | MIT |
| webencodings | 0.6.1 | BSD License |
| xlrd | 2.0.2 | BSD License |

## Scope and cautions

- No third-party dependency implementation has been copied into this project code. AKShare is called as an installed library by some scripts (for the trading calendar); importing it does not transfer rights to data providers.
- The AKShare software license governs AKShare software. Sina Finance, DCE, and other upstream market-data owners retain separate rights and terms. No source-data license or written permission for redistribution of this project's empirical outputs was found. All data-derived tables, reports, charts, and result files are therefore excluded from this code-only portfolio.
- No license has been selected for the project's own code. This file documents dependencies only and does not grant a license to project code or market data. See [CODE_PROVENANCE.md](CODE_PROVENANCE.md).
- Re-check current package metadata and bundled component notices before changing dependency versions or redistributing any dependency itself.
