# Code-Only Release Safety Review

**Status: READY TO PUBLISH AS CODE-ONLY**

## Scope and data boundary

The candidate contains project-specific research scripts and documentation only. It does not contain third-party market data, API responses, empirical tables, strategy performance figures, reports with empirical results, or other data-derived artifacts. No source-specific redistribution permission was found for the underlying AKShare/Sina/DCE inputs, so those files and all derived deliverables remain excluded. See [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md).

## Research explanation

The README describes the research objectives, moving-average/ADX/ATR methods, actual-contract and roll handling, factor Rank IC/HAC approach, time-ordered purging, missing-data handling, data requirements, reproduction steps, and the fact that independent out-of-sample validation has not been completed. It does not include unsupported or fabricated numerical results.

## Project code and license

Nine research scripts were generated for this project collaboration and reviewed for third-party implementation snippets and broker/order functionality. No dependency code is vendored. OpenAI's [Terms of Use](https://openai.com/policies/row-terms-of-use/?app=1) assign service outputs to the user as between the parties and to the extent permitted by law; they also note output may not be unique and the user is responsible for use. The code is prepared for public display as the project output.

No explicit project-code reuse license has been added. This permits a code-only portfolio presentation without granting a separate open-source license; if third parties should reuse or modify the code, the owner should choose a license. Dependency notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Third-party software dependency review

Four direct dependencies are pinned; their published software licenses are recorded. The current macOS/Python 3.13.1 resolved closure contains 35 distributions, all with license metadata in the installed package records. A version snapshot is provided; it is platform-specific and has no wheel hashes. No dependencies are bundled. The AKShare software license does not grant upstream market-data rights.

## Automated checks

- Package verifier: file inventory and SHA-256 integrity, no data/result/chart file types, no common credential patterns or personal absolute paths, and syntax compilation of all included Python scripts.
- Local Markdown links resolve.
- Original source/result hashes and the original portfolio validator remain valid; research data, strategy parameters, and historical results were not changed.
- No GitHub remote, commit, or upload was created.

## Limitations

This bundle demonstrates code and methodology but is not a turnkey empirical reproduction: user-provided licensed historical data are required. No independent out-of-sample validation has been completed. Empirical figures, tables, and reports will remain excluded unless redistribution permission covering those outputs is documented.
