# Extensions of distributional equivalence test-then-pool

Two follow-up working papers to Yang, Liu & Evans (2026), *Data Fusion with Distributional Equivalence Test-then-pool* (arXiv:2603.11867).

| | Study A | Study B |
|---|---|---|
| Title | Multi-source Distributional Test-then-pool: Risk-weighted Borrowing, Sourcewise Equivalence Fusion and Selection-aware Permutation Inference | Shrinking Equivalence Radii and Vanishing Borrowing in Distributional Test-then-pool: Consistency, Selection Stability and the Price in Local Power |
| PDF | [paper_A.pdf](docs/papers/paper_A.pdf) | [paper_B.pdf](docs/papers/paper_B.pdf) |
| LaTeX | `paper_A/paper_A.tex` | `paper_B/paper_B.tex` |

Progress log: https://yu-jusang.github.io/ttp-extensions/

## Main findings (see the papers for exact statements)

- **Calibration after selection.** Re-running the whole borrowing selection in every permutation of the current labels ("full reselection") gives exact finite-sample level for any symmetric rule. Freezing the observed selection, as in the original single-source procedure, gave null rejection rates of up to 0.14–0.16 in our simulations.
- **Risk-weighted borrowing (Study A).** Weights minimise an unbiased estimate of the kernel-mean-embedding risk. Because "no borrowing" is always a candidate, the selected weights move the fused control by at most 2κ²/(m−1) in squared MMD, deterministically. This gives uniform √n-rate consistency over all historical distributions. Power gains (+6 to +12 points) appear with compatible sources; losses (up to −15 points) appear when the historical shift mimics the treatment effect.
- **Fixed-radius blind spot.** Sourcewise fusion with a fixed radius has a fixed alternative with power → 0, even with exact calibration. Observed power was 0.009 at m = 160, and 0.000 at n = 800 in Study B.
- **Shrinking radius (Study B).** θ_n → 0 restores consistency against every fixed alternative. Keeping identical sources needs nθ_n² → ∞. The price is a local blind spot of exact order θ_n: no radius schedule both keeps identical sources and is √n-uniformly consistent. A minP safeguard, or soft risk weights, avoids this trade-off.
- **Original partial bootstrap.** It is anti-conservative with merged shifted sources at the sample sizes studied, and conservative when only identical sources are merged; the latter is stated as a conjecture.

## Reproduction

See [`code/README.md`](code/README.md). The full bundle (code, raw results, LaTeX, PDFs) is attached to the GitHub release as `ttp-extensions-reproduction.zip`.

Status: research drafts (18 October 2026), prepared with AI assistance; the proofs have not been independently refereed.
