# Non-official MDVALSE baseline

Boyu Teng's non-official experimental implementation. Project license: MIT; preserve the existing attribution in `M2VALSE_tby2.m` to Qi Zhang, Jiang Zhu and the original VALSE author. See [license and reference details](../THIRD_PARTY_NOTICES.md).

The experiment calls `M2VALSE_tby2(Y, mx, my, ha, X, Mrx, T, nu)` with `ha=2`, `Mrx=64`, `T=100`, and known complex noise variance. The estimated component with maximum amplitude supplies the LoS direction. `X` supplies diagnostic MSE only. `KR.m` provides the required Khatri–Rao product. The released computational bodies are unchanged from the verified experimental snapshot.

Run the shared-observation adapter through `matlab/run_baseline.m`. It evaluates C, B and A at six SNRs each. The internal output filename prefix remains `MVALSE_` for compatibility; plot labels use **MDVALSE**.
