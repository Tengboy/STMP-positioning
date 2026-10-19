# Reference results and statistical conventions

The bundled results are from **newly generated channels and newly trained networks**, completed on 2026-10-17, with the A/B MDVALSE extension completed on 2026-10-18. They are not the paper's original raw MAT archive. Each of 21 parameter points has 3000 trials; each of 18 C/A/B MDVALSE points has 3000 trials. Two STMP numerical variants are retained separately.

| Experiment | Paper / Response mapping | Parameters |
| --- | --- | --- |
| E0 | Paper Fig. 2; Response p. 2 Fig. A | CDL-C interference angular FFT |
| E1 | Paper Fig. 4 | C, kappa=0, SNR 0:4:20 dB |
| E2 | Paper Fig. 5 | C, SNR 20 dB, kappa -5/0/5/10; displayed 0/5/10 |
| E3 | Response p. 6 Fig. B and p. 9 Fig. A | C-trained models tested on B and A; SNR 0:4:20 dB |

## Geometry and scaling

The array is 64 by 16 with half-wavelength spacing at 28 GHz. BS position is [0,0,25] m; UE height is 1.5 m. The horizontal angle is 30–150 degrees and horizontal radius is 10–30 m. Interference mean element power is one. LoS power is `10^(kappa/10)` and complex noise power is `10^((kappa-SNR)/10)`; real and imaginary noise variances are each half of the complex variance. Position RMSE is the square root of the mean squared 3D position error in meters.

The user-confirmed aggregate generator is preserved: C has 480 paths; A/B have 460. This is the executable implementation behind the experiments. E0 uses seed 1 and an unnormalized 2D FFT; the paper's realization seed is unspecified, so its pixels are not expected to match.

## Numerical variants

| Choice | `historical` (primary executable reproduction) | `paper` (diagnostic) |
| --- | --- | --- |
| Initialization | 2D angular grid | 1D initialization |
| Score posterior variance | Historical real/imaginary mean | Sum of real/imaginary variances |
| Covariance | Historical conjugate direction | E[h h^H] in evaluator coordinates |
| Models | Same freshly trained frozen pair | Same pair, no reselection |
| Iterations and batch | Fixed 10, batch 64 | Fixed 10, batch 64 |

The second-order training target is the original componentwise diagonal expression, `sigma^2*s2 + 1 - (sigma*s1 + z)^2`, squared and reduced by the original loss. It is not silently replaced by a trace-form interpretation of the Response. Simultaneously changing initialization and variance conventions does not isolate a single cause of differences between the two variants.

## Selected models

| Order | Final training update | Selected weights | Fixed C-validation loss | Training hours |
| --- | ---: | --- | ---: | ---: |
| 1 | 100001 | EMA, update 100001 | 175.42756628 | 10.8807 |
| 2 | 141128 | EMA, update 111000 | 150.77008972 | 16.0016 |

Only the fixed 2000-channel C-validation split and fixed validation noise were used to select weights. The second-order teacher's identity was unchanged. The second-order best was not improved for the final 30128 updates; this supports a bounded plateau decision, not proof of asymptotic convergence. Model file and tensor SHA-256 identities are in `results/reference/metadata.json`; the corresponding weights are not included. Train from scratch using the recipe. A future separately hosted weight release should identify these hashes; this package does not invent a download URL.

## Representative all-trial values

Historical executable route, kappa=0, SNR=20 dB:

| Scenario | STMP RMSE (m) | MDVALSE RMSE (m) |
| --- | ---: | ---: |
| C | 0.0125173009 | 0.0285342599 |
| B | 0.0124357855 | 0.0269684356 |
| A | 0.0456278650 | 0.0537200506 |

The C LoS CRB at 20 dB is 0.0113085724 m. All 54000 MDVALSE trials in the reference extension produced finite estimates, with no direction flags. A/B MDVALSE curves are included in E3.

## Full population versus historical filtering

`*_summary.csv` includes completed, pending and failed counts, full-population RMSE, finite-subset RMSE, finite quantiles, historical retained count and historical RMSE. Nonfinite completed estimates produce infinite full RMSE and are never converted to zero. The finite-subset column is not a substitute for unconditional RMSE.

Historical C/B STMP and ML-AWGN SNR curves retain final squared errors <=1 m^2; A remains unfiltered, matching the earlier plot script's actual plotted arrays. Other methods are not automatically given that mask. Because filters are method-specific, those figures are conditional diagnostics, not fair unconditional comparisons.

E2 main views use grid initialization plus all 10 updates. Historical E2 views retain whole trajectories using final squared error <=1 m^2, display initialization plus only the first 9 updates, and use the first STMP refinement as the historical ML proxy with its own first-update mask. The true independently initialized ML-AWGN baseline is shown in the all-trial view.

The displayed E2 curves intentionally omit kappa=-5 dB, as requested by the code owner. That point remains in both summary and iteration CSVs: its historical-route all-trial final RMSE is **9.723616645 m**; retaining 2991/3000 trials gives **0.013515345 m**. The nine excluded trials are finite extreme errors. The earlier historical experiment also had such outliers. Omission from the figure is not a correction of this failure mode or evidence of unconditional strong-interference robustness.

The optional `paper` diagnostic has 144 nonfinite final estimates across 63000 trial/parameter combinations, corresponding to nonfinite initialization. All remain in the summaries. This diagnostic is not presented as a failure-free reproduction.

## Reproducibility limits

The primary historical reference comparison matched 21 old plotting points to within 2.075%, but that descriptive comparison includes documented filtering and is not a paired significance test. Full-population performance and conditional historical curves must remain distinguishable.

Strict 0.1 mm agreement across different inference batch sizes was not established: the earlier full-size check reached about 5.7 mm maximum position difference. The reference therefore fixes batch 64 and records backend/configuration identities. Float32 kernels, GPU hardware and software versions may change trajectories; new independent training is not promised to reproduce identical weights or every digit of the supplied tables.

The source package preserves these limitations rather than selecting models or samples to hide them. See [VERIFICATION.md](VERIFICATION.md) for release packaging checks.
