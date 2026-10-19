# Release verification

This release was checked on 2026-10-18. The completed full reproduction and its immutable results were reused; **full training and all 3000-trial Monte Carlo runs were not repeated during packaging**. The new packaging checks were isolated from the original experiment and did not access or modify the server source trees.

## Executed checks

- The 27-test Python suite passed: geometry/vectorization, analytical CRB, historical grid/refinement and Turbo update equivalence, finite-difference gradients, real/imaginary variance conventions, failure retention, historical masks, shared-observation identity, minibatch resume, frozen teacher updates, fixed-validation checkpoint selection CPU-safe model import, committed-history recovery, backend/network identity and nonfinite-curve annotations.
- MATLAB R2023b generated 12 small channels: C train 4, C validation 2, C test 2, A test 2 and B test 2. All complex-channel arrays were **elementwise identical** to the matching seed prefixes of the validated original reproduction. A smoke failure exposed QuaDRiGa's required folder name; retaining `quadriga_src` fixed it without changing numerical code.
- A separate run directory containing spaces completed the real nf=128 CPU path: shared observations, training covariance, first-order update 1 followed by resume to update 2, validation-selected freeze, refusal to train frozen order 1, second-order update 1 with fixed teacher, and second-order freeze. The teacher file/tensor checks passed. This tests the interfaces and guards, not the learning quality of a one-step student.
- The CPU smoke evaluation completed all 21 cases with 2 trials each and ten updates, and MATLAB MDVALSE completed all 18 cases with 2 trials each. Their shared observation identities passed summary validation. E0–E3 figures and complete summaries were generated from those tiny outputs.
- A second real CPU run injected a newer uncommitted best checkpoint and an incomplete JSONL tail, resumed the committed first-order checkpoint, and successfully froze the validation-selected model. The discarded evidence remained in the recovery archive. A CPU evaluation recorded its backend and effective network identity; repeating it left its completed result hash unchanged.
- Offline redraw consumed only the released aggregate CSV/JSON files and generated E0–E3 plus training plots. E3 A/B MDVALSE lines and E2 kappa=0/5/10 were visually inspected. The separate paper-convention E2 redraw explicitly shows 7/3000 final failures for each of its three nonfinite STMP curves.
- Source-content comparison checks that architecture, score losses, CUDA kernels, localization/Turbo arithmetic, custom channel generators and MDVALSE computational files match the completed experiment. Only runner/configuration portability, lazy extension loading, output selection and documentation are changed. `SOURCE_MANIFEST.json` records original-source hashes for copied files and shipped hashes for all files.

## Environment and boundaries

Local packaging environment:

```json
{
  "python": "3.9.21",
  "torch": "2.5.1+cu121",
  "numpy": "1.26.4",
  "scipy": "1.13.1",
  "h5py": "3.12.1",
  "matplotlib": "3.9.2"
}
```

`ml-collections` 0.1.1 and Ninja 1.11.1.1 were installed in a task-local dependency directory. The recommended `requirements*.txt` pins are the portable environment recipe; this packaging pass did not reinstall the whole Python environment to those exact versions. The full reference GPU run used Linux / Python 3.11 / PyTorch 2.4.1 / A800 80 GB, as recorded in the reference metadata and reproduction notes.

The CUDA kernels were retained, but a fresh CUDA compilation/full GPU run of the release was not performed here; the local GPU was already in use. CPU import/forward and the end-to-end CPU path were verified. Exact CUDA output equivalence across hardware/batches is not claimed. The complete Windows-to-Linux file-transfer sequence is documented rather than automatically performed by this release.

A final archive extraction, file-hash check, unit suite and aggregate redraw check are recorded in `release_checks.json`. The shipped `SOURCE_MANIFEST.json` excludes itself to avoid a circular hash. Use `python scripts/verify_package.py` after extracting the package. Newly generated `work/` files are outside that immutable source manifest.
