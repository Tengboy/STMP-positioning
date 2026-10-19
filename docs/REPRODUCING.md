# Reproduce the experiments

All commands below run from the release directory. The default generated-artifact directory is `work/`. Each Python stage accepts `--root /absolute/path/to/run`; pass the same path as MATLAB's first argument. Do not combine smoke and full data in one directory. No pretrained models or raw result downloads are required.

## Environment

Full reference computation used Linux, Python 3.11, PyTorch 2.4.1 and an NVIDIA A800 80 GB GPU. MATLAB channel/baseline computation used R2023b on Windows. The release CPU checks used Python 3.9.21 / PyTorch 2.5.1, separately documented in [VERIFICATION.md](VERIFICATION.md). Linux CUDA is the recommended full-run target. Windows GPU compilation has not been validated in this packaging pass.

Create an isolated Python environment (example on Linux):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The CUDA 12.1 wheel command follows [PyTorch's archived 2.4.1 installation instructions](https://pytorch.org/get-started/previous-versions/). The model uses custom CUDA extensions: install a compatible local CUDA toolkit containing `nvcc`, a supported C++ compiler and Ninja. A PyTorch wheel alone does not provide `nvcc`. Use `nvcc --version` and `python -c "import torch; print(torch.__version__, torch.version.cuda)"` to check the toolchain. The first CUDA forward compiles the retained original kernels into the run's `cache/torch_extensions/`; subsequent runs reuse them. The GPU architecture is detected, not forced to A800. CPU calls use the existing native PyTorch operations and do not require compilation.

For MATLAB, the retained generator calls [`physconst`](https://www.mathworks.com/help/phased/ref/physconst.html) from Phased Array System Toolbox. Parallel workers use [Parallel Computing Toolbox](https://www.mathworks.com/help/parallel-computing/). Set workers to `0` for the serial smoke path. Keep `third_party/quadriga_src` named exactly as shipped because this QuaDRiGa version locates its scenario files by that directory name.

Full batch-256 training and batch-64 evaluation were validated on an 80 GB GPU. Smaller hardware may require a smaller batch; changing it can change training and numerical results and must be recorded as a different run. Training loads all C train/validation channels into host memory; allow several GB of available RAM and several GB of disk for data and checkpoints. Step snapshots are large and are deliberately excluded from this source package.

Before a new GPU job, inspect resources and select an available GPU:

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
```

On PowerShell, use `$env:CUDA_VISIBLE_DEVICES='0'` instead of `export`. The exposed GPU is addressed as `cuda:0`. Do not change device visibility while a process is running.

## 1. Generate channels in MATLAB

```matlab
addpath('matlab');
run_dir = fullfile(pwd,'work');
run_channel_generation(run_dir,8,'full');
```

| File under `work/datasets/` | Samples | MATLAB seed IDs |
| --- | ---: | --- |
| CDLC_train.mat | 50000 | 5001–55000 |
| CDLC_val.mat | 2000 | 3001–5000 |
| CDLC_test.mat | 3000 | 1–3000 |
| CDLA_test.mat | 3000 | 1–3000 |
| CDLB_test.mat | 3000 | 1–3000 |

The aggregate C generator represents 480 paths; A/B use 460. It is the actual experiment implementation, rather than independently simulating three interferers. Each channel has unit average complex element power. Channels are saved as MATLAB-v7.3 arrays `[sample, real/imag, 16, 64]`; h5py reads the reversed dimension order. `completed`, `scenario`, `split` and `seed_ids` are embedded. Repeating the same command resumes only matching seed/split definitions. Transfer only a completed file.

## 2. Make common observations

```bash
python scripts/run_experiment.py observations
python scripts/run_experiment.py e0
```

This creates 21 immutable observations: C/A/B at kappa=0 and SNR=0,4,8,12,16,20 dB; plus C at kappa=-5,5,10 dB and SNR=20 dB. The kappa=0/SNR=20 case is shared between E1 and E2. Every saved observation includes `Y`, LoS/interference/noise components, UE position, channel index and trial ID. All methods read the same saved `Y`; they do not independently draw noise. A seeded permutation (12345) and per-trial random generator make observation construction repeatable for a fixed data population. An existing observation is checked against its SHA-256 marker before reuse.

## 3. Train and freeze first order

```bash
python scripts/run_experiment.py train1
python scripts/run_experiment.py freeze1 --inspect
python scripts/run_experiment.py freeze1
```

Defaults are batch 256, seed 42, maximum 100001 updates and 12 cumulative hours; whichever limit comes first stops training. The reference stopped at update 100001 after 10.8807 hours. Fixed C validation uses all 2000 validation channels and fixed noise seed 20261015; both raw and EMA weights are evaluated every 1000 updates and at termination. No C-test or A/B data is accepted as validation. Inspect `work/runs/order1/metrics.jsonl` and `status.json` before freezing. `freeze1` verifies the selected global minimum of the fixed validation history, then writes a compact model and `work/configs/order1_frozen.json`.

If a time limit is reached while validation is still improving, extend only before freezing:

```bash
python scripts/run_experiment.py train1 --resume --max-steps 100001 --max-hours 18
```

Limits are cumulative from the original start, not an additional allowance. `--resume` restores the optimizer, EMA, Torch RNG and minibatch sequence from `last.pth`; changed data, batch, seed, device or code are rejected. Never start two processes against the same run directory. After a hard crash, resume from the most recent saved recovery checkpoint; work after that checkpoint is replayed. The committed validation prefix and its selected best weights are restored together. Any later or incomplete JSONL tail and uncommitted best weights are retained in `runs/orderN/recovery_history/` for audit, and are excluded from subsequent model selection. A clean signal/time stop saves the final state. Do not use `best.pth` as a recovery checkpoint.

Once frozen, training that order in this run directory is refused. Use a new run directory for an independent experiment. Validation loss alone controls selection; loss plateaus do not prove asymptotic convergence.

## 4. Train and freeze second order

```bash
python scripts/run_experiment.py train2
python scripts/run_experiment.py freeze2 --inspect
python scripts/run_experiment.py freeze2
```

Defaults are maximum 141128 updates, 20 cumulative hours, batch 256 and learning rate 5e-5. The first-order frozen record must exist. The runner loads that exact teacher, disables its gradients and verifies both its file identity and its parameter identity at the end. The student alone is optimized. The original implementation's componentwise second-order loss and `[0.01,348]` VE noise range are retained; see [RESULTS.md](RESULTS.md) for the Response wording difference.

The reference ran for 16.0016 hours and selected EMA update 111000 at loss 150.77008972. Exact validation loss and selection step are not guaranteed on another backend. Continuation, if needed, follows the same `--resume` rules as first order and uses C validation only. Freezing records the teacher hash as well as the selected student.

## 5. Evaluate STMP, ML and CRB

```bash
python scripts/run_experiment.py covariance
python scripts/run_experiment.py evaluate
```

The covariance uses C training channels only. Evaluation uses both frozen models, batch 64 and ten fixed updates; no ground-truth early stopping. Each completed batch is saved atomically. Repeating `evaluate` resumes missing batches only when observation/model/code/configuration hashes and CPU/CUDA backend identity match. The effective network configuration and vendored network/kernel sources are included in that identity. Changing batch size or conventions requires a new output tag/run; incompatible partial results are not merged.

The default variant is `historical`: 2D initialization, historical real/imaginary variance averaging, and the historical conjugated covariance direction. To run the separate convention diagnostic with the same frozen pair:

```bash
python scripts/run_experiment.py evaluate --variant paper
```

This selects 1D initialization, real+imaginary variance summation and the corrected covariance direction together. It can produce nonfinite estimates; they are retained as failures, not silently removed. It must not be used to select the networks.

## 6. Evaluate MDVALSE in MATLAB

```matlab
addpath('matlab');
run_baseline(fullfile(pwd,'work'),8,3000);
```

This runs all 18 C/B/A SNR points with the shared observations and the unchanged non-official `M2VALSE_tby2` kernel (`ha=2`, maximum 100 iterations). Failures and direction flags are retained. Repeating the command skips completed trials after checking observation and trial identities.

## 7. Summarize and draw new results

```bash
python scripts/run_experiment.py plots
python scripts/run_experiment.py e0
# Only after the optional convention diagnostic is complete:
python scripts/run_experiment.py plots --variant paper
```

A complete figure run requires all 21 STMP files and all 18 matching baseline files. It writes CSV, generated MAT/NPZ summaries and PNG/PDF figures under the run directory. Those MAT/NPZ outputs are generated locally, not distributed in the source release. E2 displays kappa=0,5,10 dB; the -5 dB experiment remains in the tables. Full-population and historical-mask views are separate.

## Split MATLAB and Python across computers

Use the same release on both hosts. After channel generation completes, copy `work/datasets/` to the Python host. Copy the completed `work/observations/` back for MATLAB MDVALSE. Finally combine the Python `work/results/raw/fresh_*.mat` and MATLAB `work/results/raw/MVALSE_*.mat` files on the plotting host. Baseline SHA checks require byte-identical observation files. Do not resave the MAT files while transferring. Validate SHA-256 at both ends, for example PowerShell `Get-FileHash -Algorithm SHA256 -LiteralPath <file>` and Linux `sha256sum <file>`. Preserve each observation's JSON marker when copying its directory.

Frozen records use relative checkpoint paths, so copying `work/models/` and `work/configs/` together preserves their lookup. SHA-256 values still must match. New runs compute their own hashes; MATLAB headers can change file hashes even when numeric arrays match, so old reference file hashes are provenance rather than a universal acceptance criterion.

## Tiny end-to-end smoke test

This is an interface check, not a performance result. Use a new empty run directory:

```matlab
addpath('matlab');
run_channel_generation(fullfile(pwd,'work','smoke'),0,'smoke');
```

```bash
python scripts/run_experiment.py observations --root work/smoke --count 2
python scripts/run_experiment.py covariance --root work/smoke --device cpu
python scripts/run_experiment.py train1 --root work/smoke --device cpu --batch-size 1 --max-steps 2 --max-hours 0.5 --eval-every 1
python scripts/run_experiment.py freeze1 --root work/smoke
python scripts/run_experiment.py train2 --root work/smoke --device cpu --batch-size 1 --max-steps 2 --max-hours 0.5 --eval-every 1
python scripts/run_experiment.py freeze2 --root work/smoke
python scripts/run_experiment.py evaluate --root work/smoke --device cpu --count 2 --batch-size 2
```

```matlab
run_baseline(fullfile(pwd,'work','smoke'),0,2);
```

```bash
python scripts/run_experiment.py plots --root work/smoke --count 2
python scripts/run_experiment.py e0 --root work/smoke
```

No large datasets or pretrained weights are needed for this check, but the real nf=128 model and optimizer still create substantial temporary checkpoint files. Full reference figures are supplied separately in `results/reference/`.
