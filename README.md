# Spectral Function Modeling (Spectral_fn)

This repository implements a data-efficient deep-kernel Gaussian process surrogate model with sparse variational inference (DKL-SVGP) for spectral functions of the $t$-$t'$-$t''$-$J$ model. It includes training, checkpoint evaluation, and benchmarking scripts for comparing the DKL-SVGP model against feed-forward neural network (FFNN) baselines from Lee, Carbone, and Yin, Phys. Rev. B 107, 205132 (2023).

## Associated paper

This repository accompanies the arXiv preprint:

**[Data-efficient surrogate modeling of spectral functions using Gaussian processes: An application to the $t$-$t'$-$t''$-$J$ model](https://arxiv.org/pdf/2603.13064)**

**Sanket Jantre**, Nathan M. Urban, Weiguo Yin, and Niraj Aryal

The code here supports the Gaussian process-based spectral function modeling, FFNN baseline evaluation, and cross-model comparison workflows used for the study.

## What is in this repo

- `dkl_svgp_flexkernel.py`: main DKL-SVGP training and prediction script. Trains the sparse variational deep-kernel GP model and writes split-wise prediction files.
- `test_ffnn_ckpt.py`: evaluates Lightning FFNN `.ckpt` checkpoints on the train, validation, and test splits.
- `compare_all_models.py`: summarizes model performance and generates comparison plots for DKL-SVGP and FFNN predictions.
- `make_train_subset.py`: creates fixed-fraction training subsets, such as the 10% subset used for data-efficiency comparisons.
- `environment.yml`: conda environment specification for the active training, evaluation, and plotting scripts.

## Expected files

- `data/train-set.npz`, `data/val-set.npz`, `data/test-set.npz`: local train, validation, and test datasets with spectral-function targets.
- `data/train-subset-0.1.npz`: local fixed 10% training subset.
- `checkpoints/`: trained DKL-SVGP and FFNN checkpoint files used for evaluation.
- `outputs/`: saved model predictions and generated PDF comparison plots.

## Data format

The DKL-SVGP and FFNN evaluation scripts expect NPZ files with:

- `params`: input parameters with shape `(N, 3)`, corresponding to $(t', t'', J)$.
- `dos`: spectral-function targets with shape `(N, P)`.
- `W`: optional frequency grid with shape `(P,)`. If missing, scripts synthesize a grid in `[-6, 6]`.

## Data availability

The input datasets are not distributed directly in this repository. To request access to the data files needed to reproduce the runs, please contact Sanket Jantre at sjantre [at] bnl [dot] gov.

## Quick start

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate spectral-fn
```

The `environment.yml` file installs the core dependencies used by the active scripts: PyTorch, GPyTorch, PyTorch Lightning, scikit-learn, NumPy, Matplotlib, and SciencePlots. For GPU-enabled PyTorch on a specific CUDA version, adjust the PyTorch package/channel settings in `environment.yml` for your system.

Train the DKL-SVGP model with the best-known hyperparameters:

```bash
python dkl_svgp_flexkernel.py \
  --train-npz ./data/train-set.npz \
  --val-npz ./data/val-set.npz \
  --test-npz ./data/test-set.npz \
  --save ./checkpoints/ckpt_dkl_svgp_flex_m32_sumprod_M1536_q16_dkl_svgp.pt \
  --save-each-improve \
  --frac 0.10 --seed 0 \
  --epochs 300 --patience 40 --warmup-steps 50 \
  --batch 16384 --float64 \
  --M 1536 --lr 8e-4 \
  --use-ngd --lr-ngd 2e-4 --ngd-delay 30 --ngd-ramp 30 --freeze-kw-epochs 45 \
  --jitter 5e-3 \
  --noise-floor 1e-5 --noise-ceil-warm 0.25 --noise-ceil 0.03 --noise-warm-epochs 35 \
  --d-feat 128 --width 256 --nfreq 16 --nfreq-learned 8 \
  --sm-q 16 --add-kx --kx-kernel m32
```

Evaluate the trained DKL-SVGP checkpoint:

```bash
python dkl_svgp_flexkernel.py \
  --mode predict --float64 \
  --load ./checkpoints/ckpt_dkl_svgp_flex_m32_sumprod_M1536_q16_dkl_svgp.pt \
  --train-npz ./data/train-set.npz \
  --val-npz ./data/val-set.npz \
  --test-npz ./data/test-set.npz \
  --out-prefix ./outputs/preds_dkl_svgp
```

Evaluate an FFNN full-data checkpoint:

```bash
python test_ffnn_ckpt.py \
  --ckpt ./checkpoints/ffnn_full_best_model_[3,170,340,510,680,850,1020,301]_lr_0.001_bs_1024_sch_0.5.ckpt \
  --train-npz ./data/train-set.npz \
  --val-npz ./data/val-set.npz \
  --test-npz ./data/test-set.npz \
  --out-prefix ./outputs/preds_ffnn_full \
  --x-scaler train
```

Evaluate an FFNN subset checkpoint:

```bash
python test_ffnn_ckpt.py \
  --ckpt ./checkpoints/ffnn_subset_best_model_[3,32,64,128,256,301]_lr_0.005_bs_128_sch_0.5.ckpt \
  --train-npz ./data/train-set.npz \
  --val-npz ./data/val-set.npz \
  --test-npz ./data/test-set.npz \
  --out-prefix ./outputs/preds_ffnn_subset \
  --x-scaler train
```

Generate cross-model summary plots from saved prediction files:

```bash
python compare_all_models.py
```

This writes comparison PDFs to `outputs/`.

## Notes

- The main DKL-SVGP model is implemented in `dkl_svgp_flexkernel.py`.
- The FFNN evaluator assumes Lightning checkpoint files compatible with the `LitFFNN` class in `test_ffnn_ckpt.py`.
- `.gitignore` excludes OS/editor files, Python caches, notebook checkpoints, `data/`, and `archive/`.

## Citation

```bibtex
@article{jantre2026data,
  title={Data-efficient surrogate modeling of spectral functions using Gaussian processes: An application to the $t$-$t'$-$t''$-$J$ model},
  author={Jantre, Sanket and Urban, Nathan M and Yin, Weiguo and Aryal, Niraj},
  journal={arXiv preprint arXiv:2603.13064},
  year={2026}
}
```