#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dkl_svgp_flexkernel.py
================================
DKL + SVGP model with train and predict modes:
  k = s1*(kx * kw) + [s2*kx] + [s3*kw]

- NPZ keys expected: X='params' (N,3), Y='dos' (N,P), optional W (P,)
- If W missing, synthesize linspace[-6,6] with length P
- NGD (variational params) + Adam (others), safe param grouping
- Spectral Mixture kernel for ω with on-device initialization
- kx choices: Matérn (m12/m32/m52) or RBF
- Optional additive branches: --add-kx, --add-kw

Usage example:
python dkl_svgp_flexkernel.py \
  --train-npz ./data/train-set.npz --val-npz ./data/val-set.npz --test-npz ./data/test-set.npz \
  --save ckpt.pt --save-each-improve \
  --frac 0.10 --seed 0 \
  --epochs 300 --patience 40 --warmup-steps 50 \
  --batch 16384 --float64 \
  --M 1536 --lr 8e-4 \
  --use-ngd --lr-ngd 2e-4 --ngd-delay 30 --ngd-ramp 30 --freeze-kw-epochs 45 \
  --jitter 5e-3 \
  --noise-floor 1e-5 --noise-ceil-warm 0.25 --noise-ceil 0.03 --noise-warm-epochs 35 \
  --d-feat 128 --width 256 --nfreq 16 --nfreq-learned 8 \
  --sm-q 16 --add-kx --kx-kernel m32

python dkl_svgp_flexkernel.py \
  --mode predict --float64 \
  --load ckpt_dkl_svgp_flex_m32_sumprod_M1536_q16_dkl_svgp.pt \
  --train-npz ./data/train-set.npz --val-npz ./data/val-set.npz --test-npz ./data/test-set.npz \
  --out-prefix preds_dkl_svgp
"""

import argparse, os, time, inspect
import numpy as np
import torch
import gpytorch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from contextlib import ExitStack

# ---------------- I/O ----------------

def load_npz_fixed(path, role="train", wmin=-6.0, wmax=6.0):
    z = np.load(path, allow_pickle=True)
    files = set(z.files)

    if not {'params', 'dos'}.issubset(files):
        raise KeyError(f"[{role}] needs 'params' & 'dos'. Found: {list(z.files)}")
    X = z['params'].astype(np.float32)
    Y = z['dos'].astype(np.float32)

    if 'W' in z.files:
        W = z['W'].astype(np.float32).reshape(-1)
        src = "file:'W'"
    else:
        P = Y.shape[1]
        W = np.linspace(wmin, wmax, P, dtype=np.float32)
        src = f"synth:[{wmin},{wmax}] P={P}"

    print(f"[Load-{role}] {os.path.basename(path)}  X=params{X.shape}  Y=dos{Y.shape}  W=<{src}>")
    return X, Y, W

def device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -------------- transforms --------------

@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    def transform(self, A: np.ndarray) -> np.ndarray:
        return (A - self.mean) / (self.std + 1e-8)
    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return Z * (self.std + 1e-8) + self.mean

@dataclass
class LogStdPerW:
    mu_w: np.ndarray
    sd_w: np.ndarray
    W_ref: np.ndarray
    eps: float = 1e-3
    def _mu_sd_on(self, Wnew):
        mu = np.interp(Wnew, self.W_ref, self.mu_w).astype(np.float32)
        sd = np.interp(Wnew, self.W_ref, self.sd_w).astype(np.float32)
        return mu, sd
    def transform(self, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
        Z = np.log(np.maximum(Y, 0.0) + self.eps).astype(np.float32)
        mu, sd = self._mu_sd_on(W)
        return (Z - mu.reshape(1,-1)) / (sd.reshape(1,-1) + 1e-8)
    def inverse(self, Zstd: np.ndarray, W: np.ndarray) -> np.ndarray:
        mu, sd = self._mu_sd_on(W)
        Z = Zstd * (sd.reshape(1,-1) + 1e-8) + mu.reshape(1,-1)
        return np.exp(Z) - self.eps

def fit_x_standardizer(X):
    return Standardizer(X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True))

def fit_logstd_per_w(Y, W, eps=1e-3):
    Z = np.log(np.maximum(Y, 0.0) + eps).astype(np.float32)
    return LogStdPerW(Z.mean(axis=0), Z.std(axis=0), W.astype(np.float32), eps=eps)

def normalize_w(W):
    lo, hi = float(W.min()), float(W.max())
    return ((W - lo) / (hi - lo + 1e-12) * 2.0 - 1.0).astype(np.float32), lo, hi

def build_pairs(Xs, Ws, Y=None):
    N, P = Xs.shape[0], Ws.shape[0]
    Xrep = np.repeat(Xs, P, axis=0).astype(np.float32)
    Wrep = np.tile(Ws.reshape(1,-1), (N,1)).reshape(-1,1).astype(np.float32)
    y = None if Y is None else Y.reshape(-1).astype(np.float32)
    return np.concatenate([Xrep, Wrep], axis=1), y

def select_fixed_fraction(N, frac, seed):
    rng = np.random.RandomState(seed)
    k = max(1, int(round(frac * N)))
    idx = np.sort(rng.choice(N, size=k, replace=False))
    return idx

def safe_gpytorch_context(jitter: float):
    stack = ExitStack()
    stack.enter_context(gpytorch.settings.cholesky_jitter(jitter))
    try: stack.enter_context(gpytorch.settings.max_cholesky_size(8192))
    except Exception: pass
    try: stack.enter_context(gpytorch.settings.fast_computations(False, False, False))
    except Exception: pass
    return stack

def clamp_likelihood_noise(lik, epoch, args):
    ceil = args.noise_ceil_warm if epoch <= args.noise_warm_epochs else args.noise_ceil
    with torch.no_grad():
        lik.noise.clamp_(args.noise_floor, ceil)

def _robust_torch_load_v2(path):
    try:
        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters:
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                try:
                    from torch.serialization import add_safe_globals
                    import numpy.core.multiarray as ncm
                    add_safe_globals([ncm._reconstruct])
                    return torch.load(path, map_location="cpu", weights_only=True)
                except Exception:
                    return torch.load(path, map_location="cpu")
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")

def clamp_kernel_raw_params(module, max_len_raw=6.0, max_out_raw=6.0, max_sm_raw=8.0):
    """Clamp gpytorch raw params to reduce overflow/NaN risk."""
    for m in module.modules():
        if isinstance(m, (gpytorch.kernels.RBFKernel,
                          gpytorch.kernels.MaternKernel,
                          gpytorch.kernels.RQKernel)):
            if hasattr(m, "raw_lengthscale") and m.raw_lengthscale is not None:
                m.raw_lengthscale.data.clamp_(-max_len_raw, max_len_raw)

        if isinstance(m, gpytorch.kernels.ScaleKernel):
            if hasattr(m, "raw_outputscale") and m.raw_outputscale is not None:
                m.raw_outputscale.data.clamp_(-max_out_raw, max_out_raw)

        if isinstance(m, gpytorch.kernels.SpectralMixtureKernel):
            if hasattr(m, "raw_mixture_means"):
                m.raw_mixture_means.data.clamp_(-max_sm_raw, max_sm_raw)
            if hasattr(m, "raw_mixture_scales"):
                m.raw_mixture_scales.data.clamp_(-5.0, 5.0)
            if hasattr(m, "raw_mixture_weights"):
                m.raw_mixture_weights.data.clamp_(-max_sm_raw, max_sm_raw)

def clamp_inducing_points(model, x_clip=5.0, w_clip=1.2):
    """Keep inducing locations in a sane box."""
    with torch.no_grad():
        Z = model.variational_strategy.inducing_points
        Z[..., :3].data.clamp_(-x_clip, x_clip)
        Z[..., 3:4].data.clamp_(-w_clip, w_clip)

def set_kw_requires_grad(model, flag: bool):
    """Freeze/unfreeze ω-kernel params (Spectral Mixture)."""
    if hasattr(model, "kw"):
        for p in model.kw.parameters():
            p.requires_grad_(flag)

# -------------- Feature net --------------

class LearnableFourierTrunk(nn.Module):
    def __init__(self, nfreq_fixed=16, nfreq_learned=8, width=128):
        super().__init__()
        self.nf = nfreq_fixed
        self.nl = nfreq_learned
        self.log_freq = nn.Parameter(torch.zeros(self.nl))
        self.net = nn.Sequential(
            nn.Linear(2*self.nf + 2*self.nl, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
    def forward(self, w):
        k = torch.arange(1, self.nf+1, device=w.device, dtype=w.dtype).view(1,-1)
        a = 2.0 * np.pi * k * w
        fixed = torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

        f = torch.nn.functional.softplus(self.log_freq)
        a2 = (2.0*np.pi) * w * f.view(1,-1)
        learned = torch.cat([torch.sin(a2), torch.cos(a2)], dim=-1)

        return self.net(torch.cat([fixed, learned], dim=-1))

class FeatureNet(nn.Module):
    def __init__(self, d_feat=64, width=128, nfreq_fixed=16, nfreq_learned=8):
        super().__init__()
        self.hx = nn.Sequential(
            nn.Linear(3, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.hw = LearnableFourierTrunk(nfreq_fixed, nfreq_learned, width)
        self.proj = nn.Sequential(nn.Linear(3*width, d_feat))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xw):
        x = xw[..., :3]
        w = xw[..., 3:4]
        hx = self.hx(x)
        hw = self.hw(w)
        inter = hx * hw
        z = self.proj(torch.cat([hx, hw, inter], dim=-1))
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        z = z * (z.shape[-1] ** 0.5)
        return z

# -------------- DKL + SVGP (kernel family) --------------

def _make_kx(kind, d_feat, feat_idx):
    if kind == 'rbf':
        return gpytorch.kernels.RBFKernel(ard_num_dims=d_feat, active_dims=feat_idx)
    if kind == 'm12':
        return gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=d_feat, active_dims=feat_idx)
    if kind == 'm52':
        return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d_feat, active_dims=feat_idx)
    return gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=d_feat, active_dims=feat_idx)  # m32 default

class DKL_SVGP(gpytorch.models.ApproximateGP):
    """
    k = s1*(kx * kw) + [s2*kx] + [s3*kw]
    """
    def __init__(self, inducing_Z, feature_net, d_feat,
                 sm_q=8, kx_kind='m32',
                 add_kx=False, add_kw=False,
                 learn_Z=True):
        q_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_Z.size(-2))
        VS = gpytorch.variational.VariationalStrategy
        sig = inspect.signature(VS.__init__)
        if "whiten" in sig.parameters:
            vs = VS(self, inducing_Z, q_dist, learn_inducing_locations=learn_Z, whiten=True)
        else:
            vs = VS(self, inducing_Z, q_dist, learn_inducing_locations=learn_Z)
        super().__init__(vs)

        self.feature = feature_net
        self.mean_module = gpytorch.means.ZeroMean()

        feat_idx  = list(range(d_feat))
        omega_idx = [d_feat]  # last coord in z_joint

        self.kx  = _make_kx(kx_kind, d_feat, feat_idx)
        self.kw  = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=sm_q, ard_num_dims=1, active_dims=omega_idx
        )

        kernels = [gpytorch.kernels.ScaleKernel(self.kx * self.kw)]  # s1*(kx*kw)
        if add_kx:
            kernels.append(gpytorch.kernels.ScaleKernel(self.kx))    # s2*kx
        if add_kw:
            kernels.append(gpytorch.kernels.ScaleKernel(self.kw))    # s3*kw

        cov = kernels[0]
        for k in kernels[1:]:
            cov = cov + k
        self.covar_module = cov

    def forward(self, xw):
        z_feat = self.feature(xw)
        w_last = xw[..., 3:4]
        z_joint = torch.cat([z_feat, w_last], dim=-1)
        mean = self.mean_module(z_joint)
        cov  = self.covar_module(z_joint)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

# -------------- SM init --------------

def init_sm_on_device(sm_kernel):
    dev = sm_kernel.raw_mixture_means.device
    dt  = sm_kernel.raw_mixture_means.dtype
    Q   = sm_kernel.num_mixtures

    means  = torch.linspace(0.0, 0.5, Q, device=dev, dtype=dt).unsqueeze(-1)
    scales = torch.full_like(means, 0.15)
    weights= torch.full((Q,), 1.0/Q, device=dev, dtype=dt)

    sm_kernel.mixture_means   = means
    sm_kernel.mixture_scales  = scales
    sm_kernel.mixture_weights = weights

# -------------- inducing init --------------

def kmeans_inducing(Xw_np, M, seed):
    from sklearn.cluster import KMeans
    Z = Xw_np.astype(np.float32)
    M = min(M, Z.shape[0])
    km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(Z)
    return torch.from_numpy(km.cluster_centers_).float()

# -------------- evaluation helpers --------------

@torch.no_grad()
def predict_all(model, lik, Xs, W, y_tf, args):
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    Ws, _, _ = normalize_w(W)
    N, P = Xs.shape[0], len(Ws)
    Ystd = np.zeros((N, P), dtype=np.float32)

    model.eval(); lik.eval()
    for i in range(N):
        Xrep = np.repeat(Xs[i:i+1], P, axis=0)
        Xw = np.concatenate([Xrep, Ws.reshape(-1,1).astype(np.float32)], axis=1)

        outs = []
        start = 0
        while start < P:
            xb = torch.from_numpy(Xw[start:start+args.pred_batch]).to(dev, dtype=dtype)
            with safe_gpytorch_context(args.jitter):
                pr = lik(model(xb))
            outs.append(pr.mean.detach().cpu().numpy().ravel())
            start += args.pred_batch

        Ystd[i] = np.concatenate(outs, axis=0)

    return y_tf.inverse(Ystd, W)

def mean_row_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1)).mean())

def evaluate_rmse(model, lik, Xs, Y, W, y_tf, args):
    Yhat = predict_all(model, lik, Xs, W, y_tf, args)
    return mean_row_rmse(Y, Yhat)

# -------------- ckpt save --------------

def save_ckpt(path, model, lik, x_std, y_tf, arch_meta: dict, idx_tr, idx_va):
    ckpt = dict(
        model_state=model.state_dict(),
        lik_state=lik.state_dict(),
        inducing_points=model.variational_strategy.inducing_points.detach().cpu(),
        x_mean=x_std.mean, x_std=x_std.std,
        y_mu=y_tf.mu_w, y_sd=y_tf.sd_w, y_Wref=y_tf.W_ref, y_eps=float(y_tf.eps),
        arch=arch_meta, idx_tr=idx_tr, idx_va=idx_va,
        gpytorch_version=gpytorch.__version__, torch_version=torch.__version__
    )
    torch.save(ckpt, path)
    print(f"[Save] {path}")

def rebuild_from_ckpt(ckpt, dev, dtype):
    feat = FeatureNet(
        d_feat=ckpt['arch']['d_feat'],
        width=ckpt['arch']['width'],
        nfreq_fixed=ckpt['arch']['nfreq'],
        nfreq_learned=ckpt['arch']['nfreq_learned']
    )
    Z0 = ckpt['inducing_points'].to(dtype=torch.float32)
    a = ckpt['arch']
    model = DKL_SVGP(
        Z0, feat, d_feat=a['d_feat'],
        sm_q=a['sm_q'], kx_kind=a.get('kx_kernel', 'm32'),
        add_kx=a.get('add_kx', False), add_kw=a.get('add_kw', False),
        learn_Z=not a.get('freeze_Z', False)
    ).to(dev).to(dtype)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(dev).to(dtype)
    model.load_state_dict(ckpt['model_state'], strict=True)
    lik.load_state_dict(ckpt['lik_state'], strict=True)
    x_std = Standardizer(ckpt['x_mean'], ckpt['x_std'])
    y_tf = LogStdPerW(mu_w=ckpt['y_mu'], sd_w=ckpt['y_sd'], W_ref=ckpt['y_Wref'], eps=float(ckpt['y_eps']))
    return model, lik, x_std, y_tf

# -------------- training --------------

def prepare_subsets(Xtr_full, Ytr_full, Wtr, Xva_full, Yva_full, Wva, frac, seed):
    itr = select_fixed_fraction(Xtr_full.shape[0], frac, seed)
    iva = select_fixed_fraction(Xva_full.shape[0], frac, seed+1)
    return (Xtr_full[itr], Ytr_full[itr], Wtr, itr), (Xva_full[iva], Yva_full[iva], Wva, iva)

def train_once(args):
    dev = device()
    dtype = torch.float64 if args.float64 else torch.float32
    torch.set_default_dtype(dtype)

    # Load sets
    Xtr_full, Ytr_full, Wtr = load_npz_fixed(args.train_npz, 'train')
    Xva_full, Yva_full, Wva = load_npz_fixed(args.val_npz,   'val')
    Xte_full, Yte_full, Wte = load_npz_fixed(args.test_npz,  'test')

    # Subselect train/val
    (Xtr, Ytr, Wtr, idx_tr), (Xva, Yva, Wva, idx_va) = prepare_subsets(
        Xtr_full, Ytr_full, Wtr, Xva_full, Yva_full, Wva, args.frac, args.seed
    )

    # Transforms
    x_std = fit_x_standardizer(Xtr)
    Xtr_s = x_std.transform(Xtr).astype(np.float32)
    Xva_s = x_std.transform(Xva).astype(np.float32)

    y_tf  = fit_logstd_per_w(Ytr, Wtr, eps=args.log_eps)
    Ytr_t = y_tf.transform(Ytr, Wtr)
    Yva_t = y_tf.transform(Yva, Wva)

    # Pairs (x, ω)
    Ws_tr, _, _ = normalize_w(Wtr)
    Ws_va, _, _ = normalize_w(Wva)
    Xw_tr, y_tr = build_pairs(Xtr_s, Ws_tr, Ytr_t)
    Xw_va, y_va = build_pairs(Xva_s, Ws_va, Yva_t)

    # Model
    feat = FeatureNet(d_feat=args.d_feat, width=args.width,
                      nfreq_fixed=args.nfreq, nfreq_learned=args.nfreq_learned)

    with torch.no_grad():
        sample = Xw_tr[: min(50000, Xw_tr.shape[0])]
        Z0 = kmeans_inducing(sample, M=args.M, seed=args.seed)

    model = DKL_SVGP(
        Z0, feat, d_feat=args.d_feat,
        sm_q=args.sm_q, kx_kind=args.kx_kernel,
        add_kx=args.add_kx, add_kw=args.add_kw,
        learn_Z=not args.freeze_Z
    ).to(dev).to(dtype)

    lik = gpytorch.likelihoods.GaussianLikelihood().to(dev).to(dtype)
    with torch.no_grad():
        lik.noise = torch.tensor(0.2, dtype=dtype, device=dev)

    init_sm_on_device(model.kw)

    # Optional: initialize kx lengthscale ~ median distance in feature space
    try:
        with torch.no_grad():
            samp = torch.from_numpy(Xw_tr[: min(10000, len(Xw_tr))]).to(dev, dtype=dtype)
            z = model.feature(samp)
            med = torch.cdist(z[:2048], z[:2048]).median().clamp(min=1e-3)
            model.kx.lengthscale = med
    except Exception:
        pass

    # ---- Optimizers (disjoint param groups) ----
    var_ids = {id(p) for p in model.variational_parameters()}  # NGD only
    feat_params = [p for p in model.feature.parameters() if id(p) not in var_ids]
    feat_ids = {id(p) for p in feat_params}
    other_params = [p for p in model.parameters() if id(p) not in var_ids and id(p) not in feat_ids]
    lik_params = list(lik.parameters())

    if args.use_ngd:
        ngd = gpytorch.optim.NGD(model.variational_parameters(), num_data=len(Xw_tr), lr=args.lr_ngd)
        adam = torch.optim.Adam(
            [{"params": feat_params, "weight_decay": 1e-5},
             {"params": other_params},
             {"params": lik_params}],
            lr=args.lr
        )
        optimizers = (ngd, adam)
    else:
        adam = torch.optim.Adam(
            [{"params": feat_params, "weight_decay": 1e-5},
             {"params": other_params},
             {"params": lik_params}],
            lr=args.lr
        )
        optimizers = (adam,)

    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=len(Xw_tr))

    dl_tr = DataLoader(
        TensorDataset(torch.from_numpy(Xw_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch, shuffle=True, pin_memory=False, num_workers=0
    )
    dl_va = DataLoader(
        TensorDataset(torch.from_numpy(Xw_va), torch.from_numpy(y_va)),
        batch_size=args.batch, shuffle=False, pin_memory=False, num_workers=0
    )

    print(f"[Train] NGD={'on' if args.use_ngd else 'off'}  lr_ngd={args.lr_ngd}  Adam lr={args.lr}")

    # Warmup: optimize likelihood noise only
    if args.warmup_steps > 0:
        warm = torch.optim.Adam(lik.parameters(), lr=max(args.lr, 1e-3))
        it = iter(dl_tr)
        for t in range(args.warmup_steps):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(dl_tr)
                xb, yb = next(it)

            xb = xb.to(dev, dtype=dtype)
            yb = yb.to(dev, dtype=dtype)
            with safe_gpytorch_context(args.jitter):
                loss = -mll(model(xb), yb)

            warm.zero_grad()
            loss.backward()
            warm.step()

            clamp_kernel_raw_params(model)
            clamp_inducing_points(model)
            clamp_likelihood_noise(lik, 0, args)

        print(f"[Warmup] Finished {args.warmup_steps} steps optimizing likelihood noise.")

    best = float('inf')
    best_state = None
    best_e = 0
    t0 = time.time()

    for e in range(1, args.epochs + 1):
        ep0 = time.time()
        model.train(); lik.train()
        tr_loss = 0.0

        # NGD delay/ramp
        if args.use_ngd and len(optimizers) == 2 and e <= args.ngd_delay:
            active_opts = (optimizers[1],)  # Adam only
        else:
            active_opts = optimizers
            if args.use_ngd and len(optimizers) == 2 and args.ngd_ramp > 0:
                steps_since = max(0, e - args.ngd_delay)
                frac = min(1.0, steps_since / max(1, args.ngd_ramp))
                for g in optimizers[0].param_groups:
                    g['lr'] = args.lr_ngd * frac

        # Freeze ω-kernel params for first E epochs
        if args.freeze_kw_epochs > 0:
            set_kw_requires_grad(model, flag=(e > args.freeze_kw_epochs))

        for xb, yb in dl_tr:
            xb = xb.to(dev, dtype=dtype)
            yb = yb.to(dev, dtype=dtype)

            for opt in active_opts:
                opt.zero_grad()

            with safe_gpytorch_context(args.jitter):
                loss = -mll(model(xb), yb)

            loss.backward()

            # clip gradients (feature + likelihood), clamp others
            torch.nn.utils.clip_grad_norm_(model.feature.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(lik.parameters(), args.clip)
            for p in other_params:
                if p.grad is not None:
                    p.grad.data.clamp_(-args.clip, args.clip)

            for opt in active_opts:
                opt.step()

            clamp_kernel_raw_params(model)
            clamp_inducing_points(model)
            clamp_likelihood_noise(lik, e, args)

            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(Xw_tr)

        # Validation ELBO
        model.eval(); lik.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(dev, dtype=dtype)
                yb = yb.to(dev, dtype=dtype)
                with safe_gpytorch_context(args.jitter):
                    va_loss += (-mll(model(xb), yb)).item() * xb.size(0)
        va_loss /= len(Xw_va)

        ep_dt = time.time() - ep0
        total_dt = time.time() - t0
        print(f"[DKL-SVGP] epoch {e:4d}/{args.epochs}  train={tr_loss:.4f}  val={va_loss:.4f}  "
              f"epoch_time={ep_dt:.1f}s  elapsed={total_dt/60:.1f}m")

        # Checkpoint on improvement
        if va_loss + 1e-6 < best:
            best = va_loss
            best_e = e
            best_state = dict(model=model.state_dict(), lik=lik.state_dict())

            if args.save and args.save_each_improve:
                arch_meta = dict(
                    d_feat=args.d_feat, width=args.width,
                    nfreq=args.nfreq, nfreq_learned=args.nfreq_learned,
                    M=args.M, sm_q=args.sm_q, kx_kernel=args.kx_kernel,
                    add_kx=args.add_kx, add_kw=args.add_kw,
                    freeze_Z=args.freeze_Z
                )
                save_ckpt(args.save, model, lik, x_std, y_tf, arch_meta, idx_tr, idx_va)

            # Optional: compute val RMSE on original scale (costly but useful)
            with torch.no_grad():
                Yhat_val = predict_all(model, lik, Xva_s, Wva, y_tf, args)
            val_row_rmse = np.sqrt(np.mean((Yhat_val - Yva) ** 2, axis=1))
            print(f"[Val] mean row-RMSE (orig scale) @epoch {e}: {val_row_rmse.mean():.6f}")

        elif e - best_e >= args.patience:
            print(f"[EarlyStop] No val improvement for {args.patience} epochs; stop at {e}.")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        lik.load_state_dict(best_state["lik"])

    # Test on full test set
    Xte_s = x_std.transform(Xte_full).astype(np.float32)
    rmse_te = evaluate_rmse(model, lik, Xte_s, Yte_full, Wte, y_tf, args)
    train_seconds = time.time() - t0
    return dict(test_rmse=rmse_te, train_seconds=train_seconds, n_train=Xtr.shape[0])

def predict_mode(args):
    dev = device()
    dtype = torch.float64 if args.float64 else torch.float32
    ckpt = _robust_torch_load_v2(args.load)
    model, lik, x_std, y_tf = rebuild_from_ckpt(ckpt, dev, dtype)

    def _run_eval(path, tag, role_for_loader):
        X, Y, W = load_npz_fixed(path, role=role_for_loader)
        Xs = x_std.transform(X).astype(np.float32)
        Yhat = predict_all(model, lik, Xs, W, y_tf, args)
        outp = f"{args.out_prefix}_{tag}.npz"
        np.savez(outp, Yhat=Yhat, W=W, tag=tag)
        print(f"[Predict] Saved {tag} predictions to {outp}")
        if Y is not None and Y.shape == Yhat.shape:
            rmse = mean_row_rmse(Y, Yhat)
            print(f"[Predict] {tag} mean row-RMSE: {rmse:.6f}")

    if args.train_npz:
        _run_eval(args.train_npz, "train", "train")
    if args.val_npz:
        _run_eval(args.val_npz, "val", "val")
    if args.test_npz:
        _run_eval(args.test_npz, "test", "test")

# -------------- CLI --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'predict'], default='train')

    # Data
    ap.add_argument('--train-npz', type=str, default=None)
    ap.add_argument('--val-npz', type=str, default=None)
    ap.add_argument('--test-npz', type=str, default=None)

    # Save
    ap.add_argument('--save', type=str, default='ckpt_flex.pt')
    ap.add_argument('--save-each-improve', action='store_true')
    ap.add_argument('--load', type=str, default=None)
    ap.add_argument('--out-prefix', type=str, default='preds_dkl_svgp')

    # Subsets
    ap.add_argument('--frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=0)

    # Feature net
    ap.add_argument('--d-feat', type=int, default=64, dest='d_feat')
    ap.add_argument('--nfreq', type=int, default=16)
    ap.add_argument('--nfreq-learned', type=int, default=8, dest='nfreq_learned')
    ap.add_argument('--width', type=int, default=128)

    # Kernel / GP
    ap.add_argument('--kx-kernel', choices=['rbf','m12','m32','m52'], default='m32')
    ap.add_argument('--sm-q', type=int, default=8, dest='sm_q')
    ap.add_argument('--add-kx', action='store_true', help='Add s2*kx term')
    ap.add_argument('--add-kw', action='store_true', help='Add s3*kw term')
    ap.add_argument('--freeze-Z', action='store_true', dest='freeze_Z')
    ap.add_argument('--freeze-kw-epochs', type=int, default=0,
                    help='Freeze ω-kernel params for the first E epochs')

    # Training
    ap.add_argument('--M', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=32768)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--float64', action='store_true')
    ap.add_argument('--jitter', type=float, default=1e-2)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--warmup-steps', type=int, default=50)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--log-eps', type=float, default=1e-3)

    # NGD
    ap.add_argument('--use-ngd', action='store_true')
    ap.add_argument('--lr-ngd', type=float, default=2e-4)
    ap.add_argument('--ngd-delay', type=int, default=15)
    ap.add_argument('--ngd-ramp', type=int, default=20)

    # Noise schedule
    ap.add_argument('--noise-floor', type=float, default=1e-5)
    ap.add_argument('--noise-ceil-warm', type=float, default=0.3)
    ap.add_argument('--noise-ceil', type=float, default=0.05)
    ap.add_argument('--noise-warm-epochs', type=int, default=25)

    # Prediction chunking (used during val/test RMSE eval)
    ap.add_argument('--pred-batch', type=int, default=32768)

    args = ap.parse_args()

    if args.mode == 'train':
        for req in (args.train_npz, args.val_npz, args.test_npz):
            if req is None:
                raise ValueError("--train-npz, --val-npz, and --test-npz are required in --mode train")
        res = train_once(args)
        print(f"[Eval] Test mean row-RMSE: {res['test_rmse']:.6f} | "
              f"train_seconds={res['train_seconds']:.1f} | n_train={res['n_train']}")
    else:
        if not args.load:
            raise ValueError("--load checkpoint is required in --mode predict")
        if not any([args.train_npz, args.val_npz, args.test_npz]):
            raise ValueError("Provide at least one split path for --mode predict")
        predict_mode(args)

if __name__ == '__main__':
    main()