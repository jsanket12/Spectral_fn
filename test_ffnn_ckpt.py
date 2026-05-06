#!/usr/bin/env python3
import os, argparse
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from sklearn.preprocessing import StandardScaler

def device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ---------- data I/O ----------
def load_npz_fixed(path, role="train", wmin=-6.0, wmax=6.0):
    z = np.load(path, allow_pickle=True)
    files = set(z.files)
    if not {'params','dos'}.issubset(files):
        raise KeyError(f"[{role}] needs 'params' & 'dos'. Found: {list(z.files)}")
    X = z['params'].astype(np.float32)
    Y = z['dos'].astype(np.float32)
    if 'W' in z.files:
        W = z['W'].astype(np.float32).reshape(-1); src = "file:'W'"
    else:
        P = Y.shape[1]
        W = np.linspace(wmin, wmax, P, dtype=np.float32); src = f"synth:[{wmin},{wmax}] P={P}"
    print(f"[Load-{role}] {os.path.basename(path)}  X=params{X.shape}  Y=dos{Y.shape}  W=<{src}>")
    return X, Y, W

# ---------- model ----------
class LitFFNN(pl.LightningModule):
    def __init__(self, layer_sizes, lr=0.01, lr_factor=0.0):
        super().__init__()
        mods = []
        for i in range(len(layer_sizes)-1):
            mods.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i != len(layer_sizes)-2:
                mods.append(nn.ReLU())
        self.forward_prop = nn.Sequential(*mods)
        self.learning_rate = lr
        self.factor = lr_factor
        self.save_hyperparameters()

    def forward(self, x):
        return self.forward_prop(x)

    # not used at inference
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def get_layers_from_hparams(hparams):
    # works with both dict-like and Namespace hparams
    if hasattr(hparams, "layer_sizes"):
        return list(hparams.layer_sizes)
    if isinstance(hparams, dict) and "layer_sizes" in hparams:
        return list(hparams["layer_sizes"])
    # fallback: infer from state_dict shapes
    return None

# ---------- metrics ----------
def rmse_global(Yhat, Y):
    return float(np.sqrt(np.mean((Yhat - Y) ** 2)))

def rmse_per_row_mean(Yhat, Y):
    per = np.sqrt(np.mean((Yhat - Y) ** 2, axis=1))  # (N,)
    return float(per.mean()), per

# ---------- eval helper ----------
@torch.no_grad()
def run_split(name, X, Y, scaler_mode, train_scaler, model, dev, dtype):
    # X standardization mode
    if scaler_mode == "train":
        assert train_scaler is not None, "train scaler is None in train mode"
        Xs = train_scaler.transform(X)
    elif scaler_mode == "self":
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
    else:
        Xs = X

    xt = torch.from_numpy(Xs).to(dev, dtype=dtype)
    Yhat = model(xt).cpu().numpy().astype(np.float32)

    g = rmse_global(Yhat, Y)
    m, per = rmse_per_row_mean(Yhat, Y)
    print(f"[Predict] {name:>7s}  global_RMSE={g:.6f}  mean_perrow_RMSE={m:.6f}")
    return Yhat, g, m, per

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--train-npz", required=True, type=str)
    ap.add_argument("--val-npz", required=True, type=str)
    ap.add_argument("--test-npz", required=True, type=str)
    ap.add_argument("--out-prefix", type=str, default="preds_ffnn")
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--x-scaler", choices=["self","train","none"], default="train",
                   help=(
                        "How to standardize X. "
                        "'train' = fit on TRAIN then transform all splits (recommended, no leakage). "
                        "'self'  = fit/transform each split independently (leaks val/test stats). "
                        "'none'  = no scaling."
                    )
    )
    args = ap.parse_args()

    # load splits
    Xtr, Ytr, Wtr = load_npz_fixed(args.train_npz, "train")
    Xva, Yva, Wva = load_npz_fixed(args.val_npz, "val")
    Xte, Yte, Wte = load_npz_fixed(args.test_npz, "test")

    # model
    dev = device()
    dtype = torch.float64 if args.float64 else torch.float32
    model = LitFFNN.load_from_checkpoint(args.ckpt, map_location=dev)
    model.eval().to(dev).to(dtype)

    # log some info
    layers = get_layers_from_hparams(model.hparams)
    print(f"[Model] {os.path.basename(args.ckpt)}  device={dev.type}  "
          f"layers={layers if layers is not None else 'unknown'}  activation=relu")

    # prepare scaler (if 'train' mode)
    train_scaler = None
    if args.x_scaler == "train":
        # ensure we fit on train only
        train_scaler = StandardScaler().fit(Xtr)
        print("[Scaler] Fitted on TRAIN X and will transform {train,val,test}")
    elif args.x_scaler == "self":
        print("[Warn] Using 'self' scaling: each split is scaled independently (can inflate val/test).")
    else:
        print("[Scaler] No scaling ('none').")

    # eval
    Ytr_hat, gtr, mtr, _ = run_split("train", Xtr, Ytr, args.x_scaler, train_scaler, model, dev, dtype)
    Yva_hat, gva, mva, _ = run_split("val",   Xva, Yva, args.x_scaler, train_scaler, model, dev, dtype)
    Yte_hat, gte, mte, _ = run_split("test",  Xte, Yte, args.x_scaler, train_scaler, model, dev, dtype)

    np.savez(f"{args.out_prefix}_train.npz", Yhat=Ytr_hat, W=Wtr)
    np.savez(f"{args.out_prefix}_val.npz",   Yhat=Yva_hat, W=Wva)
    np.savez(f"{args.out_prefix}_test.npz",  Yhat=Yte_hat, W=Wte)
    print("[Save] wrote", f"{args.out_prefix}_{{train,val,test}}.npz")

if __name__ == "__main__":
    main()