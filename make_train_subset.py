#!/usr/bin/env python3
import argparse
import numpy as np

def select_fixed_fraction(N, frac, seed):
    rng = np.random.RandomState(seed)
    k = max(1, int(round(frac * N)))
    idx = np.sort(rng.choice(N, size=k, replace=False))
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", default="train-set.npz",
                    help="Path to the full train npz with keys 'params','dos' (optional 'W')")
    ap.add_argument("--frac", type=float, default=0.10, help="Fraction of rows to keep (e.g., 0.10)")
    ap.add_argument("--seed", type=int, default=0, help="Subset seed (must match your DKL runs)")
    ap.add_argument("--out", default="train-subset-0.1.npz", help="Output subset file")
    args = ap.parse_args()

    z = np.load(args.train_npz, allow_pickle=True)
    if not {"params", "dos"}.issubset(set(z.files)):
        raise KeyError(f"{args.train_npz} must contain 'params' and 'dos'")

    X = z["params"]
    Y = z["dos"]

    idx = select_fixed_fraction(X.shape[0], args.frac, args.seed)
    Xs, Ys = X[idx], Y[idx]

    save = {"params": Xs, "dos": Ys, "idx": idx}
    if "W" in z.files:  # keep W if present
        save["W"] = z["W"]

    np.savez(args.out, **save)
    print(f"[Saved] {args.out}  params{Xs.shape}  dos{Ys.shape}  (idx {idx.shape}, seed={args.seed}, frac={args.frac})")

if __name__ == "__main__":
    main()