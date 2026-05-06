import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["default", "science", "no-latex"])

# ---------- metrics ----------
def row_rmse(y_true, y_pred):
    # per-sample RMSE across the spectrum axis
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))

def row_nrmse(y_true, y_pred, eps=1e-8):
    num = np.sqrt(np.mean((y_true - y_pred)**2, axis=1))
    denom = np.sqrt(np.mean(y_true**2, axis=1)) + eps
    return num / denom


def summarize_split(name, y_true, preds_dict):
    """
    preds_dict: dict like {"FFNN": A, "GP": B, "FFNN subset": C}
    """
    print(f"=== [{name}] ===")
    rmses = {}
    for tag, pred in preds_dict.items():
        rm = row_rmse(y_true, pred)
        rmses[tag] = rm
        print(f"  {tag:<12s} mean row-RMSE: {rm.mean():.6f}  "
              f"(std {rm.std():.6f}, median {np.median(rm):.6f})")

    # Pairwise deltas relative to GP if present
    if "GP" in rmses:
        base = rmses["GP"]
        for tag, rm in rmses.items():
            if tag == "GP":
                continue
            delta = rm - base
            print(f"  Δ({tag}−GP) mean: {delta.mean():.6f}  "
                  f"(std {delta.std():.6f}, {tag} worse %: {(delta>0).mean():.2%})")
            inter = row_rmse(preds_dict[tag], preds_dict["GP"])
            print(f"  inter-model row-RMSE ({tag} vs GP): mean {inter.mean():.6f} (std {inter.std():.6f})")
    print()


# ---- styling ----
SIZE = 18
FONTSIZE = 18
EDGECOLOR = "gray"

mpl.rcParams.update({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE*0.9,
    "ytick.labelsize": FONTSIZE*0.9,
})

# ---- tiny utils ----
def _safe_W(npz_obj, P, default=(-6.0, 6.0)):
    # Prefer 'W' or 'omega' inside the split NPZ; otherwise fall back.
    if npz_obj is not None:
        try:
            if "W" in npz_obj.files:     return npz_obj["W"].squeeze()
            if "omega" in npz_obj.files: return npz_obj["omega"].squeeze()
        except Exception:
            pass
    return np.linspace(default[0], default[1], P)

def _split_registry(Xtr, Ytr, A_tr, B_tr, C_tr, tr,
                    Xva, Yva, A_va, B_va, C_va, va,
                    Xte, Yte, A_te, B_te, C_te, te):
    return {
        "train":    {"X": Xtr,   "Y": Ytr,   "A": A_tr, "B": B_tr, "C": C_tr, "npz": tr, "label": "TRAIN"},
        "val":      {"X": Xva,   "Y": Yva,   "A": A_va, "B": B_va, "C": C_va, "npz": va, "label": "VAL"},
        "test":     {"X": Xte,   "Y": Yte,   "A": A_te, "B": B_te, "C": C_te, "npz": te, "label": "TEST"},
    }

def _peak_height_and_loc(Y, W):
    idx = np.argmax(Y, axis=1)
    heights = Y[np.arange(Y.shape[0]), idx]
    locs = W[idx]
    return heights, locs


# ======================= 1) Worst-tail 2×3 grid (FFNN-scored) =======================
def make_worst_tail_percentile_grid(reg, split="test",
                                    percentiles=(0, 2, 4, 6, 8, 10),
                                    out_prefix=None,
                                    show_titles=True):
    """
    2×3 grid of spectra at {0,2,4,6,8,10}% into the WORST tail,
    where "worst" is defined by FFNN row-RMSE on the chosen split.

    Each panel shows: Ground truth, FFNN, GP, FFNN subset.
    """
    assert split in reg, f"Unknown split: {split}"

    Y = reg[split]["Y"]
    X = reg[split]["X"]
    A = reg[split]["A"]  # FFNN (full-data)
    B = reg[split]["B"]  # GP
    C = reg[split]["C"]  # FFNN subset
    npz_obj = reg[split]["npz"]

    P = Y.shape[1]
    W = _safe_W(npz_obj, P=P)

    # Score rows by FFNN error ONLY
    scores = row_rmse(Y, A)  # higher = worse

    # Sort descending so index 0 is worst row
    order = np.argsort(scores)[::-1]
    n = len(scores)

    # Pick rows at p% into the worst tail
    idx = []
    for p in percentiles:
        k = int(np.floor((p / 100.0) * (n - 1)))
        idx.append(order[k])

    if out_prefix is None:
        out_prefix = f"worst_tail_percentiles_{split}"

    fig, axes = plt.subplots(2, 3, figsize=(SIZE, SIZE * 0.54), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    axes = axes.ravel()
    handles = None

    text_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"] 

    for j, (ax, i) in enumerate(zip(axes, idx)):
        t1, t2, J = X[i]  # (t', t'', J)

        ln_true, = ax.plot(W, Y[i], lw=2.5, color="darkgreen", ls="--", label="Ground truth", zorder=1)
        ln_ffnn, = ax.plot(W, A[i], lw=1.8, color="royalblue", ls="-", label="FFNN (full)", zorder=2)
        ln_sub,  = ax.plot(W, C[i], lw=1.2, color="darkorange", ls="-.", label="FFNN Subset", zorder=3)
        ln_gp,  = ax.plot(W, B[i], lw=1.2, color="deeppink", ls="-", label="DKL-SVGP", zorder=4)
        ax.set_box_aspect(0.7)

        if handles is None:
            handles = [ln_true, ln_ffnn, ln_sub, ln_gp]

        if show_titles:
            ax.set_title(f"Percentile {percentiles[j]}")

        ax.text(
            0.97, 0.96,
            rf"$t^{{\prime}}={t1:.3f}$" "\n" rf"$t^{{\prime\prime}}={t2:.3f}$" "\n" rf"$J=$ ${J:.3f}$",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=FONTSIZE*0.9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="lightgray", alpha=0.9)
        )

        ax.grid(True, alpha=0.25)

        if j ==0 or j == 3:
            ax.text(-0.15, 1.05, text_labels[j], transform=ax.transAxes, ha="left", va="top", fontsize=FONTSIZE)
        else:
            ax.text(-0.07, 1.05, text_labels[j], transform=ax.transAxes, ha="left", va="top", fontsize=FONTSIZE)

    for ax in axes[3:]:
        ax.set_xlabel(r"$\omega$")
    for ax in axes[::3]:
        ax.set_ylabel(r"$A(\omega)$")

    leg = fig.legend(
        handles, ["Ground truth", "FFNN (full)", "FFNN Subset", "DKL-SVGP"],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.06),
        frameon=True,
        fancybox=True
    )

    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("lightgray")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(1.0)

    fig.tight_layout(rect=[0, 0, 1, 1])

    pdf = f"./outputs/{out_prefix}.pdf"
    # fig.savefig(pdf, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg,))
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.2, bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"[plot] Saved {pdf}")

# ======================= 2) Peak height/location scatter (3 models) =======================
def make_peak_scatter(reg, split="test", out_prefix=None):
    """
    Make a 2x3 figure of predicted peak height vs GT (top)
    and peak ω location vs GT (bottom): FFNN, GP, FFNN subset.
    """
    assert split in reg, f"Unknown split: {split}"

    Y = reg[split]["Y"]
    A = reg[split]["A"]
    B = reg[split]["B"]
    C = reg[split]["C"]
    npz_obj = reg[split]["npz"]
    W = _safe_W(npz_obj, P=Y.shape[1])

    if out_prefix is None:
        out_prefix = f"peaks_scatter_{split}"

    h_true, w_true = _peak_height_and_loc(Y, W)
    h_ffnn, w_ffnn = _peak_height_and_loc(A, W)
    h_gp,  w_gp  = _peak_height_and_loc(B, W)
    h_sub,  w_sub  = _peak_height_and_loc(C, W)

    def _scatter_ax(ax, x, y, xlabel, ylabel, dotcolor=None, xlim=None, xticks=None):
        ax.scatter(x, y, s=12, alpha=0.6, color=dotcolor, edgecolors=EDGECOLOR, linewidths=0.3)
        ax.set_box_aspect(1)
        lo = np.min([x.min(), y.min()])-0.14
        hi = np.max([x.max(), y.max()])+0.14
        if xlim is not None:
            lo, hi = xlim
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_yticks(xticks)
        ax.plot([lo, hi], [lo, hi], lw=1.0, color="black", alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(xlabel, fontsize=FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
        ax.grid(True, alpha=0.25)
        rmse = float(np.sqrt(np.mean((y - x)**2)))
        mae  = float(np.mean(np.abs(y - x)))
        r    = float(np.corrcoef(x, y)[0,1]) if x.size > 1 else np.nan
        ax.text(0.02, 0.98, f"RMSE={rmse:.4f}\n  MAE={mae:.4f}\n   Corr={r:.4f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=FONTSIZE*0.9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="lightgray", alpha=0.9))

    fig, axs = plt.subplots(2, 3, figsize=(SIZE, SIZE * 0.75))
    plt.subplots_adjust(hspace=0.5, wspace=1.5)

    # consistent ω limits (tweak if you want)
    wmin, wmax = -4.1, -0.2
    wticks = [-3.6, -3.0, -2.4, -1.8, -1.2, -0.6]

    # Column 1: FFNN
    _scatter_ax(axs[0,0], h_true, h_ffnn, dotcolor="royalblue",
                xlabel="Ground truth peak height", ylabel="FFNN (full) peak height")
    axs[0,0].text(-0.2, 1.05, "(a.i)", transform=axs[0,0].transAxes, ha="left", va="top", fontsize=FONTSIZE)
    _scatter_ax(axs[1,0], w_true, w_ffnn, dotcolor="royalblue",
                xlabel=r"Ground truth peak $\omega$", ylabel=r"FFNN (full) peak $\omega$", xlim=(wmin, wmax), xticks=wticks)
    axs[1,0].text(-0.18, 1.05, "(a.ii)", transform=axs[1,0].transAxes, ha="left", va="top", fontsize=FONTSIZE)

    # Column 2: FFNN subset
    _scatter_ax(axs[0,1], h_true, h_sub, dotcolor="darkorange",
                xlabel="Ground truth peak height", ylabel="FFNN Subset peak height")
    axs[0,1].text(-0.2, 1.05, "(b.i)", transform=axs[0,1].transAxes, ha="left", va="top", fontsize=FONTSIZE)
    _scatter_ax(axs[1,1], w_true, w_sub, dotcolor="darkorange",
                xlabel=r"Ground truth peak $\omega$", ylabel=r"FFNN Subset peak $\omega$", xlim=(wmin, wmax), xticks=wticks)
    axs[1,1].text(-0.18, 1.05, "(b.ii)", transform=axs[1,1].transAxes, ha="left", va="top", fontsize=FONTSIZE)

    # Column 3: GP
    _scatter_ax(axs[0,2], h_true, h_gp, dotcolor="deeppink",
                xlabel="Ground truth peak height", ylabel="DKL-SVGP peak height")
    axs[0,2].text(-0.2, 1.05, "(c.i)", transform=axs[0,2].transAxes, ha="left", va="top", fontsize=FONTSIZE)
    _scatter_ax(axs[1,2], w_true, w_gp, dotcolor="deeppink",
                xlabel=r"Ground truth peak $\omega$", ylabel=r"DKL-SVGP peak $\omega$", xlim=(-4.5, 3.8), xticks=[-3.6, -2.4, -1.2, -0.0, 1.2, 2.4])
    axs[1,2].text(-0.18, 1.05, "(c.ii)", transform=axs[1,2].transAxes, ha="left", va="top", fontsize=FONTSIZE)

    

    plt.tight_layout(rect=[0, 0, 1, 1])
    pdf = f"./outputs/{out_prefix}.pdf"
    plt.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[plot] Saved {pdf}")

def make_peak_scatter_topk(reg, split="test", out_prefix=None, track_peak=True, topk=3):
    """
    Make a 2x3 figure of predicted peak height vs GT (top)
    and peak ω location vs GT (bottom): FFNN, GP, FFNN subset.

    If track_peak=True: for each model, instead of using the model's *dominant* peak,
    we pick the predicted peak (among top-k local maxima) whose location is closest
    to the GT dominant-peak location ω_true. This isolates "peak order swaps".
    """
    assert split in reg, f"Unknown split: {split}"

    Y = reg[split]["Y"]
    A = reg[split]["A"]
    B = reg[split]["B"]
    C = reg[split]["C"]
    npz_obj = reg[split]["npz"]
    W = _safe_W(npz_obj, P=Y.shape[1])

    if out_prefix is None:
        out_prefix = f"peaks_scatter_{split}" + ("_tracked" if track_peak else "")

    # ---------- helper: dominant peak (truth) ----------
    def _peak_height_and_loc(Yarr, W):
        idx = np.argmax(Yarr, axis=1)
        heights = Yarr[np.arange(Yarr.shape[0]), idx]
        locs = W[idx]
        return heights, locs, idx

    # ---------- helper: choose predicted peak closest to true ω among top-k local maxima ----------
    def _tracked_peak(Ypred, W, w_true, topk=3):
        """
        Returns (h_match, w_match) arrays of shape (N,).
        If no local maxima exist for a spectrum, falls back to global argmax.
        """
        N, P = Ypred.shape
        h_out = np.empty(N, dtype=float)
        w_out = np.empty(N, dtype=float)

        for i in range(N):
            y = Ypred[i]
            # local maxima indices (exclude endpoints)
            locmax = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

            if locmax.size == 0:
                j = int(np.argmax(y))
                h_out[i] = y[j]
                w_out[i] = W[j]
                continue

            # take top-k by height
            if locmax.size > topk:
                top_idx = locmax[np.argsort(y[locmax])[-topk:]]
            else:
                top_idx = locmax

            # among those, pick closest ω to truth ω_true
            j = int(top_idx[np.argmin(np.abs(W[top_idx] - w_true[i]))])
            h_out[i] = y[j]
            w_out[i] = W[j]

        return h_out, w_out

    # ---------- truth dominant peak defines the reference ----------
    h_true, w_true, _ = _peak_height_and_loc(Y, W)

    # baseline (unchanged): model dominant peaks
    h_ffnn, w_ffnn, _ = _peak_height_and_loc(A, W)
    h_gp,   w_gp,   _ = _peak_height_and_loc(B, W)
    h_sub,  w_sub,  _ = _peak_height_and_loc(C, W)

    # tracking mode: replace model peaks with matched peaks
    if track_peak:
        h_ffnn, w_ffnn = _tracked_peak(A, W, w_true, topk=topk)
        h_gp,   w_gp   = _tracked_peak(B, W, w_true, topk=topk)
        h_sub,  w_sub  = _tracked_peak(C, W, w_true, topk=topk)

    # ---------- plotting (your formatting unchanged) ----------
    def _scatter_ax(ax, x, y, xlabel, ylabel, dotcolor=None, xlim=None, xticks=None):
        ax.scatter(x, y, s=12, alpha=0.6, color=dotcolor, edgecolors=EDGECOLOR, linewidths=0.3)
        ax.set_box_aspect(1)
        lo = np.min([x.min(), y.min()]) - 0.14
        hi = np.max([x.max(), y.max()]) + 0.14
        if xlim is not None:
            lo, hi = xlim
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_yticks(xticks)
        ax.plot([lo, hi], [lo, hi], lw=1.0, color="black", alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(xlabel, fontsize=FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
        ax.grid(True, alpha=0.25)
        rmse = float(np.sqrt(np.mean((y - x)**2)))
        mae  = float(np.mean(np.abs(y - x)))
        r    = float(np.corrcoef(x, y)[0,1]) if x.size > 1 else np.nan
        ax.text(0.02, 0.98, f"RMSE={rmse:.4f}\n  MAE={mae:.4f}\n   Corr={r:.4f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=FONTSIZE*0.9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="lightgray", alpha=0.9))

    fig, axs = plt.subplots(2, 3, figsize=(SIZE, SIZE * 0.75))
    plt.subplots_adjust(hspace=0.5, wspace=1.5)

    wmin, wmax = -4.1, -0.2
    wticks = [-3.6, -3.0, -2.4, -1.8, -1.2, -0.6]

    # Column 1: FFNN
    _scatter_ax(axs[0,0], h_true, h_ffnn, dotcolor="royalblue",
                xlabel="Ground truth peak height", ylabel="FFNN (full) peak height")
    axs[0,0].text(-0.2, 1.05, "(a.i)", transform=axs[0,0].transAxes, ha="left", va="top", fontsize=FONTSIZE)
    _scatter_ax(axs[1,0], w_true, w_ffnn, dotcolor="royalblue",
                xlabel=r"Ground truth peak $\omega$", ylabel=r"FFNN (full) peak $\omega$", xlim=(wmin, wmax), xticks=wticks)
    axs[1,0].text(-0.18, 1.05, "(a.ii)", transform=axs[1,0].transAxes, ha="left", va="top", fontsize=FONTSIZE)

    # Column 2: FFNN subset
    _scatter_ax(axs[0,1], h_true, h_sub, dotcolor="darkorange",
                xlabel="Ground truth peak height", ylabel="FFNN Subset peak height")
    axs[0,1].text(-0.2, 1.05, "(b.i)", transform=axs[0,1].transAxes, ha="left", va="top", fontsize=FONTSIZE)
    _scatter_ax(axs[1,1], w_true, w_sub, dotcolor="darkorange",
                xlabel=r"Ground truth peak $\omega$", ylabel=r"FFNN Subset peak $\omega$", xlim=(wmin, wmax), xticks=wticks)
    axs[1,1].text(-0.18, 1.05, "(b.ii)", transform=axs[1,1].transAxes, ha="left", va="top", fontsize=FONTSIZE)

    # Column 3: GP
    _scatter_ax(axs[0,2], h_true, h_gp, dotcolor="deeppink",
                xlabel="Ground truth peak height", ylabel="DKL-SVGP peak height")
    axs[0,2].text(-0.2, 1.05, "(c.i)", transform=axs[0,2].transAxes, ha="left", va="top", fontsize=FONTSIZE)
    _scatter_ax(axs[1,2], w_true, w_gp, dotcolor="deeppink",
                xlabel=r"Ground truth peak $\omega$", ylabel=r"DKL-SVGP peak $\omega$",
                xlim=(wmin, wmax), xticks=wticks,
                # xlim=(-4.5, 3.8), xticks=[-3.6, -2.4, -1.2, -0.0, 1.2, 2.4]
                )
    axs[1,2].text(-0.18, 1.05, "(c.ii)", transform=axs[1,2].transAxes, ha="left", va="top", fontsize=FONTSIZE)

    plt.tight_layout(rect=[0, 0, 1, 1])
    pdf = f"./outputs/{out_prefix}.pdf"
    plt.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[plot] Saved {pdf}  (track_peak={track_peak}, topk={topk})")

def summarize_peakloc_outliers(reg, split="test", thresholds=(0.1, 0.2, 0.3, 0.5)):
    Y = reg[split]["Y"]
    B = reg[split]["B"]
    npz_obj = reg[split]["npz"]
    W = _safe_W(npz_obj, P=Y.shape[1])

    wt = W[np.argmax(Y, axis=1)]
    wg = W[np.argmax(B, axis=1)]
    dW = np.abs(wg - wt)

    print(f"[{split}] |Δω_max| summary (truth vs GP):")
    qs = [50, 90, 95, 99, 99.5, 99.9]
    print("  quantiles:", ", ".join([f"p{q}={np.percentile(dW,q):.4f}" for q in qs]))
    print(f"  max: {dW.max():.4f}   mean: {dW.mean():.4f}   median: {np.median(dW):.4f}")

    N = len(dW)
    for t in thresholds:
        c = int((dW >= t).sum())
        print(f"  count(|Δω| ≥ {t:g}) = {c}/{N} ({100*c/N:.2f}%)")

    return dW

def inspect_peakloc_outliers(reg, split="test", k=15):
    Y = reg[split]["Y"]; B = reg[split]["B"]
    X = reg[split]["X"]; npz_obj = reg[split]["npz"]
    W = _safe_W(npz_obj, P=Y.shape[1])

    it = np.argmax(Y, axis=1)
    ig = np.argmax(B, axis=1)
    dW = np.abs(W[ig] - W[it])

    top = np.argsort(dW)[::-1][:k]
    print("top outliers:")
    for idx in top:
        edge_t = (it[idx] in [0, len(W)-1])
        edge_g = (ig[idx] in [0, len(W)-1])
        print(f"  i={idx:5d}  |Δω|={dW[idx]:.4f}  it={it[idx]:3d} ig={ig[idx]:3d}  edge(t,g)=({edge_t},{edge_g})"
              f"  x={X[idx]}")
    return top

def plot_peakloc_outlier_spectra_5x3_truth_vs_gp(
    reg,
    split="test",
    outliers=None,                 # list of dicts: {"i":int, "dW":float, "x":[t',t'',J]}
    out_prefix="peakloc_outliers_truth_vs_gp_5x3",
    show_titles=False,
    ):
    """
    5x3 grid: 15 slots; fill first 13 in row-major order; turn off last row col2-3.
    Plots Ground truth vs DKL-SVGP over full spectrum.
    """
    assert split in reg, f"Unknown split: {split}"
    assert outliers is not None and len(outliers) == 13, "Provide exactly 13 outliers."

    Y = reg[split]["Y"]
    X = reg[split]["X"]
    B = reg[split]["B"]  # GP prediction
    npz_obj = reg[split]["npz"]

    P = Y.shape[1]
    W = _safe_W(npz_obj, P=P)

    fig, axes = plt.subplots(5, 3, figsize=(SIZE*1.25, SIZE*1.35), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.55, wspace=0.45)
    axes = axes.ravel()

    # panel labels (a) ... (m)
    text_labels = [f"({chr(ord('a')+k)})" for k in range(13)]

    # turn off last row col2-3 => indices (4,1) and (4,2) in 5x3
    axes[12].axis("off")
    axes[14].axis("off")

    handles = None

    for j in list(range(12)) + [13]:
        ax = axes[j]
        out = outliers[j] if j < 13 else outliers[j-1]
        i = int(out["i"])
        dW = float(out["dW"])
        x = np.asarray(out.get("x", X[i]), dtype=float).ravel()

        y_true = Y[i]
        y_gp   = B[i]

        wpk_true = W[np.argmax(y_true)]
        wpk_gp   = W[np.argmax(y_gp)]

        ln_true, = ax.plot(W, y_true, lw=1.8, color="darkgreen", ls="--", label="Ground truth", zorder=2)
        ln_gp,   = ax.plot(W, y_gp,   lw=1.4, color="deeppink",  ls="-",  label="DKL-SVGP",   zorder=3)

        ax.axvline(wpk_true, color="darkgreen", ls="--", lw=0.6, alpha=0.7, zorder=1)
        ax.axvline(wpk_gp,   color="deeppink",  ls="--", lw=0.6, alpha=0.7, zorder=1)

        ax.set_box_aspect(0.6)
        ax.grid(True, alpha=0.25)

        if handles is None:
            handles = [ln_true, ln_gp]

        # (a),(b),... label placement similar to your reference
        # left column gets a bit more negative x offset
        col = j % 3
        xoff = -0.12 if col == 0 else -0.08
        text_label = text_labels[j] if j < 13 else text_labels[j-1]
        ax.text(xoff, 1.05, text_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=FONTSIZE)

        # annotation box: ONLY |Δω|, t', t'', J (each on new line)
        ax.text(
            0.97, 0.96,
            rf"$|\Delta\omega|={dW:.3f}$" "\n"
            rf"$t^{{\prime}}={x[0]:.3f}$" "\n"
            rf"$t^{{\prime\prime}}={x[1]:.3f}$" "\n"
            rf"$J={x[2]:.3f}$",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=FONTSIZE*0.85,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="lightgray", alpha=0.9)
        )

        if show_titles:
            ax.set_title(f"outlier {j+1}", fontsize=FONTSIZE)

    # axis labels: bottom row that still has plots is row 5 col1 only (index 12)
    # but safest: label all axes in last row that are not off
    for ax in axes:
        if ax.has_data():
            ax.set_xlabel(r"$\omega$")
            ax.set_ylabel(r"$A(\omega)$")

    # legend (2 entries only)
    leg = fig.legend(
        handles, ["Ground truth", "DKL-SVGP"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.05),
        frameon=True,
        fancybox=True
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("lightgray")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(1.0)

    # give extra room for panel labels + legend
    fig.tight_layout(rect=[0, 0.05, 1, 0.99])

    pdf = f"./outputs/{out_prefix}.pdf"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.3, bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"[plot] Saved {pdf}")

# ======================= MAIN =======================

# ---------- load ground truth ----------
tr = np.load("./data/train-set.npz"); Xtr, Ytr = tr["params"], tr["dos"]
va = np.load("./data/val-set.npz");   Xva, Yva = va["params"], va["dos"]
te = np.load("./data/test-set.npz");  Xte, Yte = te["params"], te["dos"]

# ---------- load model predictions ----------
# Model A (FFNN, full data)
A_tr = np.load("./outputs/preds_ffnn_full_train.npz")["Yhat"]
A_va = np.load("./outputs/preds_ffnn_full_val.npz")["Yhat"]
A_te = np.load("./outputs/preds_ffnn_full_test.npz")["Yhat"]

# Model B (GP)
# GP_model = "preds_safeA8_1"
B_tr = np.load("./outputs/preds_dkl_svgp_train.npz")["Yhat"]
B_va = np.load("./outputs/preds_dkl_svgp_val.npz")["Yhat"]
B_te = np.load("./outputs/preds_dkl_svgp_test.npz")["Yhat"]

# Model C (FFNN trained on subset)
C_tr = np.load("./outputs/preds_ffnn_subset_train.npz")["Yhat"]
C_va = np.load("./outputs/preds_ffnn_subset_val.npz")["Yhat"]
C_te = np.load("./outputs/preds_ffnn_subset_test.npz")["Yhat"]

# ---------- registry for plotting ----------
reg = _split_registry(
    Xtr, Ytr, A_tr, B_tr, C_tr, tr,
    Xva, Yva, A_va, B_va, C_va, va,
    Xte, Yte, A_te, B_te, C_te, te
)

# ---------- summarize ----------
summarize_split("train",    Ytr,   {"FFNN": A_tr, "GP": B_tr, "FFNN subset": C_tr})
summarize_split("val",      Yva,   {"FFNN": A_va, "GP": B_va, "FFNN subset": C_va})
summarize_split("test",     Yte,   {"FFNN": A_te, "GP": B_te, "FFNN subset": C_te})

# ---------- mean NRMSE ----------
for name, (Y, A, B, C) in {
    "train": (Ytr, A_tr, B_tr, C_tr),
    "val": (Yva, A_va, B_va, C_va),
    "test": (Yte, A_te, B_te, C_te),
}.items():
    print(f"[{name}] mean NRMSE  "
          f"FFNN={row_nrmse(Y, A).mean():.6f}  "
          f"GP={row_nrmse(Y, B).mean():.6f}  "
          f"FFNN subset={row_nrmse(Y, C).mean():.6f}")
    print(f"[{name}] median NRMSE  "
            f"FFNN={np.median(row_nrmse(Y, A)):.6f}  "
            f"GP={np.median(row_nrmse(Y, B)):.6f}  "
            f"FFNN subset={np.median(row_nrmse(Y, C)):.6f}")
    print(f"[{name}] std NRMSE  "
            f"FFNN={row_nrmse(Y, A).std():.6f}  "
            f"GP={row_nrmse(Y, B).std():.6f}  "
            f"FFNN subset={row_nrmse(Y, C).std():.6f}")
print()


# ---------- plots ----------
make_worst_tail_percentile_grid(reg, split="test",
                                out_prefix=f"worst_tail_percentiles_test_dkl_svgp")

make_peak_scatter(reg, split="test", out_prefix=f"peaks_scatter_test_dkl_svgp")

make_peak_scatter_topk(reg, split="test", out_prefix="peaks_scatter_test_tracked_top5", track_peak=True, topk=5)

summarize_peakloc_outliers(reg, split="test", thresholds=(0.05,0.1,0.2,0.3,0.5))

inspect_peakloc_outliers(reg, "test", k=15)

outliers = [
  {"i": 3415, "dW": 6.0400, "x": [-0.48, -0.48, 0.2]},
  {"i": 206,  "dW": 5.9600, "x": [-0.5,  -0.48, 0.264]},
  {"i": 9885, "dW": 5.9600, "x": [-0.48, -0.5,  0.264]},
  {"i": 9046, "dW": 5.9600, "x": [-0.5,  -0.48, 0.248]},
  {"i": 13190,"dW": 1.1200, "x": [-0.38, 0.5,  0.2]},
  {"i": 6159, "dW": 1.1200, "x": [-0.38, 0.48, 0.216]},
  {"i": 12852,"dW": 1.1200, "x": [-0.48, 0.5,  0.232]},
  {"i": 1345, "dW": 1.0800, "x": [-0.4,  0.48, 0.2]},
  {"i": 1721, "dW": 1.0800, "x": [-0.28, 0.5,  0.2]},
  {"i": 5331, "dW": 1.0000, "x": [0.0,   0.5,  0.2]},
  {"i": 8631, "dW": 1.0000, "x": [-0.02, 0.5,  0.2]},
  {"i": 7355, "dW": 0.9600, "x": [0.06,  0.5,  0.2]},
  {"i": 8856, "dW": 0.9600, "x": [0.08,  0.5,  0.2]},
]

plot_peakloc_outlier_spectra_5x3_truth_vs_gp(
    reg,
    split="test",
    outliers=outliers,
    out_prefix="peakloc_outliers_truth_vs_gp_test",
)