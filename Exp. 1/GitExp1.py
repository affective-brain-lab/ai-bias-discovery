# =============================================================================
# Information Valuation under Complexity - Experiment 1 Analysis
# =============================================================================
#
# Description
# -----------
# Generates three behavioral figures from Cohort B trial-level data:
#     Figure 1 - Bid distribution conditional on whether information was
#                received (infoValue >= randNum) vs. not received.
#     Figure 2 - Distribution of valuation bias (subjective bid -
#                normative information value), with reference lines at
#                zero and at the sample mean.
#     Figure 3 - Probability of playing the lottery as a function of
#                expected value, fit with a pooled logistic regression
#                and a 95% bootstrap confidence interval (subject-wise
#                if a subject ID column is present, otherwise trial-wise).
#
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# Config & helpers
# -----------------------------------------------------------------------------
cfg = {
    "outdir": ".",
    "export": {"svg": True, "pdf": True, "png": True, "dpi": 300},
}


def save_figure(fig, base, cfg):
    """Save figure in formats specified by cfg['export'] with white background."""
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    exp = cfg.get("export", {})
    if exp.get("svg", True):
        fig.savefig(f"{base}.svg", bbox_inches="tight", facecolor="white")
    if exp.get("pdf", True):
        fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    if exp.get("png", True):
        fig.savefig(f"{base}.png", bbox_inches="tight",
                    facecolor="white", dpi=exp.get("dpi", 300))


# Shared color
COLOR_DARK = "#2C3E50"

# -----------------------------------------------------------------------------
# Load & prep
# -----------------------------------------------------------------------------
data = pd.read_csv("dataExp1.csv")

for col in ["infoValue", "infoNorm", "randNum"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data["received_info"] = data["infoValue"] >= data["randNum"]


# =============================================================================
# FIGURE 1: BID DISTRIBUTION BY INFO RECEIPT
# =============================================================================
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))

received_bids = data[data["received_info"]]["infoValue"]
not_received_bids = data[~data["received_info"]]["infoValue"]

ax1.hist(not_received_bids, bins=20, alpha=0.5, label="Info Not Received",
         color="#D49764", edgecolor=COLOR_DARK, linewidth=0.5, density=True)
ax1.hist(received_bids, bins=20, alpha=0.5, label="Info Received",
         color="#5B7DAA", edgecolor=COLOR_DARK, linewidth=0.5, density=True)

ax1.set_xlabel("Bid Amount", fontweight="normal", fontsize=18)
ax1.set_ylabel("Probability Density", fontweight="normal", fontsize=18)
ax1.set_title("Bid Distribution by Info Receipt", fontweight="normal", fontsize=22)
ax1.legend(frameon=False, fontsize=14)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

plt.tight_layout()
base = os.path.join(cfg["outdir"], "bid_distribution")
save_figure(fig1, base, cfg)
print("✓ Figure 1 saved as SVG/PDF/PNG:", base)
plt.show()


# =============================================================================
# FIGURE 2: BIAS DISTRIBUTION (subjective - normative)
# =============================================================================
bias_df = data[["infoValue", "infoNorm"]].dropna()
bias = (bias_df["infoValue"] - bias_df["infoNorm"]).to_numpy()

# Palette
N_COLOR_FILL = "#5B7C99"   # slate blue-gray (hist fill)
N_COLOR_EDGE = "#1F2D3A"   # deep slate (axes/edges/zero line)
N_COLOR_MEAN = "#8B6F47"   # warm brown (mean line)

fig2, ax2 = plt.subplots(figsize=(6, 5))

bins = np.linspace(-1.5, 1.5, 31)
ax2.hist(bias, bins=bins, density=True, alpha=0.5,
         color=N_COLOR_FILL, edgecolor=N_COLOR_EDGE, linewidth=0.6,
         label="Bias distribution")

# Zero-bias reference (not in legend)
ax2.axvline(0, linestyle="--", linewidth=1.2, color=N_COLOR_EDGE, alpha=0.75)

# Mean bias line
mu = float(np.mean(bias)) if len(bias) else np.nan
ax2.axvline(mu, linestyle="-", linewidth=1.8, color=N_COLOR_MEAN, alpha=0.95,
            label="Mean bias")

ax2.set_xlabel("Bias (subjective - normative)", fontsize=11)
ax2.set_ylabel("Probability Density", fontsize=11)
ax2.set_title("Distribution of Bias in Information Bids", fontsize=12)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(frameon=False, fontsize=10, loc="upper left")

plt.tight_layout()
base = os.path.join(cfg["outdir"], "bias_distribution")
save_figure(fig2, base, cfg)
print("✓ Figure 2 saved as SVG/PDF/PNG:", base)
plt.show()


# =============================================================================
# FIGURE 3: PLAY PROBABILITY vs EV (pooled logistic + bootstrap 95% CI)
# =============================================================================
# Subject-wise bootstrap if SubID/SubNum exists; else trial-wise.

ev = pd.to_numeric(data["trialEV"], errors="coerce")
y  = data["choice"]

# Coerce play to {0,1}
if y.dtype == bool:
    y = y.astype(int)
else:
    y = pd.to_numeric(y, errors="coerce")
    uniq = np.sort(y.dropna().unique())
    if not set(np.unique(y.dropna())).issubset({0, 1}):
        if len(uniq) >= 2:
            y = y.map({uniq.min(): 0, uniq.max(): 1})
        else:
            raise ValueError("`choice` could not be coerced to binary {0,1}.")

# Pick subject id column if available
sub_col = "SubID" if "SubID" in data.columns else (
          "SubNum" if "SubNum" in data.columns else None)

df = pd.DataFrame({"ev": ev, "play": y})
if sub_col:
    df[sub_col] = data[sub_col]
df = df.dropna()
df = df[(~np.isinf(df["ev"])) & (~np.isinf(df["play"]))]

if len(df) < 10:
    raise ValueError("Not enough rows after cleaning to fit the model.")

# --- Fit pooled logistic once (point curve) ---
X = sm.add_constant(df["ev"].to_numpy())
yy = df["play"].to_numpy().astype(int)
fit = sm.GLM(yy, X, family=sm.families.Binomial()).fit()
beta_hat = fit.params

# Prediction grid & point estimate
ev_min, ev_max = float(df["ev"].min()), float(df["ev"].max())
grid = np.linspace(ev_min, ev_max, 300)
Xg = np.column_stack([np.ones_like(grid), grid])
p_point = expit(Xg @ beta_hat)

# --- Bootstrap 95% CI ---
R = 500  # increase for smoother bands
rng = np.random.default_rng(42)
p_boot = np.empty((R, grid.size), dtype=float)

if sub_col:
    # Subject-wise bootstrap
    subjects = df[sub_col].unique()
    for r in range(R):
        samp_ids = rng.choice(subjects, size=len(subjects), replace=True)
        boot_df = pd.concat([df[df[sub_col] == sid] for sid in samp_ids],
                            ignore_index=True)
        Xb = sm.add_constant(boot_df["ev"].to_numpy())
        yb = boot_df["play"].to_numpy().astype(int)
        try:
            b = sm.GLM(yb, Xb, family=sm.families.Binomial()).fit().params
        except Exception:
            b = beta_hat  # fallback on separation/convergence issues
        p_boot[r, :] = expit(Xg @ b)
else:
    # Trial-wise bootstrap
    n = len(df)
    ev_arr = df["ev"].to_numpy()
    y_arr  = df["play"].to_numpy().astype(int)
    for r in range(R):
        idx = rng.integers(0, n, size=n)
        Xb = sm.add_constant(ev_arr[idx])
        yb = y_arr[idx]
        try:
            b = sm.GLM(yb, Xb, family=sm.families.Binomial()).fit().params
        except Exception:
            b = beta_hat
        p_boot[r, :] = expit(Xg @ b)

p_lo, p_hi = np.percentile(p_boot, [2.5, 97.5], axis=0)

# --- Plot ---
LINE_COLOR = "#5B7C99"
BAND_COLOR = "#C5D3E0"

fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.fill_between(grid, p_lo, p_hi, color=BAND_COLOR, alpha=1.0,
                 label="95% CI (bootstrap)", zorder=2)
ax3.plot(grid, p_lo, color=LINE_COLOR, alpha=0.5, linewidth=1.0, zorder=2)
ax3.plot(grid, p_hi, color=LINE_COLOR, alpha=0.5, linewidth=1.0, zorder=2)
ax3.plot(grid, p_point, color=LINE_COLOR, linewidth=3.0,
         label="Logistic fit", zorder=3)

ax3.set_xlim(ev_min, ev_max)
ax3.set_ylim(-0.02, 1.02)
ax3.set_xlabel("Lottery EV", fontsize=11)
ax3.set_ylabel("P(play)", fontsize=11)
ax3.set_title("Play probability vs EV (logistic, bootstrap 95% CI)", fontsize=12)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.legend(frameon=False, fontsize=10, loc="lower right")

plt.tight_layout()
base = os.path.join(cfg["outdir"], "play_vs_ev_logistic_boot")
save_figure(fig3, base, cfg)
print("✓ Figure 3 saved as SVG/PDF/PNG:", base)
plt.show()