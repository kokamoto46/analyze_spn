#!/usr/bin/env python3
"""
Key features:
- Pure local file I/O
- Robust CLI with sensible defaults
- Headless-safe plotting (matplotlib Agg)
- Clear outputs in ./outputs and ./outputs/figtables

Usage example:
    python analyze_spn.py \
        --data_dir ./data \
        --spn spn_electrode_long.csv \
        --meta electrode_metadata.csv \
        --beh speech_accuracy.csv \
        --out_dir ./outputs \
        --n_perm 5000 \
        --seed 42
"""
from __future__ import annotations

import os
import sys
import warnings
import argparse
import json
from typing import List, Tuple, Optional, Dict

# Use headless-safe backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.stats import norm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

import mne
from mne.stats import permutation_cluster_1samp_test

# =========================
#  Utilities
# =========================
def log(msg: str):
    print(msg, flush=True)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def find_file(base_dir: str, target: str) -> str:
    """
    Resolve an input file path:
    - If 'target' is an existing file path, return it.
    - Else, search under base_dir for an exact filename match.
    - Else, search under base_dir for a file containing 'target' (case-insensitive).
    """
    target = str(target).strip()
    if os.path.isfile(target):
        return os.path.abspath(target)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # exact filename match
    for fn in os.listdir(base_dir):
        if fn == target:
            full = os.path.join(base_dir, fn)
            if os.path.isfile(full):
                return os.path.abspath(full)

    # substring match (case-insensitive)
    t_low = target.lower()
    cands = []
    for fn in os.listdir(base_dir):
        full = os.path.join(base_dir, fn)
        if os.path.isfile(full) and t_low in fn.lower():
            cands.append(full)
    if len(cands) == 1:
        return os.path.abspath(cands[0])
    elif len(cands) > 1:
        # tie-breaker: prefer CSV
        csvs = [c for c in cands if c.lower().endswith(".csv")]
        if len(csvs) == 1:
            return os.path.abspath(csvs[0])
        raise FileNotFoundError(f"Multiple files matched '{target}' in {base_dir}: {cands}")
    raise FileNotFoundError(f"No file matched '{target}' in {base_dir} and not found at path.")

def fe_table(res) -> pd.DataFrame:
    """MixedLM fixed-effects table (b, SE, z, p, 95% CI)."""
    fe = res.fe_params
    se = res.bse_fe
    z  = fe / se
    p  = 2 * norm.sf(np.abs(z))
    ci_low  = fe - 1.96 * se
    ci_high = fe + 1.96 * se
    out = (pd.DataFrame({"b": fe, "SE": se, "z": z, "p": p,
                         "CI_low": ci_low, "CI_high": ci_high})
              .reset_index().rename(columns={"index":"Term"}))
    return out

def try_fit_mixed(formula, data, re="~Condition_c", group="Participant"):
    """Try random slope, fallback to random intercept."""
    kind = "random-slope"
    try:
        res = mixedlm(formula, data=data, groups=data[group], re_formula=re).fit(method="lbfgs")
        return res, kind
    except Exception:
        kind = "random-intercept"
        res = mixedlm(formula, data=data, groups=data[group]).fit(method="lbfgs")
        return res, kind

def ids_with_two_levels(df, id_col, factor_col):
    counts = df.groupby(id_col)[factor_col].nunique()
    return counts[counts >= 2].index.tolist()

def encode_condition(series, pos="in silence", neg="in noise"):
    """
    Coding rule: pos→+0.5, neg→-0.5. Normalize 'noise'/'silence' labels.
    """
    s = series.astype(str).str.strip().str.lower()
    s = s.replace({"noise": "in noise", "silence": "in silence"})
    mapping = {str(pos).lower(): 0.5, str(neg).lower(): -0.5}
    out = s.map(mapping)
    if out.isna().any():
        unk = sorted(set(s[out.isna()].unique()) - set(mapping))
        if len(unk) > 0:
            log(f"[WARN] unexpected Condition labels (ignored): {unk}")
    return out.astype(float)

def build_meta_from_spn(spn):
    """Minimal metadata (ROI/Hemisphere) if electrode_metadata.csv is absent."""
    elec = sorted(spn["Electrode"].astype(str).str.strip().unique())
    def guess_roi(ch):
        chU = ch.upper()
        if chU.startswith("F"): return "Frontal"
        if chU.startswith("C"): return "Central"
        if chU.startswith("P"): return "Parietal"
        if chU.startswith("O"): return "Occipital"
        if chU.startswith("T"): return "Temporal"
        return "Other"
    mid = {"Fz","Cz","Pz","Oz"}
    def guess_hemi(ch):
        ch = str(ch).strip()
        if ch in mid or ch.endswith(("z","Z")): return "Midline"
        if len(ch)>0 and ch[-1].isdigit():
            return "Left" if int(ch[-1])%2==1 else "Right"
        mp = {"T3":"Left","T5":"Left","T7":"Left", "T4":"Right","T6":"Right","T8":"Right"}
        return mp.get(ch, "Midline")
    meta = pd.DataFrame({"Electrode": elec})
    meta["ROI"] = meta["Electrode"].map(guess_roi)
    meta["Hemisphere"] = meta["Electrode"].map(guess_hemi)
    return meta

def guess_hemi_from_name(ch: str) -> str:
    ch = str(ch).strip()
    if ch.endswith(("z","Z")): return "Midline"
    mp = {"T3":"Left","T5":"Left","T7":"Left", "T4":"Right","T6":"Right","T8":"Right"}
    if ch in mp: return mp[ch]
    if len(ch)>0 and ch[-1].isdigit():
        return "Left" if int(ch[-1])%2==1 else "Right"
    return "Midline"

def coalesce_cols(df, base):
    """Merge split columns like ROI_x/ROI_y into base."""
    cols = [c for c in df.columns if c==base or c.startswith(base+"_")]
    if not cols:
        return df
    if base not in df.columns:
        df[base] = pd.NA
    for c in cols:
        if c == base:
            continue
        df[base] = df[base].fillna(df[c])
        df.drop(columns=[c], inplace=True)
    return df

def prepare_meta_safe(meta, spn):
    """
    Prepare Electrode/ROI/Hemisphere safely (Hemisphere optional).
    - cover union of electrodes in spn
    - fill ROI/Hemisphere with guesses if missing
    - merge split columns
    """
    meta = meta.copy()
    meta.columns = [c.strip() for c in meta.columns]
    if "Electrode" not in meta.columns:
        raise ValueError("electrode_metadata.csv must contain 'Electrode' column.")
    meta["Electrode"] = meta["Electrode"].astype(str).str.strip()

    all_elec = pd.DataFrame({"Electrode": sorted(spn["Electrode"].astype(str).str.strip().unique())})
    meta = all_elec.merge(meta, on="Electrode", how="left")

    if "ROI" not in meta.columns:
        aux = build_meta_from_spn(spn)[["Electrode","ROI"]]
        meta = meta.merge(aux, on="Electrode", how="left", suffixes=("","_aux"))
    if "Hemisphere" not in meta.columns:
        aux = build_meta_from_spn(spn)[["Electrode","Hemisphere"]]
        meta = meta.merge(aux, on="Electrode", how="left", suffixes=("","_aux"))

    meta = coalesce_cols(meta, "ROI")
    meta = coalesce_cols(meta, "Hemisphere")

    if meta["Hemisphere"].isna().any():
        meta["Hemisphere"] = meta.apply(
            lambda r: guess_hemi_from_name(r["Electrode"]) if pd.isna(r["Hemisphere"]) else r["Hemisphere"], axis=1
        )
    if meta["ROI"].isna().any():
        aux = build_meta_from_spn(spn)[["Electrode","ROI"]]
        meta = meta.merge(aux, on="Electrode", how="left", suffixes=("","_aux2"))
        if "ROI_aux2" in meta.columns:
            meta["ROI"] = meta["ROI"].fillna(meta["ROI_aux2"])
            meta.drop(columns=["ROI_aux2"], inplace=True)
    return meta[["Electrode","ROI","Hemisphere"]]

def make_diff_matrix(pivot, chan_order, method="electrode_mean"):
    """
    Build X (subjects × channels).
    method='electrode_mean': impute missing cells by channel mean across subjects
                             (and fallback to global mean for all-NaN channels).
    Returns: X, fill_rate, fill_summary
    """
    subs = pivot["Participant"].unique().tolist()
    elec_mean = (pivot.set_index(["Participant","Electrode"])["Diff"]
                      .unstack(0).mean(axis=1, skipna=True))  # index=Electrode
    overall_mean = np.nanmean(elec_mean.values.astype(float))
    X = np.zeros((len(subs), len(chan_order)))
    missing_counter = np.zeros(len(chan_order), dtype=int)
    filled_counter  = np.zeros(len(chan_order), dtype=int)

    for i, s in enumerate(subs):
        df_s = pivot[pivot["Participant"]==s].set_index("Electrode")
        vec = df_s.reindex(chan_order)["Diff"].astype(float)
        miss_idx = np.where(vec.isna().values)[0]
        missing_counter[miss_idx] += 1
        if method == "electrode_mean":
            fill_vals = elec_mean.reindex(chan_order).values.astype(float)
            where_nan_elec_mean = np.isnan(fill_vals)
            if where_nan_elec_mean.any():
                fill_vals[where_nan_elec_mean] = overall_mean if np.isfinite(overall_mean) else 0.0
            vec = vec.fillna(pd.Series(fill_vals, index=vec.index))
        else:
            fill_vals = elec_mean.reindex(chan_order).values.astype(float)
            vec = vec.fillna(pd.Series(fill_vals, index=vec.index))
        filled_counter[miss_idx] += 1
        X[i,:] = vec.values

    total_cells = len(subs) * len(chan_order)
    fill_rate = float(filled_counter.sum()) / float(total_cells) if total_cells > 0 else 0.0
    fill_summary = pd.DataFrame({
        "Electrode": chan_order,
        "N_missing": missing_counter.astype(int),
        "N_filled":  filled_counter.astype(int)
    })
    return X, fill_rate, fill_summary

def find_sig_clusters(T_obs, clusters, p_vals, alpha=0.05, select_rule="min_p"):
    """
    Return list of significant cluster indices; also a representative cluster index.
    select_rule: 'min_p' or 'max_mass'
    """
    sig_idx = [i for i,p in enumerate(p_vals) if p < alpha]
    if len(sig_idx)==0:
        return [], None, []
    masses = []
    for i in range(len(clusters)):
        mask = clusters[i]
        mass = np.nansum(np.abs(T_obs[mask]))
        masses.append(float(mass))
    masses = np.array(masses, dtype=float)
    rep = int(max(sig_idx, key=lambda i: masses[i])) if select_rule=="max_mass" \
          else int(min(sig_idx, key=lambda i: p_vals[i]))
    info = [dict(cluster_id=int(i), pval=float(p_vals[i]), mass=float(masses[i])) for i in sig_idx]
    info = sorted(info, key=lambda d: (d["pval"], -d["mass"]))
    return sig_idx, rep, info

# ---- Figure export (DPI policy) ----
DPI_MAP = {"line": 600, "halftone": 300, "bitmap": 1200}
OUT_EXTS = [".tif", ".png"]  # TIF for submission, PNG for on-screen check

def save_fig(fig, stem, out_dir, kind="halftone", size=(4,4), tight=True):
    """Save figure to out_dir/figtables with specified DPI and both TIF/PNG."""
    try:
        fig.set_size_inches(*size)
    except Exception:
        pass
    dpi = DPI_MAP.get(kind, 300)
    figdir = os.path.join(out_dir, "figtables")
    ensure_dir(figdir)

    saved = []
    for ext in OUT_EXTS:
        fname = os.path.join(figdir, f"{stem}{ext}")
        fig.savefig(fname, dpi=dpi,
                    bbox_inches=("tight" if tight else None),
                    facecolor="white")
        saved.append(fname)
    plt.close(fig)
    log(f"[Saved] {stem} ({kind}, {dpi} dpi) -> {', '.join(saved)}")

# =========================
#  Main pipeline
# =========================
def run_pipeline(spn_path: str,
                 meta_path: Optional[str],
                 beh_path: Optional[str],
                 out_dir: str,
                 n_perm: int = 5000,
                 seed: int = 42) -> None:
    warnings.filterwarnings("ignore")
    ensure_dir(out_dir)

    # ===== 1) Load CSVs =====
    log(f"[INFO] using spn CSV : {spn_path}")
    spn  = pd.read_csv(spn_path)

    if meta_path is None or (isinstance(meta_path, str) and len(meta_path.strip()) == 0):
        log("[WARN] electrode_metadata.csv not provided. Building minimal metadata from spn.")
        meta = build_meta_from_spn(spn)
    else:
        log(f"[INFO] using meta CSV: {meta_path}")
        meta = pd.read_csv(meta_path)

    log(f"[INFO] loaded spn  shape: {spn.shape}")
    log(f"[INFO] loaded meta shape: {meta.shape}")

    # Column normalization
    spn.columns  = [c.strip() for c in spn.columns]
    meta.columns = [c.strip() for c in meta.columns]

    # Required columns
    need_spn_cols = {"Participant","Sex","Condition","Electrode","SPN_uV"}
    miss = need_spn_cols - set(spn.columns)
    if miss:
        raise ValueError(f"spn_electrode_long.csv missing columns: {miss}")

    if "Electrode" not in meta.columns:
        raise ValueError("electrode_metadata.csv must contain 'Electrode' column.")

    # Prepare ROI/Hemisphere safely
    meta = prepare_meta_safe(meta, spn)

    # ===== 2) Preprocess =====
    spn["Participant"] = spn["Participant"].astype(str)
    spn["Sex"] = spn["Sex"].astype(str)

    # Normalize and encode Condition
    spn["Condition"] = spn["Condition"].astype(str).str.strip().str.lower() \
                                     .replace({"noise":"in noise","silence":"in silence"})
    spn["Condition_c"] = encode_condition(spn["Condition"], pos="in silence", neg="in noise")

    # Merge ROI/Hemisphere
    spn = spn.merge(meta, on="Electrode", how="left")

    # Keep participants who have both conditions
    ids = ids_with_two_levels(spn, "Participant", "Condition")
    spn = spn[spn["Participant"].isin(ids)].copy()
    log(f"[INFO] participants: {spn['Participant'].nunique()}")
    log(f"[INFO] electrodes : {spn['Electrode'].nunique()}")

    # Effect coding
    spn["Hemisphere"] = spn["Hemisphere"].fillna(spn["Electrode"].map(guess_hemi_from_name))
    spn["Sex_c"] = spn["Sex"].map({"F": 0.5, "M": -0.5}).astype(float)
    spn["Hem_c"] = spn["Hemisphere"].map({"Right": 0.5, "Left": -0.5, "Midline": 0.0}).fillna(0.0).astype(float)

    # ===== 3) ROI × Hemisphere LMM =====
    formula_lr = "SPN_uV ~ Condition_c * Hem_c + C(ROI) + Sex_c"
    res_lr, kind_lr = try_fit_mixed(formula_lr, spn, re="~Condition_c", group="Participant")

    fe = fe_table(res_lr)
    fe_path = os.path.join(out_dir, "Table_LMM_ROIxHem_fixed_effects.csv")
    fe.to_csv(fe_path, index=False)
    log("\n=== ROI×Hemisphere LMM (fixed effects) ===")
    log(f"Random effects: {kind_lr} / N participants: {spn['Participant'].nunique()} / N rows: {spn.shape[0]}")
    log(fe.to_string(index=False))
    log(f"[Saved] {fe_path}")

    # ===== 4) Per-electrode LMM (Condition main effect, FDR) =====
    rows = []
    for elec, df_e in spn.groupby("Electrode"):
        try:
            model_e = mixedlm("SPN_uV ~ Condition_c + Sex_c", data=df_e,
                              groups=df_e["Participant"], re_formula="~Condition_c").fit(method="lbfgs")
            b = float(model_e.fe_params.get("Condition_c", np.nan))
            se = float(model_e.bse_fe.get("Condition_c", np.nan))
            z  = b / se if np.isfinite(se) and se>0 else np.nan
            p  = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
            rows.append((elec, b, se, p))
        except Exception:
            rows.append((elec, np.nan, np.nan, np.nan))

    elec_tbl = pd.DataFrame(rows, columns=["Electrode","b_Condition","SE","p"])
    elec_tbl["p_FDR"] = np.nan
    mask_valid = elec_tbl["p"].notna()
    if mask_valid.any():
        elec_tbl.loc[mask_valid, "p_FDR"] = multipletests(elec_tbl.loc[mask_valid, "p"], method="fdr_bh")[1]
    elec_tbl = elec_tbl.sort_values("p")

    elec_path = os.path.join(out_dir, "Table_perElectrode_LMM.csv")
    elec_tbl.to_csv(elec_path, index=False)
    log("\n=== Mass-univariate per-electrode (Condition: in noise vs. in silence) ===")
    log(elec_tbl.head(20).to_string(index=False))
    log(f"[Saved] {elec_path}")

    # ===== 5) Topomap & cluster test (in noise − in silence) =====
    mean_cond = (spn.groupby(["Participant","Electrode","Condition"], as_index=False)
                    .agg(SPN=("SPN_uV","mean")))
    pivot = mean_cond.pivot_table(index=["Participant","Electrode"], columns="Condition", values="SPN").reset_index()
    need_cols = {"in noise","in silence"}
    if not need_cols.issubset(pivot.columns):
        raise ValueError("Both conditions (in noise, in silence) are required.")

    pivot["Diff"] = pivot["in noise"] - pivot["in silence"]

    chan_order = meta["Electrode"].tolist()
    X, fill_rate, fill_summary = make_diff_matrix(pivot, chan_order, method="electrode_mean")
    log(f"[INFO] missing filled (electrode_mean): {fill_rate*100:.2f}% of cells")

    fill_summary_path = os.path.join(out_dir, "FillSummary_perElectrode.csv")
    fill_summary.to_csv(fill_summary_path, index=False)
    log(f"[Saved] {fill_summary_path}")

    # MNE info & montage (T3/T4/T5/T6 → T7/T8/P7/P8 for visualization only)
    rename_map = {"T3":"T7","T4":"T8","T5":"P7","T6":"P8"}
    ch_mont = [rename_map.get(ch, ch) for ch in chan_order]
    info = mne.create_info(ch_names=ch_mont, sfreq=500, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    # Evoked single-frame (n_chan × 1)
    data_topo = X.mean(axis=0).reshape(-1,1)
    evk = mne.EvokedArray(data_topo, info, tmin=0)

    # Adjacency & cluster test
    adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    T_obs, clusters, p_vals, _ = permutation_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=int(n_perm), tail=0, out_type="mask", seed=int(seed)
    )

    # List significant clusters + representative selection
    sig_idx, rep_idx, cluster_info = find_sig_clusters(T_obs, clusters, p_vals, alpha=0.05, select_rule="min_p")
    log("\n=== Sensor-space cluster test ===")
    log("significant clusters (id, p, mass): " + json.dumps(cluster_info, ensure_ascii=False))

    # Save cluster membership tables
    if len(sig_idx) > 0:
        all_tabs = []
        for cid in sig_idx:
            mask_bool = clusters[cid]
            tab = pd.DataFrame({"Electrode": chan_order,
                                "InCluster": mask_bool.astype(bool),
                                "DiffMean_uV": X.mean(0)})
            tab = tab.merge(meta[["Electrode","ROI","Hemisphere"]], on="Electrode", how="left")
            tab["SignAgreement"] = np.mean(np.sign(X)==np.sign(X.mean(0)), axis=0)
            tab["cluster_id"] = cid
            tab["p_value"] = float(p_vals[cid])
            tab["cluster_mass"] = float(np.nansum(np.abs(T_obs[mask_bool])))
            fn = os.path.join(out_dir, f"Supp_Table_cluster_electrodes_cluster{cid}.csv")
            tab.to_csv(fn, index=False)
            log(f"[Saved] {fn}")
            all_tabs.append(tab)
        all_tab = pd.concat(all_tabs, ignore_index=True)
        all_all_path = os.path.join(out_dir, "Supp_Table_cluster_electrodes_ALL.csv")
        all_tab.to_csv(all_all_path, index=False)
        log(f"[Saved] {all_all_path}")

    # Masked topomap for representative cluster
    mask_bool = np.zeros(len(chan_order), dtype=bool) if rep_idx is None else clusters[rep_idx]
    mask_bool_plot = (mask_bool[:, np.newaxis] if mask_bool.ndim==1 else mask_bool)
    fig1 = evk.plot_topomap(times=[0], time_format="", scalings=1,
                            units=dict(eeg="µV"), mask=mask_bool_plot,
                            mask_params=dict(markersize=10))
    # mne returns a Figure
    ttl = "in noise − in silence (−200–0 ms SPN)"
    if rep_idx is not None:
        ttl += f"  [rep cluster id={rep_idx}, p={float(p_vals[rep_idx]):.4f}]"
    try:
        fig1.suptitle(ttl)
    except Exception:
        pass
    save_fig(fig1, stem="Fig1_topomap_mask", out_dir=out_dir, kind="halftone", size=(4,4), tight=True)

    # ===== 6) LMM of cluster-mean SPN (Condition*Sex) =====
    if rep_idx is None:
        log("[WARN] No significant cluster. Skipping cluster-mean analyses.")
        cluster_ch = []
    else:
        cluster_ch = [ch for ch, m in zip(chan_order, mask_bool.astype(bool)) if m]

    if len(cluster_ch)>0:
        spn_cl = (spn[spn["Electrode"].isin(cluster_ch)]
                  .groupby(["Participant","Sex","Sex_c","Condition","Condition_c"], as_index=False)
                  .agg(SPN_cluster_uV=("SPN_uV","mean")))
        res_cl, kind_cl = try_fit_mixed("SPN_cluster_uV ~ Condition_c * Sex_c", spn_cl,
                                        re="~Condition_c", group="Participant")
        fe_cl = fe_table(res_cl)
        fe_cl_path = os.path.join(out_dir, "Table_LMM_clusterMean_fixed_effects.csv")
        fe_cl.to_csv(fe_cl_path, index=False)
        log("\n=== LMM (cluster-mean SPN) ===")
        log(f"Random effects: {kind_cl}")
        log(fe_cl.to_string(index=False))
        log(f"[Saved] {fe_cl_path}")

        # simple effects (Sex: F=+0.5, M=-0.5)
        b_cond = float(res_cl.fe_params.get("Condition_c", np.nan))
        b_int  = float(res_cl.fe_params.get("Condition_c:Sex_c", 0.0))
        b_F = b_cond + 0.5*b_int
        b_M = b_cond - 0.5*b_int
        log("\n[Simple effect] Condition (noise vs silence)")
        log(f"  Sex=F: {b_F}")
        log(f"  Sex=M: {b_M}")

        # Violin plot
        fig2, ax = plt.subplots(figsize=(4,4))
        g = [spn_cl.loc[spn_cl["Condition"]==cond, "SPN_cluster_uV"].values
             for cond in ["in silence","in noise"]]
        ax.violinplot(g, showmeans=True, showmedians=False)
        ax.set_xticks([1,2]); ax.set_xticklabels(["in silence","in noise"])
        ax.set_ylabel("Cluster-mean SPN (µV)")
        save_fig(fig2, stem="Fig2_cluster_violin_cond", out_dir=out_dir, kind="line", size=(4,4), tight=True)

    # ===== 7) Sensitivity analyses (LOEO & participant bootstrap) =====
    if len(cluster_ch)>0:
        # (a) LOEO
        loeo_rows = []
        for drop in cluster_ch:
            use_ch = [c for c in cluster_ch if c!=drop]
            tmp = (spn[spn["Electrode"].isin(use_ch)]
                   .groupby(["Participant","Sex","Sex_c","Condition","Condition_c"], as_index=False)
                   .agg(SPN_uV=("SPN_uV","mean")))
            try:
                res_tmp, kind_tmp = try_fit_mixed("SPN_uV ~ Condition_c * Sex_c", tmp,
                                                  re="~Condition_c", group="Participant")
                b_cond = float(res_tmp.fe_params.get("Condition_c", np.nan))
                b_int  = float(res_tmp.fe_params.get("Condition_c:Sex_c", np.nan))
                p_cond = 2 * norm.sf(abs(b_cond / res_tmp.bse_fe.get("Condition_c", np.nan)))
                p_int  = 2 * norm.sf(abs(b_int  / res_tmp.bse_fe.get("Condition_c:Sex_c", np.nan)))
            except Exception:
                b_cond = b_int = p_cond = p_int = np.nan
                kind_tmp = "failed"
            loeo_rows.append((drop, kind_tmp, b_cond, p_cond, b_int, p_int))
        loeo = pd.DataFrame(loeo_rows, columns=["DroppedElectrode","Random",
                                                "b_Condition","p_Condition",
                                                "b_CondxSex","p_CondxSex"])
        loeo_path = os.path.join(out_dir, "sens_LOEO.csv")
        loeo.to_csv(loeo_path, index=False)
        log(f"[Saved] {loeo_path}")

        # (b) participant bootstrap (true resampling with replacement)
        rng = np.random.default_rng(seed)
        spn_cl = (spn[spn["Electrode"].isin(cluster_ch)]
                  .groupby(["Participant","Sex","Sex_c","Condition","Condition_c"], as_index=False)
                  .agg(SPN_cluster_uV=("SPN_uV","mean")))
        subs_unique = sorted(spn_cl["Participant"].astype(str).unique())
        parts = {sid: spn_cl[spn_cl["Participant"].astype(str)==sid].copy() for sid in subs_unique}

        B = 500
        boot = []
        for b in range(B):
            samp = rng.choice(subs_unique, size=len(subs_unique), replace=True)
            tmp_list = []
            for j, sid in enumerate(samp):
                df_sid = parts[sid].copy()
                df_sid["Participant"] = f"{sid}_b{b}_{j}"
                tmp_list.append(df_sid)
            tmp = pd.concat(tmp_list, ignore_index=True)
            try:
                res_b, _ = try_fit_mixed("SPN_cluster_uV ~ Condition_c * Sex_c",
                                         tmp, re="~Condition_c", group="Participant")
                b_cond = float(res_b.fe_params.get("Condition_c", np.nan))
                b_int  = float(res_b.fe_params.get("Condition_c:Sex_c", np.nan))
            except Exception:
                b_cond = b_int = np.nan
            boot.append((b_cond, b_int))
        boot = pd.DataFrame(boot, columns=["b_Condition","b_CondxSex"])
        boot_path = os.path.join(out_dir, "sens_bootstrap.csv")
        boot.to_csv(boot_path, index=False)
        log(f"[Saved] {boot_path}")

        # Figures
        fig3a, ax = plt.subplots(figsize=(6,3.2))
        ax.scatter(range(len(loeo)), loeo["b_Condition"], s=20)
        ax.set_xticks(range(len(loeo))); ax.set_xticklabels(loeo["DroppedElectrode"], rotation=90)
        ax.set_ylabel("b (Condition)")
        save_fig(fig3a, stem="Fig3a_LOEO_b_condition_swarm", out_dir=out_dir, kind="line", size=(6,3.2), tight=True)

        fig3b, ax = plt.subplots(figsize=(6,3.2))
        ax.scatter(range(len(loeo)), loeo["p_CondxSex"], s=20)
        ax.axhline(0.05, linestyle="--")
        ax.set_xticks(range(len(loeo))); ax.set_xticklabels(loeo["DroppedElectrode"], rotation=90)
        ax.set_ylabel("p (Cond×Sex)")
        save_fig(fig3b, stem="Fig3b_LOEO_p_condxsex_swarm", out_dir=out_dir, kind="line", size=(6,3.2), tight=True)

        fig3c, ax = plt.subplots(figsize=(4,3))
        ax.hist(boot["b_Condition"].dropna(), bins=30)
        ax.set_xlabel("b (Condition)"); ax.set_ylabel("Count")
        save_fig(fig3c, stem="Fig3c_boot_b_condition_density", out_dir=out_dir, kind="line", size=(4,3), tight=True)

        fig3d, ax = plt.subplots(figsize=(4,3))
        ax.hist(boot["b_CondxSex"].dropna(), bins=30)
        ax.set_xlabel("b (Condition×Sex)"); ax.set_ylabel("Count")
        save_fig(fig3d, stem="Fig3d_boot_b_condxsex_density", out_dir=out_dir, kind="line", size=(4,3), tight=True)

    log("\n=== Main analysis block completed ===")
    log(f"Outputs are saved to: {out_dir} and {os.path.join(out_dir, 'figtables')}")

    # ===== 8) GLM: SPN(in noise, cluster mean) × speech-in-noise (by SNR) =====
    if beh_path is None or len(str(beh_path).strip()) == 0:
        log("[WARN] speech_accuracy.csv not provided. Skipping GLM.")
        return

    beh = pd.read_csv(beh_path)
    log(f"[INFO] speech_accuracy.csv shape: {beh.shape}")

    if len(cluster_ch)==0:
        log("[WARN] No significant cluster. Skipping GLM.")
        return

    # cluster-mean (representative cluster) at participant × condition average
    if "Trial" in spn.columns:
        clu_trial = (spn[spn["Electrode"].isin(cluster_ch)]
                     .groupby(["Participant","Sex","Condition","Trial"], as_index=False)
                     .agg(SPN_cluster_uV=("SPN_uV","mean")))
        clu = (clu_trial
               .groupby(["Participant","Sex","Condition"], as_index=False)
               .agg(SPN_cluster_uV=("SPN_cluster_uV","mean")))
    else:
        clu = (spn[spn["Electrode"].isin(cluster_ch)]
               .groupby(["Participant","Sex","Condition"], as_index=False)
               .agg(SPN_cluster_uV=("SPN_uV","mean")))

    # in noise centered SPN
    clu["Condition"] = clu["Condition"].astype(str).str.strip().str.lower() \
                                     .replace({"noise":"in noise","silence":"in silence"})
    clu_noise = clu[clu["Condition"]=="in noise"].copy()
    if clu_noise.empty:
        raise ValueError("Cluster-mean SPN (in noise) is empty. Check Condition labels and cluster extraction.")
    clu_noise["Sex01"] = clu_noise["Sex"].map({"F":1, "M":0}).astype(int)
    spn_mean = clu_noise["SPN_cluster_uV"].mean()
    clu_noise["SPN_c"] = clu_noise["SPN_cluster_uV"] - spn_mean

    # required columns
    need_beh = {"Participant","SNR_dB","n_correct","n_trials"}
    miss_beh = need_beh - set(beh.columns)
    if miss_beh:
        raise ValueError(f"speech_accuracy.csv missing columns: {miss_beh}")

    merged = beh.merge(clu_noise[["Participant","SPN_c","Sex01"]], on="Participant", how="inner").copy()
    if merged.empty:
        raise ValueError("No overlapping participants between SPN and behavior. Check 'Participant' IDs.")
    merged["SNR_dB"]    = merged["SNR_dB"].astype(int)
    merged["n_trials"]  = pd.to_numeric(merged["n_trials"], errors="coerce")
    merged["n_correct"] = pd.to_numeric(merged["n_correct"], errors="coerce")
    bad = merged[(~np.isfinite(merged["n_trials"])) | (~np.isfinite(merged["n_correct"])) |
                 (merged["n_trials"]<=0) | (merged["n_correct"]<0) | (merged["n_correct"]>merged["n_trials"])]
    if not bad.empty:
        raise ValueError("Invalid values in n_trials/n_correct.")
    merged["prop"] = merged["n_correct"] / merged["n_trials"]

    # reference SNR: -15 dB if available, otherwise the minimum
    snr_all = sorted(merged["SNR_dB"].unique().tolist())
    cats = [-15] + [s for s in snr_all if s != -15] if (-15 in snr_all) else [snr_all[0]] + snr_all[1:]
    merged["SNR_dB"] = pd.Categorical(merged["SNR_dB"], categories=cats, ordered=True)

    from patsy import dmatrices
    y, X = dmatrices("prop ~ C(SNR_dB) + SPN_c + C(SNR_dB):SPN_c + Sex01",
                     data=merged, return_type="dataframe")

    glm = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=merged["n_trials"])
    res = glm.fit(cov_type="cluster", cov_kwds={"groups": merged["Participant"]})

    # Write summary to text
    summ_path = os.path.join(out_dir, "GLM_logit_summary.txt")
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(str(res.summary()))
    log("\n=== Binomial GLM (logit) with cluster-robust SE (by Participant) ===")
    log(f"[Saved] {summ_path}")

    # OR table
    coefs, ses = res.params, res.bse
    OR    = np.exp(coefs); CI_lo = np.exp(coefs - 1.96*ses); CI_hi = np.exp(coefs + 1.96*ses)
    or_table = pd.DataFrame({"coef(log-odds)": coefs, "SE": ses, "OR": OR, "OR_CI95_low": CI_lo, "OR_CI95_high": CI_hi, "p": res.pvalues})
    or_path = os.path.join(out_dir, "GLM_odds_ratios.csv")
    or_table.to_csv(or_path, index=False)
    log(f"[Saved] {or_path}")

    # Simple effects of SPN_c by SNR (cluster-robust t, df = G-1)
    from scipy.stats import t as student_t
    G = int(merged["Participant"].nunique()); df_t = max(G - 1, 1)
    cov = res.cov_params()
    snr_levels = [merged["SNR_dB"].cat.categories[0]] + [lvl for lvl in merged["SNR_dB"].cat.categories[1:]]
    rows = []
    for snr in snr_levels:
        base_key = "SPN_c"
        int_key  = f"C(SNR_dB)[T.{snr}]:SPN_c" if snr != snr_levels[0] else None
        b = coefs.get(base_key, np.nan); var = cov.loc[base_key, base_key] if base_key in cov.index else np.nan
        if int_key is not None and int_key in coefs.index:
            b += coefs[int_key]
            var = var + cov.loc[int_key, int_key] + 2*cov.loc[base_key, int_key]
        se = float(np.sqrt(var)) if np.isfinite(var) else np.nan
        tval = float(b / se) if (np.isfinite(b) and np.isfinite(se) and se > 0) else np.nan
        p = float(2 * student_t.sf(abs(tval), df_t)) if np.isfinite(tval) else np.nan
        rows.append(dict(SNR_dB=int(snr), logit_b=b, SE=se, t=tval, df=df_t, p=p,
                         OR=float(np.exp(b)) if np.isfinite(b) else np.nan,
                         OR_CI_low=float(np.exp(b - 1.96*se)) if np.isfinite(se) else np.nan,
                         OR_CI_high=float(np.exp(b + 1.96*se)) if np.isfinite(se) else np.nan))
    simp_or = pd.DataFrame(rows).sort_values("SNR_D_b".lower().replace("_d_b","_dB") if False else "SNR_dB").reset_index(drop=True)
    simp_or["p_FDR"] = multipletests(simp_or["p"], method="fdr_bh")[1]
    simp_or_path = os.path.join(out_dir, "Table_simple_OR_by_SNR.csv")
    simp_or.to_csv(simp_or_path, index=False)
    log("\n=== Simple effects of SPN_c (OR) by SNR (cluster-t, df = G-1) ===")
    log(simp_or.to_string(index=False))
    log(f"[Saved] {simp_or_path}")

    log("\n=== All done ===")

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="SPN EEG analysis (local/GitHub-ready).")
    p.add_argument("--data_dir", type=str, default="./data",
                   help="Directory to search for input CSVs when relative names are given.")
    p.add_argument("--spn", type=str, required=True,
                   help="Path to spn_electrode_long.csv OR filename to find under --data_dir.")
    p.add_argument("--meta", type=str, default="",
                   help="Path to electrode_metadata.csv OR filename to find under --data_dir (optional).")
    p.add_argument("--beh", type=str, default="",
                   help="Path to speech_accuracy.csv OR filename to find under --data_dir (optional; needed for GLM).")
    p.add_argument("--out_dir", type=str, default="./outputs",
                   help="Output directory.")
    p.add_argument("--n_perm", type=int, default=5000,
                   help="Number of permutations for cluster test.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    # Resolve file paths
    data_dir = args.data_dir
    spn_path  = find_file(data_dir, args.spn)
    meta_path = None
    if args.meta and len(args.meta.strip())>0:
        try:
            meta_path = find_file(data_dir, args.meta)
        except Exception as e:
            log(f"[WARN] meta file not found; will build from spn. ({e})")
            meta_path = None
    beh_path = None
    if args.beh and len(args.beh.strip())>0:
        beh_path = find_file(data_dir, args.beh)
    out_dir = args.out_dir

    run_pipeline(spn_path=spn_path,
                 meta_path=meta_path,
                 beh_path=beh_path,
                 out_dir=out_dir,
                 n_perm=args.n_perm,
                 seed=args.seed)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Provide a concise error message and exit non-zero
        log(f"[ERROR] {e.__class__.__name__}: {e}")
        sys.exit(1)
