#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_spn_cnv_analysis.py

Desktop-oriented analysis script for SPN/CNV EEG data.

This script:
1. Loads non-interpolated trial-wise long-format EEG data
2. Optionally loads electrode metadata and run-level QC tables
3. Applies primary-analysis filtering
4. Runs ROI × Hemisphere mixed-effects models for SPN and CNV
5. Optionally fits supplementary models with Sex
6. Runs electrode-wise mixed-effects models with FDR correction
7. Generates descriptive topographic and ROI-panel figures
8. Optionally summarizes speech-in-noise performance and
   computes exploratory correlations with EEG measures

Inputs
------
Required:
    anticipatory_electrode_long_noninterp.csv

Optional:
    electrode_metadata.csv
    run_qc.csv
    speech_accuracy.csv

Outputs
-------
Tables and figures are written to the output directory.
"""

import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm, t as student_t, pearsonr, spearmanr
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

import mne

warnings.filterwarnings("ignore")


def find_input_file(input_dir: Path, target_name: str) -> Path:
    matches = sorted(input_dir.glob(f"*{target_name}*"))
    files = [p for p in matches if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No file containing '{target_name}' was found in {input_dir}")
    return files[0]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")


def fe_table(res):
    fe = res.fe_params
    se = res.bse_fe
    z = fe / se
    p = 2 * norm.sf(np.abs(z))
    ci_low = fe - 1.96 * se
    ci_high = fe + 1.96 * se
    out = (
        pd.DataFrame({
            "b": fe,
            "SE": se,
            "z": z,
            "p": p,
            "CI_low": ci_low,
            "CI_high": ci_high
        })
        .reset_index()
        .rename(columns={"index": "Term"})
    )
    return out


def try_fit_mixed(formula, data, re="~Condition_c", group="Participant"):
    kind = "random-slope"
    try:
        res = mixedlm(
            formula,
            data=data,
            groups=data[group],
            re_formula=re
        ).fit(method="lbfgs", reml=True)
        return res, kind
    except Exception:
        kind = "random-intercept"
        res = mixedlm(
            formula,
            data=data,
            groups=data[group]
        ).fit(method="lbfgs", reml=True)
        return res, kind


def ids_with_two_levels(df, id_col, factor_col):
    counts = df.groupby(id_col)[factor_col].nunique()
    return counts[counts >= 2].index.tolist()


def encode_condition(series, pos="in silence", neg="in noise"):
    s = series.astype(str).str.strip().str.lower()
    s = s.replace({"noise": "in noise", "silence": "in silence"})
    mapping = {str(pos).lower(): 0.5, str(neg).lower(): -0.5}
    return s.map(mapping).astype(float)


def guess_roi(ch):
    chU = str(ch).upper()
    if chU.startswith("F"):
        return "Frontal"
    if chU.startswith("C"):
        return "Central"
    if chU.startswith("P"):
        return "Parietal"
    if chU.startswith("O"):
        return "Occipital"
    if chU.startswith("T"):
        return "Temporal"
    return "Other"


def guess_hemi_from_name(ch):
    ch = str(ch).strip()
    if ch.endswith(("z", "Z")):
        return "Midline"
    mp = {"T3": "Left", "T5": "Left", "T7": "Left",
          "T4": "Right", "T6": "Right", "T8": "Right"}
    if ch in mp:
        return mp[ch]
    if len(ch) > 0 and ch[-1].isdigit():
        return "Left" if int(ch[-1]) % 2 == 1 else "Right"
    return "Midline"


def build_meta_from_spn(spn):
    elec = sorted(spn["Electrode"].astype(str).str.strip().unique())
    meta = pd.DataFrame({"Electrode": elec})
    meta["ROI"] = meta["Electrode"].map(guess_roi)
    meta["Hemisphere"] = meta["Electrode"].map(guess_hemi_from_name)
    return meta


def prepare_meta_safe(meta, df):
    meta = meta.copy()
    meta.columns = [c.strip() for c in meta.columns]
    if "Electrode" not in meta.columns:
        raise ValueError("electrode_metadata.csv must contain an 'Electrode' column.")
    meta["Electrode"] = meta["Electrode"].astype(str).str.strip()

    all_elec = pd.DataFrame({
        "Electrode": sorted(df["Electrode"].astype(str).str.strip().unique())
    })
    meta = all_elec.merge(meta, on="Electrode", how="left")

    if "ROI" not in meta.columns or meta["ROI"].isna().any():
        aux = build_meta_from_spn(df)[["Electrode", "ROI"]]
        meta = meta.drop(columns=[c for c in ["ROI"] if c in meta.columns], errors="ignore")
        meta = meta.merge(aux, on="Electrode", how="left")

    if "Hemisphere" not in meta.columns or meta["Hemisphere"].isna().any():
        aux = build_meta_from_spn(df)[["Electrode", "Hemisphere"]]
        meta = meta.drop(columns=[c for c in ["Hemisphere"] if c in meta.columns], errors="ignore")
        meta = meta.merge(aux, on="Electrode", how="left")

    meta["Hemisphere"] = meta["Hemisphere"].fillna(meta["Electrode"].map(guess_hemi_from_name))
    return meta[["Electrode", "ROI", "Hemisphere"]]


def save_fig(fig, path: Path, dpi=300, tight=True):
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight" if tight else None,
        facecolor="white"
    )
    plt.close(fig)
    print(f"[Saved] {path}")


def plot_roi_condition_panels(desc_roi, out_path: Path,
                              roi_order=None,
                              title="Mean anticipatory activity by ROI and condition"):
    if roi_order is None:
        roi_order = ["Frontal", "Central", "Parietal", "Occipital"]
    roi_order = [r for r in roi_order if r in desc_roi["ROI"].unique()]
    cond_order = ["in silence", "in noise"]

    n_panels = len(roi_order)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 4.2), sharey=True)
    if n_panels == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, roi in zip(axes, roi_order):
        wide = (
            desc_roi[desc_roi["ROI"] == roi]
            .pivot_table(index="Participant", columns="Condition", values="Mean_uV", aggfunc="mean")
            .reindex(columns=cond_order)
        )

        x = [0, 1]
        for _, row in wide.iterrows():
            if row.notna().all():
                ax.plot(x, row.values, color="0.75", linewidth=1.0, alpha=0.8, zorder=1)

        for i, cond in enumerate(cond_order):
            vals = wide[cond].dropna().values
            jitter = rng.normal(loc=0.0, scale=0.035, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=26, color="black", alpha=0.85, zorder=2)

            if len(vals) >= 2:
                mu = np.mean(vals)
                se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                tcrit = student_t.ppf(1 - 0.05 / 2, len(vals) - 1)
                ci_lo = mu - tcrit * se
                ci_hi = mu + tcrit * se
                ax.hlines(mu, i - 0.18, i + 0.18, color="black", linewidth=2.0, zorder=3)
                ax.vlines(i, ci_lo, ci_hi, color="black", linewidth=1.5, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(["silence", "noise"])
        ax.set_title(roi, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlim(-0.35, 1.35)

    axes[0].set_ylabel("Mean amplitude (µV)", fontsize=12)
    fig.suptitle(title, fontsize=14, y=1.02)
    save_fig(fig, out_path, dpi=300, tight=True)


def make_topomap_from_condition(mean_cond_df, condition_label, title, out_path: Path):
    tmp = mean_cond_df[mean_cond_df["Condition"] == condition_label].copy()
    tmp = tmp.sort_values("Electrode")
    ch_names = tmp["Electrode"].astype(str).tolist()

    rename_map = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    ch_names_plot = [rename_map.get(ch, ch) for ch in ch_names]

    info = mne.create_info(ch_names=ch_names_plot, sfreq=500, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    data_v = (tmp["Mean_uV"].values * 1e-6).reshape(-1, 1)
    evk = mne.EvokedArray(data_v, info, tmin=0)

    fig = evk.plot_topomap(
        times=[0],
        time_format="",
        scalings=dict(eeg=1e6),
        units=dict(eeg="µV"),
        show=False
    )
    fig.suptitle(title)
    save_fig(fig, out_path, dpi=300, tight=True)


def build_roi_trial(df, meta):
    x = df.merge(meta, on="Electrode", how="left").copy()
    x["Hemisphere"] = x["Hemisphere"].fillna(x["Electrode"].map(guess_hemi_from_name))
    x["Condition_c"] = encode_condition(x["Condition"], pos="in silence", neg="in noise")
    x["Hem_c"] = x["Hemisphere"].map({"Right": -0.5, "Left": 0.5, "Midline": 0.0}).fillna(0.0).astype(float)

    roi_trial = (
        x.groupby(["Participant", "RunID", "Trial", "Condition", "ROI", "Hemisphere"], as_index=False)
        .agg(
            Mean_uV=("Value_uV", "mean"),
            n_chan_used=("Electrode", "nunique")
        )
    )

    roi_total = (
        meta.groupby(["ROI", "Hemisphere"], as_index=False)
        .agg(n_chan_total=("Electrode", "nunique"))
    )

    roi_trial = roi_trial.merge(roi_total, on=["ROI", "Hemisphere"], how="left")
    roi_trial["prop_chan_used"] = roi_trial["n_chan_used"] / roi_trial["n_chan_total"]
    roi_trial = roi_trial[roi_trial["prop_chan_used"] >= 0.5].copy()
    roi_trial["Condition_c"] = encode_condition(roi_trial["Condition"], pos="in silence", neg="in noise")
    roi_trial["Hem_c"] = roi_trial["Hemisphere"].map({"Right": -0.5, "Left": 0.5, "Midline": 0.0}).fillna(0.0).astype(float)
    return x, roi_trial


def fit_with_sex(roi_df, long_df, out_csv: Path):
    if "Sex" not in long_df.columns or long_df["Sex"].isna().all():
        return None
    aux = long_df[["Participant", "RunID", "Trial", "Sex"]].drop_duplicates()
    x = roi_df.merge(aux, on=["Participant", "RunID", "Trial"], how="left")
    x["Sex"] = x["Sex"].astype(str)
    x["Sex_c"] = x["Sex"].map({"F": -0.5, "M": 0.5})
    x = x[x["Sex_c"].notna()].copy()
    if len(x) == 0:
        return None
    formula = "Mean_uV ~ Condition_c * Hem_c + C(ROI) + Sex_c + Condition_c:Sex_c"
    res, kind = try_fit_mixed(formula, x, re="~Condition_c", group="Participant")
    tbl = fe_table(res)
    save_table(tbl, out_csv)
    return tbl, kind


def electrodewise_lmm(df, out_csv: Path):
    rows = []
    for elec, df_e in df.groupby("Electrode"):
        try:
            model_e = mixedlm(
                "Value_uV ~ Condition_c",
                data=df_e.assign(Condition_c=encode_condition(df_e["Condition"])),
                groups=df_e["Participant"],
                re_formula="~Condition_c"
            ).fit(method="lbfgs", reml=True)

            b = float(model_e.fe_params.get("Condition_c", np.nan))
            se = float(model_e.bse_fe.get("Condition_c", np.nan))
            z = b / se if np.isfinite(se) and se > 0 else np.nan
            p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
            rows.append((elec, b, se, z, p))
        except Exception:
            rows.append((elec, np.nan, np.nan, np.nan, np.nan))

    tbl = pd.DataFrame(rows, columns=["Electrode", "b_Condition", "SE", "z", "p"])
    mask = tbl["p"].notna()
    tbl.loc[mask, "p_FDR"] = multipletests(tbl.loc[mask, "p"], method="fdr_bh")[1]
    tbl = tbl.sort_values("p")
    save_table(tbl, out_csv)
    return tbl


def roi_desc(roi_df):
    return (
        roi_df.groupby(["Participant", "Condition", "ROI"], as_index=False)
        .agg(Mean_uV=("Mean_uV", "mean"))
    )


def corr_row(participant_summary, x, y, xname, yname):
    df = participant_summary[[x, y]].dropna()
    if len(df) < 3:
        return {
            "X": xname, "Y": yname, "N": len(df),
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan
        }
    pr, pp = pearsonr(df[x], df[y])
    sr, sp = spearmanr(df[x], df[y])
    return {
        "X": xname, "Y": yname, "N": len(df),
        "pearson_r": pr, "pearson_p": pp,
        "spearman_rho": sr, "spearman_p": sp
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing anticipatory_electrode_long_noninterp.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output tables and figures")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to anticipatory_electrode_long_noninterp.csv")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Optional path to electrode_metadata.csv")
    parser.add_argument("--run_qc", type=str, default=None,
                        help="Optional path to run_qc.csv")
    parser.add_argument("--speech", type=str, default=None,
                        help="Optional path to speech_accuracy.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    data_csv = Path(args.data) if args.data else find_input_file(input_dir, "anticipatory_electrode_long_noninterp")
    print(f"[INFO] Using data CSV: {data_csv}")
    dat = pd.read_csv(data_csv)

    if args.metadata:
        meta_csv = Path(args.metadata)
        meta = pd.read_csv(meta_csv)
        print(f"[INFO] Using electrode metadata: {meta_csv}")
    else:
        try:
            meta_csv = find_input_file(input_dir, "electrode_metadata")
            meta = pd.read_csv(meta_csv)
            print(f"[INFO] Using electrode metadata: {meta_csv}")
        except Exception:
            print("[WARN] electrode_metadata.csv not found. Metadata will be inferred from channel names.")
            meta = build_meta_from_spn(dat)

    if args.run_qc:
        qc_csv = Path(args.run_qc)
        run_qc = pd.read_csv(qc_csv)
        print(f"[INFO] Using QC CSV: {qc_csv}")
    else:
        try:
            qc_csv = find_input_file(input_dir, "run_qc")
            run_qc = pd.read_csv(qc_csv)
            print(f"[INFO] Using QC CSV: {qc_csv}")
        except Exception:
            run_qc = None
            print("[WARN] run_qc.csv not found. Proceeding without QC filtering.")

    dat.columns = [c.strip() for c in dat.columns]
    meta.columns = [c.strip() for c in meta.columns]

    need_cols = {"Participant", "RunID", "Trial", "Condition", "Component", "Electrode", "Value_uV"}
    miss = need_cols - set(dat.columns)
    if miss:
        raise ValueError(f"Missing required columns in anticipatory_electrode_long_noninterp.csv: {miss}")

    meta = prepare_meta_safe(meta, dat)

    dat["Participant"] = dat["Participant"].astype(str).str.zfill(2)
    dat["RunID"] = dat["RunID"].astype(str).str.strip()
    dat["Trial"] = pd.to_numeric(dat["Trial"], errors="coerce")
    dat["Condition"] = (
        dat["Condition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"noise": "in noise", "silence": "in silence"})
    )
    dat["Component"] = dat["Component"].astype(str).str.strip().str.upper()

    if run_qc is not None:
        run_qc["participant"] = run_qc["participant"].astype(str).str.zfill(2)
        run_qc["runID"] = run_qc["runID"].astype(str).str.strip()
        keep_cols = ["runID", "participant", "keep_primary", "keep_primary_participant"]
        keep_cols = [c for c in keep_cols if c in run_qc.columns]
        run_qc = run_qc[keep_cols].drop_duplicates()
        run_qc = run_qc.rename(columns={"runID": "RunID", "participant": "Participant"})
        dat = dat.merge(run_qc, on=["RunID", "Participant"], how="left")
        if "keep_primary" in dat.columns:
            dat = dat[dat["keep_primary"] == True].copy()

    ids = ids_with_two_levels(dat, "Participant", "Condition")
    dat = dat[dat["Participant"].isin(ids)].copy()

    print(f"[INFO] Participants in primary analysis: {dat['Participant'].nunique()}")
    print(f"[INFO] Runs in primary analysis        : {dat['RunID'].nunique()}")
    print(f"[INFO] Rows                            : {dat.shape[0]}")

    spn = dat[dat["Component"] == "SPN"].copy()
    cnv = dat[dat["Component"] == "CNV"].copy()

    spn_long, spn_roi = build_roi_trial(spn, meta)
    cnv_long, cnv_roi = build_roi_trial(cnv, meta)

    formula_primary = "Mean_uV ~ Condition_c * Hem_c + C(ROI)"
    res_spn, kind_spn = try_fit_mixed(formula_primary, spn_roi, re="~Condition_c", group="Participant")
    tbl_spn = fe_table(res_spn)
    save_table(tbl_spn, tables_dir / "Table_Primary_SPN_ROI_Hemisphere_LMM.csv")

    res_cnv, kind_cnv = try_fit_mixed(formula_primary, cnv_roi, re="~Condition_c", group="Participant")
    tbl_cnv = fe_table(res_cnv)
    save_table(tbl_cnv, tables_dir / "Table_Control_CNV_ROI_Hemisphere_LMM.csv")

    fit_with_sex(spn_roi, spn_long, tables_dir / "Supp_Table_SPN_LMM_withSex.csv")
    fit_with_sex(cnv_roi, cnv_long, tables_dir / "Supp_Table_CNV_LMM_withSex.csv")

    electrodewise_lmm(spn_long, tables_dir / "Supp_Table_SPN_ElectrodeWise_LMM_FDR.csv")
    electrodewise_lmm(cnv_long, tables_dir / "Supp_Table_CNV_ElectrodeWise_LMM_FDR.csv")

    mean_cond_spn = (
        spn_long.groupby(["Participant", "Electrode", "Condition"], as_index=False)
        .agg(Mean_uV=("Value_uV", "mean"))
    )
    cond_means_spn = (
        mean_cond_spn.groupby(["Condition", "Electrode"], as_index=False)["Mean_uV"]
        .mean()
    )

    make_topomap_from_condition(
        cond_means_spn,
        condition_label="in silence",
        title="SPN in silence (-200 to 0 ms)",
        out_path=figures_dir / "Fig2a_SPN_topomap_in_silence.png"
    )
    make_topomap_from_condition(
        cond_means_spn,
        condition_label="in noise",
        title="SPN in noise (-200 to 0 ms)",
        out_path=figures_dir / "Fig2b_SPN_topomap_in_noise.png"
    )

    pivot_diff = (
        mean_cond_spn.pivot_table(index=["Participant", "Electrode"], columns="Condition", values="Mean_uV")
        .reset_index()
    )
    if {"in silence", "in noise"}.issubset(pivot_diff.columns):
        pivot_diff["Mean_uV"] = pivot_diff["in noise"] - pivot_diff["in silence"]
        diff_mean = pivot_diff.groupby("Electrode", as_index=False)["Mean_uV"].mean()
        diff_mean["Condition"] = "noise_minus_silence"

        make_topomap_from_condition(
            diff_mean,
            condition_label="noise_minus_silence",
            title="SPN Noise - Silence (µV)",
            out_path=figures_dir / "Fig2c_SPN_topomap_difference.png"
        )

    spn_desc = roi_desc(spn_roi)
    cnv_desc = roi_desc(cnv_roi)

    save_table(spn_desc, tables_dir / "Table_Descriptive_ROI_ByCondition_SPN.csv")
    save_table(cnv_desc, tables_dir / "Table_Descriptive_ROI_ByCondition_CNV.csv")

    plot_roi_condition_panels(
        spn_desc,
        out_path=figures_dir / "Fig3_SPN_roi_panels_main.png",
        roi_order=["Frontal", "Central", "Parietal", "Occipital"],
        title="Mean SPN by ROI and condition"
    )

    plot_roi_condition_panels(
        cnv_desc,
        out_path=figures_dir / "Supp_Fig_CNV_roi_panels.png",
        roi_order=["Frontal", "Central", "Parietal", "Occipital"],
        title="Mean CNV by ROI and condition"
    )

    notes = [
        "Primary SPN analysis uses NON-INTERPOLATED data.",
        "CNV is included as a response-locked control analysis.",
        "Primary inference is ROI x Hemisphere LMM without Sex covariate.",
        "Sex-adjusted models are supplementary only.",
        "Cluster-based inferential analyses are not used.",
        "SPN topomaps are descriptive only and are shown in µV."
    ]
    notes_path = output_dir / "ANALYSIS_NOTES_SPN_CNV_REVISED.txt"
    with open(notes_path, "w", encoding="utf-8") as f:
        for line in notes:
            f.write(line + "\n")
    print(f"[Saved] {notes_path}")

    if args.speech:
        speech_csv = Path(args.speech)
        speech = pd.read_csv(speech_csv)
        print(f"[INFO] Using speech CSV: {speech_csv}")
    else:
        try:
            speech_csv = find_input_file(input_dir, "speech_accuracy")
            speech = pd.read_csv(speech_csv)
            print(f"[INFO] Using speech CSV: {speech_csv}")
        except Exception:
            speech = None
            print("[WARN] speech_accuracy.csv not found. Skipping speech-in-noise analyses.")

    if speech is not None:
        speech.columns = [c.strip() for c in speech.columns]

        need_speech_cols = {"Participant", "SNR_dB", "Accuracy", "n_trials", "n_correct"}
        miss = need_speech_cols - set(speech.columns)
        if miss:
            raise ValueError(f"Missing required columns in speech_accuracy.csv: {miss}")

        speech["Participant"] = (
            speech["Participant"]
            .astype(str)
            .str.strip()
            .str.replace("^P", "", regex=True)
            .str.zfill(2)
        )

        if run_qc is not None and "keep_primary_participant" in run_qc.columns:
            keep_part = run_qc[["Participant", "keep_primary_participant"]].drop_duplicates()
            speech = speech.merge(keep_part, on="Participant", how="left")
            speech = speech[speech["keep_primary_participant"] == True].copy()

        speech_desc = (
            speech.groupby("SNR_dB", as_index=False)
            .agg(
                n_participants=("Participant", "nunique"),
                mean_accuracy=("Accuracy", "mean"),
                sd_accuracy=("Accuracy", "std"),
                mean_n_correct=("n_correct", "mean"),
                sd_n_correct=("n_correct", "std"),
                n_trials=("n_trials", "mean")
            )
            .sort_values("SNR_dB", ascending=False)
        )
        save_table(speech_desc, tables_dir / "Table1_SpeechAccuracy_BySNR.csv")

        speech_0 = speech[speech["SNR_dB"] == 0].copy()
        speech_0 = (
            speech_0.groupby("Participant", as_index=False)
            .agg(
                accuracy_0dB=("Accuracy", "mean"),
                n_correct_0dB=("n_correct", "mean")
            )
        )

        speech_all = (
            speech.groupby("Participant", as_index=False)
            .agg(
                accuracy_allSNR=("Accuracy", "mean"),
                n_correct_allSNR=("n_correct", "mean")
            )
        )

        speech_diff = speech[speech["SNR_dB"].isin([-5, -10, -15])].copy()
        speech_diff = (
            speech_diff.groupby("Participant", as_index=False)
            .agg(
                accuracy_diffSNR=("Accuracy", "mean"),
                n_correct_diffSNR=("n_correct", "mean")
            )
        )

        speech_part = speech_all.merge(speech_0, on="Participant", how="outer")
        speech_part = speech_part.merge(speech_diff, on="Participant", how="outer")
        save_table(speech_part, tables_dir / "Supp_Table_SpeechAccuracy_ParticipantSummary.csv")

        spn_part = (
            spn_roi.groupby(["Participant", "Condition"], as_index=False)
            .agg(SPN_mean_uV=("Mean_uV", "mean"))
        )
        spn_wide = spn_part.pivot(index="Participant", columns="Condition", values="SPN_mean_uV").reset_index()

        if {"in silence", "in noise"}.issubset(spn_wide.columns):
            spn_wide["SPN_noise_minus_silence"] = spn_wide["in noise"] - spn_wide["in silence"]

        cnv_part = (
            cnv_roi.groupby(["Participant", "Condition"], as_index=False)
            .agg(CNV_mean_uV=("Mean_uV", "mean"))
        )
        cnv_wide = cnv_part.pivot(index="Participant", columns="Condition", values="CNV_mean_uV").reset_index()

        if {"in silence", "in noise"}.issubset(cnv_wide.columns):
            cnv_wide["CNV_noise_minus_silence"] = cnv_wide["in noise"] - cnv_wide["in silence"]

        spn_wide = spn_wide.rename(columns={"in silence": "SPN_silence", "in noise": "SPN_noise"})
        cnv_wide = cnv_wide.rename(columns={"in silence": "CNV_silence", "in noise": "CNV_noise"})

        participant_summary = speech_part.merge(spn_wide, on="Participant", how="inner")
        participant_summary = participant_summary.merge(cnv_wide, on="Participant", how="inner")
        save_table(participant_summary, tables_dir / "Supp_Table_ParticipantSummary_SPN_CNV_Speech.csv")

        corr_rows = []
        predictors = [
            ("accuracy_0dB", "Accuracy at 0 dB"),
            ("accuracy_allSNR", "Accuracy across all SNRs"),
            ("accuracy_diffSNR", "Accuracy at difficult SNRs")
        ]
        outcomes = [
            ("SPN_noise_minus_silence", "SPN noise-silence"),
            ("SPN_noise", "SPN in noise"),
            ("CNV_noise_minus_silence", "CNV noise-silence"),
            ("CNV_noise", "CNV in noise")
        ]

        for px, pxname in predictors:
            for oy, oyname in outcomes:
                if px in participant_summary.columns and oy in participant_summary.columns:
                    corr_rows.append(corr_row(participant_summary, px, oy, pxname, oyname))

        corr_tbl = pd.DataFrame(corr_rows)
        save_table(corr_tbl, tables_dir / "Supp_Table_Exploratory_SpeechBrain_Relations.csv")

        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        tmp = speech[speech["SNR_dB"] == 0].copy()

        if len(tmp) > 0:
            vals = tmp.groupby("Participant")["Accuracy"].mean().values
            rng = np.random.default_rng(42)
            jitter = rng.normal(0, 0.04, size=len(vals))
            ax.scatter(np.ones(len(vals)) + jitter, vals, color="black", alpha=0.85, s=30)

            mu = np.mean(vals)
            if len(vals) >= 2:
                se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                tcrit = student_t.ppf(1 - 0.05 / 2, len(vals) - 1)
                ci_lo = mu - tcrit * se
                ci_hi = mu + tcrit * se
                ax.hlines(mu, 0.82, 1.18, color="black", linewidth=2.0)
                ax.vlines(1.0, ci_lo, ci_hi, color="black", linewidth=1.5)

            ax.set_xlim(0.7, 1.3)
            ax.set_xticks([1.0])
            ax.set_xticklabels(["0 dB"])
            ax.set_ylabel("Accuracy")
            ax.set_title("Speech accuracy at 0 dB SNR")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            save_fig(fig, figures_dir / "Supp_Fig_SpeechAccuracy_0dB.png", dpi=300, tight=True)

    print("\n=== Analysis complete ===")


if __name__ == "__main__":
    main()
