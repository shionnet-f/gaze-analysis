import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 設定 ===
screen_w, screen_h = 1920, 1080
filtered_dir = "./exported_csv/filtered_IVT_corrected"
sampling_dir = "./exported_csv/sampling_df_plus03_08"
flags_path = "./exported_csv/condition_flags.csv"
precision_path = "./exported_csv/pose_df/pose_fix_df/precision_summary.csv"
output_dir = "./exported_csv/heatmaps_per_image"
os.makedirs(output_dir, exist_ok=True)

# === 品質条件を満たす subject × experiment 一覧を取得 ===
flags_df = pd.read_csv(flags_path)
precision_df = pd.read_csv(precision_path)

valid_pairs = flags_df[flags_df["all_ok"] == True].merge(
    precision_df[precision_df["precision_bool"] == 1],
    on=["subject_id", "experiment_id"]
)[["subject_id", "experiment_id"]].drop_duplicates()

# === ヒートマップデータを画像単位で集約 ===
heatmap_data = {}  # {(experiment_id, trial): [ (x, y), ... ]}

for _, row in valid_pairs.iterrows():
    subject_id = row["subject_id"]
    experiment_id = row["experiment_id"]

    # samplingファイル読み込み
    sampling_path = f"{sampling_dir}/sampling_df_id{subject_id:03}-{experiment_id:03}_plus03_08.csv"
    if not os.path.exists(sampling_path):
        continue

    try:
        sampling_df = pd.read_csv(sampling_path)
    except:
        continue

    for _, trial_row in sampling_df.iterrows():
        trial = int(trial_row["trial"])
        t_start = trial_row["start_sec"]
        t_end = t_start + 2.0

        filtered_path = f"{filtered_dir}/filtered_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"
        if not os.path.exists(filtered_path):
            continue

        try:
            df = pd.read_csv(filtered_path)
        except:
            continue

        if "epoch_sec" not in df.columns or df.empty:
            continue

        df = df[(df["epoch_sec"] >= t_start) & (df["epoch_sec"] <= t_end)]
        df = df.dropna(subset=["filtered_x", "filtered_y"])
        if df.empty:
            continue

        df["x_px"] = (df["filtered_x"] * screen_w).clip(0, screen_w - 1)
        df["y_px"] = (df["filtered_y"] * screen_h).clip(0, screen_h - 1)

        key = (experiment_id, trial)
        if key not in heatmap_data:
            heatmap_data[key] = []

        heatmap_data[key].extend(zip(df["x_px"], df["y_px"]))

# === 画像ごとにヒートマップを描画・保存 ===
for (exp_id, trial), coords in heatmap_data.items():
    heatmap = np.zeros((screen_h, screen_w))

    for x, y in coords:
        xi, yi = int(x), int(y)
        heatmap[yi, xi] += 1

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap,
        cmap="jet",
        cbar=False,
        xticklabels=False,
        yticklabels=False
    )
    plt.gca().invert_yaxis()
    plt.title(f"Heatmap exp{exp_id:03}-trial{trial}")
    out_path = os.path.join(output_dir, f"heatmap_{exp_id:03}-{trial}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
