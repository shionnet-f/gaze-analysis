import os
import pandas as pd
import numpy as np
import math

# === パラメータ ===
monitor_width_cm = 47.6
monitor_height_cm = 26.8
resolution = (1920, 1080)
viewer_distance_cm = 60.0
cm_per_pixel_x = monitor_width_cm / resolution[0]
cm_per_pixel_y = monitor_height_cm / resolution[1]

# === パス設定 ===
summary_path = "./exported_csv/pose_df/pose_fix_df/precision_summary.csv"
filtered_input_dir = "./exported_csv/filtered_IVT"
fixation_input_dir = "./exported_csv/fixation_IVT"
filtered_output_dir = "./exported_csv/filtered_IVT_corrected"
fixation_output_dir = "./exported_csv/fixation_IVT_corrected"
os.makedirs(filtered_output_dir, exist_ok=True)
os.makedirs(fixation_output_dir, exist_ok=True)

# === 対象試行の抽出 ===
summary_df = pd.read_csv(summary_path)
targets_df = summary_df[
    (summary_df["accuracy_bool"] == 0) & (summary_df["precision_bool"] == 1)
]

# === 補正・変換関数（正規化座標ベース） ===
def correct_and_convert_norm(df, x_col, y_col, dx, dy):
    df = df.copy()
    df[f"{x_col}_raw"] = df[x_col]
    df[f"{y_col}_raw"] = df[y_col]

    df[x_col] = (df[f"{x_col}_raw"] + dx).clip(0, 1)
    df[y_col] = (df[f"{y_col}_raw"] + dy).clip(0, 1)

    df["x_px"] = df[x_col] * resolution[0]
    df["y_px"] = df[y_col] * resolution[1]

    x_cm = (df[x_col] - 0.5) * resolution[0] * cm_per_pixel_x
    y_cm = (df[y_col] - 0.5) * resolution[1] * cm_per_pixel_y
    df["x_deg"] = np.degrees(np.arctan2(x_cm, viewer_distance_cm))
    df["y_deg"] = np.degrees(np.arctan2(y_cm, viewer_distance_cm))
    return df

# === 補正処理ループ ===
for _, row in targets_df.iterrows():
    subject_id = int(row["subject_id"])
    experiment_id = int(row["experiment_id"])
    dx = 0.5 - row["center_x_norm"]
    dy = 0.5 - row["center_y_norm"]
    print(f"▶ subject {subject_id}, experiment {experiment_id}: dx={dx:.4f}, dy={dy:.4f}")

    for trial in range(8):
        # === 視線データ（filtered） ===
        filtered_path = f"{filtered_input_dir}/filtered_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"
        if os.path.exists(filtered_path):
            df = pd.read_csv(filtered_path)
            if not df.empty and "filtered_x" in df.columns:
                corrected = correct_and_convert_norm(df, "filtered_x", "filtered_y", dx, dy)
                output_path = f"{filtered_output_dir}/filtered_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"
                corrected.to_csv(output_path, index=False, encoding="utf-8-sig")
                print(f"  ✅ filtered 保存: {output_path}")

        # === 注視データ（fix_df） ===
        fix_path = f"{fixation_input_dir}/fix_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"
        if os.path.exists(fix_path):
            df = pd.read_csv(fix_path)
            if not df.empty and "x_mean_norm" in df.columns:
                corrected = correct_and_convert_norm(df, "x_mean_norm", "y_mean_norm", dx, dy)
                output_path = f"{fixation_output_dir}/fix_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"
                corrected.to_csv(output_path, index=False, encoding="utf-8-sig")
                print(f"  ✅ fixation 保存: {output_path}")
