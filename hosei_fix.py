import pandas as pd
import numpy as np
import os

# === 設定 ===
summary_path = "./exported_csv/pose_df/pose_fix_df/precision_summary.csv"
input_dir = "./exported_csv/fixation_IVT"
output_dir = "./exported_csv/fixation_IVT"
os.makedirs(output_dir, exist_ok=True)

# === summaryを読み込む ===
summary_df = pd.read_csv(summary_path)

# 補正対象 (正確度NG & 精密度OK)
targets_df = summary_df[
    (summary_df["accuracy_bool"] == 0) &
    (summary_df["precision_bool"] == 1)
]

print(f"補正対象の実験課題数: {len(targets_df)}")

# === モニター解像度 ===
resolution_x = 1920
resolution_y = 1080

# === 各対象実験課題の試行を処理 ===
for _, row in targets_df.iterrows():
    subject_id = int(row["subject_id"])
    experiment_id = int(row["experiment_id"])

    # 補正量を課題単位で計算
    center_x = row["center_x_norm"]
    center_y = row["center_y_norm"]
    dx = 0.5 - center_x
    dy = 0.5 - center_y

    print(f"▶ subject {subject_id}, experiment {experiment_id}: dx={dx:.4f}, dy={dy:.4f}")

    any_file_found = False

    for trial in range(8):
        filename = f"{input_dir}/fix_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"
        output_filename = f"{output_dir}/fix_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"

        if not os.path.exists(filename):
            print(f"  ⭕ 試行{trial}: ファイルが見つかりません。スキップします。")
            continue

        any_file_found = True

        df = pd.read_csv(filename)

        if df.empty:
            print(f"  ⭕ 試行{trial}: データが空です。スキップします。")
            continue

        # === 正規化座標を補正 ===
        df["x_mean_norm_corrected"] = df["x_norm"] + dx
        df["y_mean_norm_corrected"] = df["y_norm"] + dy
        df["x_mean_norm_corrected"] = df["x_mean_norm_corrected"].clip(0, 1)
        df["y_mean_norm_corrected"] = df["y_mean_norm_corrected"].clip(0, 1)

        # === ピクセル座標も補正 ===
        df["x_px_raw"] = df["x_px"]
        df["y_px_raw"] = df["y_px"]

        df["x_px"] = df["x_mean_norm_corrected"] * resolution_x
        df["y_px"] = df["y_mean_norm_corrected"] * resolution_y

        # 保存
        df.to_csv(output_filename, index=False, encoding="utf-8-sig")
        print(f"  ✅ 試行{trial}: 補正して保存しました。")

    if not any_file_found:
        print(f"⚠️ 注意: subject {subject_id} experiment {experiment_id} は対象試行が1つも存在しませんでした。")

print("✅ 全補正処理が完了しました。")
