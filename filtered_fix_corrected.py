import pandas as pd
import numpy as np
import os

# === 設定 ===
summary_path = "./exported_csv/pose_df/pose_fix_df/precision_summary.csv"
input_dir = "./exported_csv/filtered_IVT"
output_dir = "./exported_csv/filtered_IVT_corrected"
os.makedirs(output_dir, exist_ok=True)

# === モニター解像度 ===
resolution_x = 1920
resolution_y = 1080

# === summaryを読み込む ===
summary_df = pd.read_csv(summary_path)

# 精密度OKかつ正確度NGの行のみ抽出
targets_df = summary_df[
    (summary_df["accuracy_bool"] == 0) &
    (summary_df["precision_bool"] == 1)
]

print(f"補正対象の実験課題数: {len(targets_df)}")

# === 各実験課題を処理 ===
for _, row in targets_df.iterrows():
    subject_id = int(row["subject_id"])
    experiment_id = int(row["experiment_id"])

    # 中心からの補正量を計算（0.5に揃える）
    dx = 0.5 - row["center_x_norm"]
    dy = 0.5 - row["center_y_norm"]
    print(f"▶ subject {subject_id}, experiment {experiment_id}: dx={dx:.4f}, dy={dy:.4f}")

    for trial in range(8):
        filename = f"{input_dir}/filtered_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"
        output_filename = f"{output_dir}/filtered_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"

        if not os.path.exists(filename):
            print(f"  ⭕ 試行{trial}: ファイルが存在しません。スキップ。")
            continue

        df = pd.read_csv(filename)
        if df.empty:
            print(f"  ⭕ 試行{trial}: データが空です。スキップ。")
            continue

        # === 補正前の値を保存 ===
        df["filtered_x_raw"] = df["filtered_x"]
        df["filtered_y_raw"] = df["filtered_y"]

        # === filtered_x/y を中心補正して上書き ===
        df["filtered_x"] = (df["filtered_x_raw"] + dx).clip(0, 1)
        df["filtered_y"] = (df["filtered_y_raw"] + dy).clip(0, 1)

        # === ピクセル座標を補正後の値で再計算 ===
        df["x_px"] = df["filtered_x"] * resolution_x
        df["y_px"] = df["filtered_y"] * resolution_y

        # === 保存 ===
        df.to_csv(output_filename, index=False, encoding="utf-8-sig")
        print(f"  ✅ 試行{trial}: 補正して保存しました。")

print("✅ filtered_IVT_corrected 全補正が完了しました。")
