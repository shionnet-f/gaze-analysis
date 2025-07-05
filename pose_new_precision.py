import pandas as pd
import numpy as np
import os

# モニター情報
resolution_x = 1920
resolution_y = 1080
monitor_width_cm = 47.6
monitor_height_cm = 26.8
viewing_distance_cm = 70

# summaryファイル名
summary_file = "./exported_csv/pose_df/pose_fix_df/precision_summary.csv"

# 対象の被験者19名分
for subject_id in range(1, 20):
    # 各被験者の実験課題3回分
    for experiment_id in range(1, 4):
        # ファイルパス
        file_name = f"./exported_csv/pose_df/pose_fix_df/pose_fix_df_id{subject_id:03}-{experiment_id:03}.csv"
        print(f"▶ 処理中: {file_name}")

        if not os.path.exists(file_name):
            # ファイルがない場合
            print("⚠️ ファイルが存在しません。NaNを記録します。")
            summary_row = pd.DataFrame([{
                "subject_id": subject_id,
                "experiment_id": experiment_id,
                "fixation_count": 0,
                "center_x_norm": np.nan,
                "center_y_norm": np.nan,
                "rms_norm": np.nan,
                "rms_deg": np.nan
            }])
        else:
            # ファイルはあるので読み込みを試みる
            try:
                df = pd.read_csv(file_name)
            except pd.errors.EmptyDataError:
                print("⚠️ 空ファイルです。NaNを記録します。")
                df = pd.DataFrame()

            if df.empty:
                # データが無い場合
                print("⚠️ データが空です。NaNを記録します。")
                summary_row = pd.DataFrame([{
                    "subject_id": subject_id,
                    "experiment_id": experiment_id,
                    "fixation_count": 0,
                    "center_x_norm": np.nan,
                    "center_y_norm": np.nan,
                    "rms_norm": np.nan,
                    "rms_deg": np.nan
                }])
            else:
                fixation_count = len(df)

                # 正規化座標の重心
                center_x = df["x_mean_norm"].mean()
                center_y = df["y_mean_norm"].mean()

                # RMS距離(正規化座標)
                df["distance_norm"] = np.sqrt(
                    (df["x_mean_norm"] - center_x) ** 2 + (df["y_mean_norm"] - center_y) ** 2
                )
                rms_norm = np.sqrt(np.mean(df["distance_norm"] ** 2))

                # ピクセル変換
                cm_per_pixel_x = monitor_width_cm / resolution_x
                cm_per_pixel_y = monitor_height_cm / resolution_y

                df["x_px"] = df["x_mean_norm"] * resolution_x
                df["y_px"] = df["y_mean_norm"] * resolution_y
                center_x_px = center_x * resolution_x
                center_y_px = center_y * resolution_y

                df["x_cm"] = (df["x_px"] - resolution_x / 2) * cm_per_pixel_x
                df["y_cm"] = (df["y_px"] - resolution_y / 2) * cm_per_pixel_y
                center_x_cm = (center_x_px - resolution_x / 2) * cm_per_pixel_x
                center_y_cm = (center_y_px - resolution_y / 2) * cm_per_pixel_y

                df["distance_cm"] = np.sqrt(
                    (df["x_cm"] - center_x_cm) ** 2 + (df["y_cm"] - center_y_cm) ** 2
                )
                df["distance_deg"] = np.degrees(
                    np.arctan(df["distance_cm"] / viewing_distance_cm)
                )
                rms_deg = np.sqrt(np.mean(df["distance_deg"] ** 2))

                # summaryデータ
                summary_row = pd.DataFrame([{
                    "subject_id": subject_id,
                    "experiment_id": experiment_id,
                    "fixation_count": fixation_count,
                    "center_x_norm": center_x,
                    "center_y_norm": center_y,
                    "rms_norm": rms_norm,
                    "rms_deg": rms_deg
                }])

        # summaryファイルに追記
        if os.path.exists(summary_file):
            summary_row.to_csv(summary_file, mode="a", index=False, header=False)
        else:
            summary_row.to_csv(summary_file, index=False, header=True)

print("✅ 全処理が完了しました。")
