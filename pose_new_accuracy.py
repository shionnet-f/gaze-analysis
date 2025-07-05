import pandas as pd
import numpy as np
import os

summary_records = []
RADIUS_PX = 75
center_x = 1920 / 2
center_y = 1080 / 2

for subject_id in range(1, 20):
    for experiment_id in range(1, 4):
        # ファイル名
        filename = f"./exported_csv/pose_df/pose_fix_df/pose_fix_df_id{subject_id:03}-{experiment_id:03}.csv"
        total_count = 0
        inside_count = 0
        outside_count = 0
        inside_ratio = np.nan

        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)

                if len(df) > 0:
                    df["distance_px"] = np.sqrt(
                        (df["x_mean_px"] - center_x) ** 2 + (df["y_mean_px"] - center_y) ** 2
                    )
                    within_radius = df["distance_px"] <= RADIUS_PX
                    total_count = len(df)
                    inside_count = within_radius.sum()
                    outside_count = total_count - inside_count
                    inside_ratio = inside_count / total_count

            except pd.errors.EmptyDataError:
                # 中身が完全に空の場合も件数0で記録
                pass

        summary_records.append({
            "subject_id": subject_id,
            "experiment_id": experiment_id,
            "total_fixations": total_count,
            "inside_count": inside_count,
            "outside_count": outside_count,
            "inside_ratio": inside_ratio
        })

summary_df = pd.DataFrame(summary_records)


summary_df.to_csv("./exported_csv/pose_df/pose_fix_df/fixation_radius_summary.csv", index=False, encoding="utf-8-sig")
