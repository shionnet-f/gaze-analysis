import pandas as pd
import numpy as np
import math
import os

# 出力用リスト
results = []

viewing_distance_cm = 70       # モニタまでの距離
screen_width_cm = 47.6           # モニタの横幅
screen_height_cm = 26.8          # モニタの高さ
diag_length_cm = math.sqrt(screen_width_cm**2 + screen_height_cm**2)

# 理想注視位置（中央）
ideal_x_deg = 0.0
ideal_y_deg = 0.0

# 対象の被験者19名分
for subject_id in range(1,20):
    # 各被験者の実験課題3回分
    for experiment_id in range(1, 4):
        
        file_path = f"./exported_csv/pose_df/pose_fix_df/pose_fix_df_id{subject_id:03}-{experiment_id:03}.csv"

        if not os.path.exists(file_path):
            print(f"ファイルが存在しません: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)

            # 空ファイル対応
            if df.empty:
                print(f"ファイルは空です: {file_path}")
                continue

        except pd.errors.EmptyDataError:
            print(f"EmptyDataError: ファイルにデータがありません: {file_path}")
            continue

        # 距離計算
        df["distance_deg"] = np.sqrt(
            (df["x_mean_deg"] - ideal_x_deg)**2 +
            (df["y_mean_deg"] - ideal_y_deg)**2
        )

        mean_distance_deg = df["distance_deg"].mean()

        results.append({
            "subject_id": subject_id,
            "experiment_id": experiment_id,
            "fixation_count": len(df),
            "mean_distance_deg": mean_distance_deg
        })

results_df = pd.DataFrame(results)
print(results_df)