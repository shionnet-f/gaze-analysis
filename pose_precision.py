import pandas as pd
import math
from math import atan, sqrt

viewing_distance_cm = 70       # モニタまでの距離
screen_width_cm = 47.6           # モニタの横幅
screen_height_cm = 26.8          # モニタの高さ
diag_length_cm = math.sqrt(screen_width_cm**2 + screen_height_cm**2)

# 対象の被験者19名分
for subject_id in range(1,20):
    # 各被験者の実験課題3回分
    for experiment_id in range(1, 4):
        
        pose1_df = pd.read_csv(f"./exported_csv/pose_df/pose1_df_id{subject_id:03}-{experiment_id:03}.csv")
        pose2_df = pd.read_csv(f"./exported_csv/pose_df/pose2_df_id{subject_id:03}-{experiment_id:03}.csv")
        
 
        def evaluate_precision(df, pose_name):
            total_len = len(df)

            # validity_sum == 2 のデータのみを使う
            valid_df = df[df["validity_sum"] == 2].copy()

            std_x = valid_df["mean_x"].std()
            std_y = valid_df["mean_y"].std()

            precision_rms = sqrt(std_x**2 + std_y**2)

            precision_cm = precision_rms * diag_length_cm
            precision_deg = atan(precision_cm / viewing_distance_cm)



            # サマリデータの返却
            return {
                "subject_id": subject_id,
                "task_id": experiment_id,
                "pose": pose_name,
                "pose_length": total_len,
                "valid_data_count": len(valid_df),
                "precision_rms": precision_rms,
                "precision_cm": precision_cm,
                "precision_deg": precision_deg
            }
            
        summary_list = []
        summary_list.append(evaluate_precision(pose1_df, "pose1"))
        summary_list.append(evaluate_precision(pose2_df, "pose2"))
        
        summary_df = pd.DataFrame(summary_list)
        
        summary_df.to_csv(f"./exported_csv/pose_df/precision_summary_all.csv", mode='a', index=False, header=False)
                   