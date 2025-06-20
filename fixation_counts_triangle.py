import cv2
import numpy as np
import pandas as pd
import os

summary_df = pd.DataFrame(columns=["id", "trial", "left_eye", "right_eye", "nose", "mouth", "outside"])

fix_type="IDT"

# 対象の被験者19名分
for subject_id in range(1,20): 
    # 各被験者の実験課題3回分
    for experiment_id in range(1, 4):
        for trial_num in range(8):

            fixation_path = f"exported_csv/fixation_{fix_type}/fix_df_{subject_id:03}-{experiment_id:03}-{trial_num}.csv"
            # === fixation_df（CSV）の存在チェック ===
            if not os.path.exists(fixation_path):
                print(f" fixation_df が存在しません: subject {subject_id:03}, experiment {experiment_id:03}, trial {trial_num}")

                summary_row = {
                    "id": f"{subject_id:03}-{experiment_id:03}",
                    "trial": trial_num,
                    "left_eye": "empty",
                    "right_eye": "empty",
                    "nose": "empty",
                    "mouth": "empty",
                    "outside": "empty"
                }

                summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
                continue

            # 顔画像の読み込み
            img = cv2.imread(f'../face_aoi_project/output_aoi_triangle/{experiment_id}-{trial_num + 1}.jpg')
            fixation_df = pd.read_csv(fixation_path)



            total_fix = len(fixation_df)

            # AOI検出
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_cyan = np.array([80, 50, 50])
            upper_cyan = np.array([100, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_cyan, upper_cyan)
            kernel = np.ones((3, 3), np.uint8)
            mask_clean = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

            # 輪郭抽出
            contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 最大のAOIのみ使用（1つだけ）
            contour = max(contours, key=cv2.contourArea)
            mask = np.zeros(mask_clean.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            aoi_masks = []
            aoi_counts = []

            # AOI内カウント
            count_in = 0
            for _, row in fixation_df.iterrows():
                x, y = int(row["x_px"]), int(row["y_px"])
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] == 255:
                    count_in += 1

            count_out = total_fix - count_in

            # === 保存 ===
            output_path = f"exported_csv/fixation_counts/{fix_type}_triangle/aoi1_df_{subject_id:03}-{experiment_id:03}-{trial_num}.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            result_df = pd.DataFrame([{
                "label": "aoi",
                "count": count_in,
                "total_fixation": total_fix
            }, {
                "label": "outside",
                "count": count_out,
                "total_fixation": total_fix
            }])
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

            # === summary_df に追加 ===
            summary_df = pd.concat([summary_df, pd.DataFrame([{
                "id": f"{subject_id:03}-{experiment_id:03}",
                "trial": trial_num,
                "aoi": count_in,
                "outside": count_out
            }])], ignore_index=True)

# 最後にサマリー保存
summary_df.to_csv(f"exported_csv/fixation_counts/{fix_type}_triangle/summary_aoi1.csv", index=False, encoding='utf-8-sig')