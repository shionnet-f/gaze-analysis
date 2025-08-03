import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

# === 設定 ===
input_dir = "./exported_csv/fixation_IVT_split"
output_dir = "./exported_csv/fixation_IVT_split/fixation_with_AOI_labels"
aoi_dir = "../face_aoi_project/output_aoi_triangle"
interval = "0.0-2.0"

screen_width, screen_height = 1920, 1080  # ピクセル変換用

os.makedirs(output_dir, exist_ok=True)

for subject_id in range(1, 20):
    for experiment_id in range(1, 4):
        for trial_num in range(8):
            # === ファイルパス定義 ===
            fname = f"fix_df_{subject_id:03}-{experiment_id:03}-{trial_num}_{interval}.csv"
            fixation_csv_path = os.path.join(input_dir, fname)
            aoi_image_path = os.path.join(aoi_dir, f"{experiment_id}-{trial_num + 1}.jpg")
            output_csv_path = os.path.join(output_dir, f"fixation_AOI_label_{subject_id:03}-{experiment_id:03}-{trial_num}.csv")

            # AOI画像存在チェック
            if not os.path.exists(aoi_image_path):
                print(f"⚠️ AOI画像が存在しません: {aoi_image_path}")
                continue

            # 注視ファイルが存在しない → no_fixation と記録
            if not os.path.exists(fixation_csv_path):
                pd.DataFrame([{
                    "subject_id": subject_id,
                    "experiment_id": experiment_id,
                    "trial": trial_num,
                    "SOI_label": "no_fixation"
                }]).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
                continue

            fixation_df = pd.read_csv(fixation_csv_path)
            if fixation_df.empty:
                pd.DataFrame([{
                    "subject_id": subject_id,
                    "experiment_id": experiment_id,
                    "trial": trial_num,
                    "SOI_label": "no_fixation"
                }]).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
                continue

            # === AOI画像とマスク作成 ===
            img = cv2.imread(aoi_image_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_cyan = np.array([80, 50, 50])
            upper_cyan = np.array([100, 255, 255])
            mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

            plt.imshow(img)
            plt.title(f"{experiment_id}-{trial_num + 1}")
            plt.axis('off')
            plt.show()

            # === 各注視点をAOI内か判定 ===
            labels = []
            for _, row in fixation_df.iterrows():
                if pd.isna(row["x_mean_px"]) or pd.isna(row["y_mean_px"]):
                    labels.append("outside")
                    continue
                
                x = int(np.clip(row["x_mean_px"], 0, screen_width - 1))
                y = int(np.clip(row["y_mean_px"], 0, screen_height - 1))

                if mask[y, x] == 255:
                    labels.append("inside")
                else:
                    labels.append("outside")


            fixation_df["AOI_label"] = labels
            fixation_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
            print(f"✅ 保存: {output_csv_path}")
