import cv2
import numpy as np
import pandas as pd
import os

fix_type = "IVT"

# === 出力先フォルダ ==
output_dir = f"exported_csv/fixation_counts/{fix_type}_triangle/fixation_with_AOI_label_triangle"
os.makedirs(output_dir, exist_ok=True)

# === 被験者・実験課題・試行ループ ===
for subject_id in range(1, 20):
    for experiment_id in range(1, 4):
        for trial_num in range(8):

            fixation_csv_path = f"exported_csv/fixation_sliced_by_time/fixation_{subject_id:03}-{experiment_id:03}-{trial_num}_interval_0.0-2.0s.csv"
            aoi_image_path = f"../face_aoi_project/output_aoi_triangle/{experiment_id}-{trial_num + 1}.jpg"
            output_csv_path = os.path.join(
                output_dir, f"fixation_AOI_label_triangle_{subject_id:03}-{experiment_id:03}-{trial_num}.csv"
            )

            if not os.path.exists(fixation_csv_path):
                print(f"⚠️ 注視CSVが存在しません: {fixation_csv_path}")
                continue
            if not os.path.exists(aoi_image_path):
                print(f"⚠️ AOI画像が存在しません: {aoi_image_path}")
                continue

            try:
                fixation_df = pd.read_csv(fixation_csv_path)
            except pd.errors.EmptyDataError:
                print(f"⚠️ 空ファイル（ヘッダーなし）: {fixation_csv_path}")
                pd.DataFrame(columns=["AOI_label_triangle"]).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
                print(f"✅ 完了（空でも出力）: {output_csv_path}")
                continue  # ← ここでスキップ！
            


            img = cv2.imread(aoi_image_path)
            if img is None:
                print(f"⚠️ AOI画像が読み込めません: {aoi_image_path}")
                continue

            # === AOIマスク作成 ===
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_cyan = np.array([80, 50, 50])
            upper_cyan = np.array([100, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_cyan, upper_cyan)

            contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print(f"⚠️ AOI輪郭が見つかりません: {aoi_image_path}")
                continue

            main_contour = max(contours, key=cv2.contourArea)
            aoi_mask = np.zeros(mask_blue.shape, dtype=np.uint8)
            cv2.drawContours(aoi_mask, [main_contour], -1, 255, thickness=cv2.FILLED)

            # === 全注視点に「face_inside」 or 「outside」ラベル付け ===
            labels_per_fixation = []
            for _, row in fixation_df.iterrows():
                x, y = int(row["x_px"]), int(row["y_px"])
                if 0 <= x < aoi_mask.shape[1] and 0 <= y < aoi_mask.shape[0]:
                    label = "face_inside" if aoi_mask[y, x] == 255 else "outside"
                else:
                    label = "outside"
                labels_per_fixation.append(label)

            fixation_df["AOI_label_triangle"] = labels_per_fixation
            fixation_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
            print(f"✅ 完了: {output_csv_path}")
