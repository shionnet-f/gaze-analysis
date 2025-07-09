import cv2
import numpy as np
import pandas as pd
import os

# === 出力先フォルダ ===
output_dir = "exported_csv/fixation_counts/IVT/fixation_with_AOI_labels"
os.makedirs(output_dir, exist_ok=True)

fix_type = "IVT"

# === 被験者・実験課題・試行ループ ===
for subject_id in range(1, 20):
    for experiment_id in range(1, 4):
        for trial_num in range(8):

            # === 入力ファイルパス ===
            fixation_csv_path = f"exported_csv/fixation_{fix_type}/fix_df_{subject_id:03}-{experiment_id:03}-{trial_num}.csv"
            aoi_image_path = f"../face_aoi_project/output_aoi/{experiment_id}-{trial_num + 1}.jpg"

            # === 出力ファイル名 ===
            output_csv_path = os.path.join(
                output_dir, f"fixation_AOI_label_{subject_id:03}-{experiment_id:03}-{trial_num}.csv"
            )

            # === ファイル存在チェック ===
            if not os.path.exists(fixation_csv_path):
                print(f"⚠️ 注視CSVが存在しません: {fixation_csv_path}")
                continue

            if not os.path.exists(aoi_image_path):
                print(f"⚠️ AOI画像が存在しません: {aoi_image_path}")
                continue

            # === データ読み込み ===
            fixation_df = pd.read_csv(fixation_csv_path)
            if fixation_df.empty:
                print(f"⚠️ データが空です: {fixation_csv_path}")
                continue

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
            if len(contours) < 4:
                print(f"⚠️ AOI輪郭が4つ未満です: {aoi_image_path}")
                continue

            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

            aoi_masks = []
            for contour in sorted_contours:
                mask = np.zeros(mask_blue.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                aoi_masks.append(mask)

            # AOI位置情報
            aoi_info = []
            for contour in sorted_contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    aoi_info.append({"contour": contour, "cx": cx, "cy": cy})
                else:
                    cx, cy = 0, 0
                    aoi_info.append({"contour": contour, "cx": cx, "cy": cy})

            # AOIラベル決定
            aoi_info_sorted = sorted(aoi_info, key=lambda d: d["cy"])
            eyes = sorted(aoi_info_sorted[:2], key=lambda d: d["cx"])
            nose = aoi_info_sorted[2]
            mouth = aoi_info_sorted[3]

            aoi_labels = ["left_eye", "right_eye", "nose", "mouth"]

            # === 全注視点にラベル付け ===
            labels_per_fixation = []
            for _, row in fixation_df.iterrows():
                x, y = int(row["x_px"]), int(row["y_px"])
                label = "outside"
                for i, mask in enumerate(aoi_masks):
                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                        if mask[y, x] == 255:
                            label = aoi_labels[i]
                            break
                labels_per_fixation.append(label)

            # === ラベル追加
            fixation_df["AOI_label"] = labels_per_fixation

            # === 保存
            fixation_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

            print(f"✅ 完了: {output_csv_path}")
