import cv2
import numpy as np
import pandas as pd
import os

summary_df = pd.DataFrame(columns=["id", "trial", "left_eye", "right_eye", "nose", "mouth", "outside"])

fix_type="IVT"

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

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_cyan = np.array([80, 50, 50])
            upper_cyan = np.array([100, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_cyan, upper_cyan)
            kernel = np.ones((3, 3), np.uint8)
            mask_clean = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

            # 輪郭抽出
            contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 面積の大きい輪郭から上位4つをAOIとして採用
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

            aoi_masks = []
            aoi_counts = []

            for i, contour in enumerate(sorted_contours):
                # AOIごとのマスク作成
                mask = np.zeros(mask_clean.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                aoi_masks.append(mask)

                # 注視点カウント（ピクセル内包判定）
                count = 0
                for _, row in fixation_df.iterrows():
                    x, y = int(row['x_px']), int(row['y_px'])
                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                        if mask[y, x] == 255:
                            count += 1
                aoi_counts.append(count)

            # AOI情報まとめ（重心・ラベル・カウント）
            aoi_info = []

            overlay = img.copy()

            # AOIマスクを赤で半透明に重ねる
            for mask in aoi_masks:
                red_mask = np.zeros_like(img)
                red_mask[mask == 255] = [0, 0, 255]
                overlay = cv2.addWeighted(overlay, 1.0, red_mask, 0.4, 0)

            # 注視点を緑で描く
            for _, row in fixation_df.iterrows():
                cv2.circle(overlay, (int(row['x_px']), int(row['y_px'])), 4, (0, 255, 0), -1)

            for contour, count in zip(sorted_contours, aoi_counts):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    aoi_info.append({'contour': contour, 'count': count, 'cx': cx, 'cy': cy})

            # === 並び替えとラベル付け ===
            aoi_info_sorted = sorted(aoi_info, key=lambda d: d['cy'])
            eyes = sorted(aoi_info_sorted[:2], key=lambda d: d['cx'])
            nose = aoi_info_sorted[2]
            mouth = aoi_info_sorted[3]

            # === ラベル付きAOI ===
            aoi_labeled = [
                {'label': 'left_eye',  'count': eyes[0]['count'], 'cx': eyes[0]['cx'], 'cy': eyes[0]['cy']},
                {'label': 'right_eye', 'count': eyes[1]['count'], 'cx': eyes[1]['cx'], 'cy': eyes[1]['cy']},
                {'label': 'nose',      'count': nose['count'],    'cx': nose['cx'],    'cy': nose['cy']},
                {'label': 'mouth',     'count': mouth['count'],   'cx': mouth['cx'],   'cy': mouth['cy']}
            ]

            # === AOI外の注視数を計算 ===
            inside_total = sum([aoi['count'] for aoi in aoi_labeled])
            outside_count = total_fix - inside_total

            # === outside の行を追加 ===
            aoi_labeled.append({
                'label': 'outside',
                'count': outside_count,
                'cx': np.nan,
                'cy': np.nan
            })


            # === DataFrame化 ===
            df_result = pd.DataFrame(aoi_labeled)
            df_result.insert(1, 'total_fixation', total_fix)

            # ファイル名
            output_path = f"exported_csv/fixation_counts/{fix_type}_triangle/result_df_{subject_id:03}-{experiment_id:03}-{int(trial_num)}.csv"
            # 保存（index=Falseでインデックス列を除く）
            df_result.to_csv(output_path, index=False, float_format="%.6f", encoding="utf-8-sig")

            # === summary_df に1行追加 ===
            summary_row = {
                "id": f"{subject_id:03}-{experiment_id:03}",
                "trial": trial_num
            }
            for label in ["left_eye", "right_eye", "nose", "mouth", "outside"]:
                count = df_result.loc[df_result["label"] == label, "count"].values
                summary_row[label] = int(count[0]) if len(count) > 0 else 0

            summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)

            # === summary_df 保存 ===
            summary_path = f"exported_csv/fixation_counts/{fix_type}_triangle/summary_all_trials.csv"
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
