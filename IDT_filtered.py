import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter1d

# IDT法による注視検出と可視化

# 対象の被験者19名分
for subject_id in range(1,20):
    # 各被験者の実験課題3回分
    for experiment_id in range(3, 4):
        # IDT法のパラメータ
        DISPERSION_THRESHOLD = 1.0  # deg
        DURATION_THRESHOLD_MS = 100  # ms

        eye_df = pd.read_csv(
            f"exported_csv/eye_df_id{subject_id:03}-{experiment_id:03}.csv"
        )
        sampling_df = pd.read_csv(
            f"exported_csv/sampling_df_id{subject_id:03}-{experiment_id:03}.csv"
        )

        # モニターサイズ(物理)
        monitor_width_cm = 47.6
        monitor_height_cm = 26.8
        # モニター解像度(px)
        monitor_resolution_px = (1920, 1080)
        # 視距離(cm)
        viewer_distance_cm = 70.0

        # cm/pxの変換係数
        cm_per_pixel_x = monitor_width_cm / monitor_resolution_px[0]
        cm_per_pixel_y = monitor_height_cm / monitor_resolution_px[1]

        # ディスプレイ中心を(0, 0)とするための変換
        eye_df["gx_centered"] = eye_df["gx"] - 0.5
        eye_df["gy_centered"] = eye_df["gy"] - 0.5

        # 中心(0,0)での物理距離変換
        eye_df["x_cm"] = (
            eye_df["gx_centered"] * monitor_resolution_px[0] * cm_per_pixel_x
        )
        eye_df["y_cm"] = (
            eye_df["gy_centered"] * monitor_resolution_px[1] * cm_per_pixel_y
        )

        # 視野角の計算
        eye_df["x_deg"] = np.degrees(np.arctan2(eye_df["x_cm"], viewer_distance_cm))
        eye_df["y_deg"] = np.degrees(np.arctan2(eye_df["y_cm"], viewer_distance_cm))

        # データの有効性
        eye_df["is_valid"] = eye_df["validity_sum"] > 1

        # 線形補完の処理
        def interpolate_missing(df, time_col="epoch_sec", max_gap_ms=100):
            df = df.copy()
            df["valid"] = df["is_valid"]
            df["interp_x"] = np.nan
            df["interp_y"] = np.nan

            # 有効データを代入
            df.loc[df["valid"], "interp_x"] = df.loc[df["valid"], "x_deg"]
            df.loc[df["valid"], "interp_y"] = df.loc[df["valid"], "y_deg"]

            # 内部のみ線形補完
            df["interp_x"] = df["interp_x"].interpolate(limit_area="inside")
            df["interp_y"] = df["interp_y"].interpolate(limit_area="inside")

            # 無効区間の連続ブロックを取得
            invalid_mask = ~df["valid"]
            group_id = (invalid_mask != invalid_mask.shift()).cumsum()
            invalid_blocks = df[invalid_mask].groupby(group_id)

            for _, block in invalid_blocks:
                if len(block) == 0:
                    continue
                t_start = block[time_col].iloc[0]
                t_end = block[time_col].iloc[-1]
                duration_ms = (t_end - t_start) * 1000
                if duration_ms > max_gap_ms:
                    # 100ms超えたら補完結果をNaNに戻す
                    df.loc[block.index, ["interp_x", "interp_y"]] = np.nan

            return df

        # ガウシアンフィルタをブロックごとに適用
        def apply_gaussian_filter_by_block(df, col_x="x_deg", col_y="y_deg", sigma=1.0):
            df = df.copy()
            df["valid"] = df["is_valid"]
            df["filtered_x"] = np.nan
            df["filtered_y"] = np.nan

            # 有効なデータ（NaNでない）だけを連続ブロックとして抽出
            valid_mask = df[col_x].notna() & df[col_y].notna()
            block_id = (valid_mask != valid_mask.shift()).cumsum()
            blocks = df[valid_mask].groupby(block_id)

            for _, block in blocks:
                idx = block.index
                smoothed_x = gaussian_filter1d(block[col_x], sigma=sigma)
                smoothed_y = gaussian_filter1d(block[col_y], sigma=sigma)
                df.loc[idx, "filtered_x"] = smoothed_x
                df.loc[idx, "filtered_y"] = smoothed_y

            return df
        
        # IDT法による注視検出
        def detect_fixations_idt(
            df,
            dispersion_threshold=DISPERSION_THRESHOLD,
            duration_threshold_ms=DURATION_THRESHOLD_MS,
        ):
            fixations = []
            timestamps = df["epoch_sec"].to_numpy()
            xs = df["filtered_x"].to_numpy()
            ys = df["filtered_y"].to_numpy()

            i = 0
            while i < len(df):
                if np.isnan(xs[i]) or np.isnan(ys[i]):
                    i += 1
                    continue

                window = [(xs[i], ys[i])]
                t_start = timestamps[i]
                j = i + 1

                while j < len(df):
                    if np.isnan(xs[j]) or np.isnan(ys[j]):
                        break

                    window.append((xs[j], ys[j]))
                    t_end = timestamps[j]
                    duration = (t_end - t_start) * 1000

                    x_vals, y_vals = zip(*window)
                    x_center = np.mean(x_vals)
                    y_center = np.mean(y_vals)
                    distances = np.sqrt(
                        (np.array(x_vals) - x_center) ** 2
                        + (np.array(y_vals) - y_center) ** 2
                    )

                    if np.max(distances) > dispersion_threshold:
                        break

                    j += 1

                # 注視条件を満たしていれば1つだけ追加
                if (timestamps[j - 1] - t_start) * 1000 >= duration_threshold_ms:
                    x_vals, y_vals = zip(*window)
                    fixations.append(
                        {
                            "start_time": t_start,
                            "end_time": timestamps[j - 1],
                            "duration_ms": (timestamps[j - 1] - t_start) * 1000,
                            "x_mean_deg": np.mean(x_vals),
                            "y_mean_deg": np.mean(y_vals),
                        }
                    )

                i = j  # 次の未使用インデックスに進む

            return pd.DataFrame(fixations)

        # 度数法からピクセル単位に変換
        def deg_to_px(x_deg, y_deg):
            x_cm = np.tan(np.radians(x_deg)) * viewer_distance_cm
            y_cm = np.tan(np.radians(y_deg)) * viewer_distance_cm

            x_px = (x_cm / cm_per_pixel_x) + (monitor_resolution_px[0] / 2)
            y_px = (y_cm / cm_per_pixel_y) + (monitor_resolution_px[1] / 2)

            return x_px, y_px

        all_fixations = []

        for _, row in sampling_df.iterrows():
            t_start = row["start_sec"]
            t_end = row["end_sec"]
            trial_num = row["trial"]

            df_trial = eye_df[
                (eye_df["epoch_sec"] >= t_start) & (eye_df["epoch_sec"] <= t_end)
            ]
            df_interp = interpolate_missing(df_trial)
            df_filtered = apply_gaussian_filter_by_block(df_interp)
            fix_df = detect_fixations_idt(
                df_filtered, DISPERSION_THRESHOLD, DURATION_THRESHOLD_MS
            )

            if fix_df.empty:
                print(f"Trial {trial_num}: No fixations detected.")
                continue

            fix_df["trial"] = trial_num
            fix_df["x_px"], fix_df["y_px"] = deg_to_px(
                fix_df["x_mean_deg"], fix_df["y_mean_deg"]
            )
            all_fixations.append(fix_df)

            # print("**************************************")
            # print(fix_df)
            # print(len(fix_df))
            
            
            """
            # 背景画像を読み込み
            img = mpimg.imread(
                f"output_aoi/{experiment_id}-{int(trial_num)+1}.jpg"
            )  # 例: "background.png"

            # 図の作成
            fig, ax = plt.subplots()

            # 背景画像の表示（軸にフィットさせて）
            ax.imshow(img, extent=[0, 1920, 1080, 0])  # 上下反転（y軸を上→下に）

            # 散布図の描画（fix_dfは事前に用意）
            ax.scatter(fix_df["x_px"], fix_df["y_px"], alpha=0.6, c="blue", s=10)

            # 軸設定（アスペクト比保持）
            ax.set_xlim(0, 1920)
            ax.set_ylim(1080, 0)  # y軸を反転
            ax.set_box_aspect(1080 / 1920)  # 縦横比を固定

            fig.text(
                0.1,
                0.05,
                f"Fixations: {len(fix_df)}",
                color="black",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="black"),
            )

            # ラベルや装飾
            ax.set_title(f"IVT Fixations in Trial {int(trial_num + 1)}")
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            ax.grid(True)

            print(
                f"ID{subject_id:03}-{experiment_id:03}の画像{experiment_id}-{int(trial_num + 1)}: {len(fix_df)} fixations detected."
            )

            # レイアウト調整＆表示
            # plt.tight_layout()
            # plt.show()

            fig.savefig(
                f"plotscatter_fixation_IdtFiltered/fixation_id{subject_id:03}-{experiment_id:03}_trial{int(trial_num + 1)}.png",
                dpi=300,
                bbox_inches="tight",
            )

            # plt.close(fig)  
            
            """

# new1_df=interpolate_missing(eye_df)
# # print(new1_df.head())
# new2_df=apply_gaussian_filter_by_block(new1_df)
# new3_df=detect_fixations_idt(new2_df)
# deg_to_px(new3_df["x_mean_deg"], new3_df["y_mean_deg"])
