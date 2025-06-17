import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter1d


# IVT法による注視検出と可視化

# 対象の被験者19名分
for subject_id in range(1,20): 
    # 各被験者の実験課題3回分
    for experiment_id in range(1, 4):

        # モニターサイズ(物理)
        monitor_width_cm = 47.6
        monitor_height_cm = 26.8
        # モニター解像度(px)
        monitor_resolution_px = (1920, 1080)
        # 視距離(cm)
        viewer_distance_cm = 60.0

        # IVT法のパラメータ
        VELOCITY_THRESHOLD = 30  # deg
        DURATION_THRESHOLD_MS = 100  # ms

        eye_df = pd.read_csv(
            f"exported_csv/eye_df_id{subject_id:03}-{experiment_id:03}.csv"
        )
        sampling_df = pd.read_csv(
            f"exported_csv/sampling_df_id{subject_id:03}-{experiment_id:03}.csv"
        )

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

        # 100ms以下の欠損を線形補完の処理
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
        def apply_gaussian_filter_by_block(
            df, col_x="interp_x", col_y="interp_y", sigma=1.0
        ):
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

        # IVT法による注視検出
        def detect_fixations_ivt(
            df,
            velocity_threshold=VELOCITY_THRESHOLD,
            duration_threshold_ms=DURATION_THRESHOLD_MS,
        ):
            fixations = []
            timestamps = df["epoch_sec"].to_numpy()
            xs = df["filtered_x"].to_numpy()
            ys = df["filtered_y"].to_numpy()

            # 速度（deg/s）を計算
            delta_t = np.diff(timestamps)
            delta_x = np.diff(xs)
            delta_y = np.diff(ys)

            safe_delta_t = np.where(delta_t == 0, np.nan, delta_t)

            velocities = np.sqrt(delta_x**2 + delta_y**2) / safe_delta_t
            velocities = np.insert(velocities, 0, 0)

            velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)

            in_fixation = False
            start_idx = 0

            for i in range(len(df)):
                if np.isnan(xs[i]) or np.isnan(ys[i]):
                    if in_fixation:
                        # 注視の終了
                        in_fixation = False
                        t_start = timestamps[start_idx]
                        t_end = timestamps[i - 1]
                        duration = (t_end - t_start) * 1000
                        if duration >= duration_threshold_ms:
                            fixations.append(
                                {
                                    "start_time": t_start,
                                    "end_time": t_end,
                                    "duration_ms": duration,
                                    "x_mean_deg": np.mean(xs[start_idx:i]),
                                    "y_mean_deg": np.mean(ys[start_idx:i]),
                                }
                            )
                    continue

                if velocities[i] < velocity_threshold:
                    if not in_fixation:
                        in_fixation = True
                        start_idx = i
                else:
                    if in_fixation:
                        in_fixation = False
                        t_start = timestamps[start_idx]
                        t_end = timestamps[i - 1]
                        duration = (t_end - t_start) * 1000
                        if duration >= duration_threshold_ms:
                            fixations.append(
                                {
                                    "start_time": t_start,
                                    "end_time": t_end,
                                    "duration_ms": duration,
                                    "x_mean_deg": np.mean(xs[start_idx:i]),
                                    "y_mean_deg": np.mean(ys[start_idx:i]),
                                }
                            )

            # 最後が注視で終わっていた場合
            if in_fixation:
                t_start = timestamps[start_idx]
                t_end = timestamps[-1]
                duration = (t_end - t_start) * 1000
                if duration >= duration_threshold_ms:
                    fixations.append(
                        {
                            "start_time": t_start,
                            "end_time": t_end,
                            "duration_ms": duration,
                            "x_mean_deg": np.mean(xs[start_idx:]),
                            "y_mean_deg": np.mean(ys[start_idx:]),
                        }
                    )

            return pd.DataFrame(fixations)

        # 度数法からピクセル単位に変換
        def deg_to_px(x_mean_deg, y_mean_deg):
            x_cm = np.tan(np.radians(x_mean_deg)) * viewer_distance_cm
            y_cm = np.tan(np.radians(y_mean_deg)) * viewer_distance_cm
            x_mean_px = (x_cm / cm_per_pixel_x) + (monitor_resolution_px[0] / 2)
            y_mean_px = (y_cm / cm_per_pixel_y) + (monitor_resolution_px[1] / 2)
            return x_mean_px, y_mean_px

        all_fixations = []

        for _, row in sampling_df.iterrows():
            t_start = row["start_sec"]
            t_end = row["end_sec"]
            trial_num = row["trial"]

            trial_df = eye_df[
                (eye_df["epoch_sec"] >= t_start) & (eye_df["epoch_sec"] <= t_end)
            ]
            interp_df = interpolate_missing(trial_df)
            filtered_df = apply_gaussian_filter_by_block(interp_df)
            fix_df = detect_fixations_ivt(filtered_df)

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

            # 背景画像を読み込み
            img = mpimg.imread(
                f"output_aoi/{experiment_id}-{int(trial_num)+1}.jpg"
            )  # 例: "background.png"

            # 図の作成
            fig, ax = plt.subplots()

            # 背景画像の表示（軸にフィットさせて）
            ax.imshow(img, extent=[0, 1920, 1080, 0])  # 上下反転（y軸を上→下に）

            # 散布図の描画（fix_dfは事前に用意）
            ax.scatter(fix_df["x_px"], fix_df["y_px"], alpha=0.5, c='green', s=10)

            # 軸設定（アスペクト比保持）
            ax.set_xlim(0, 1920)
            ax.set_ylim(1080, 0)  # y軸を反転
            ax.set_box_aspect(1080 / 1920)  # 縦横比を固定

            fig.text(0.1,0.05, f"Fixations: {len(fix_df)}", color="black", fontsize=14, bbox=dict(facecolor='white', edgecolor='black'))


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
            fig.savefig(f"plotscatter_fixation_IvtFiltered/fixation_id{subject_id:03}-{experiment_id:03}_trial{int(trial_num + 1)}.png",
            dpi=300, bbox_inches='tight')

            # plt.close(fig)  # ← これを忘れない

# new1_df=interpolate_missing(eye_df)

# new2_df=apply_gaussian_filter_by_block(new1_df)

# new3_df=detect_fixations_ivt(new2_df)

# deg_to_px(new3_df["x_mean_deg"], new3_df["y_mean_deg"])
