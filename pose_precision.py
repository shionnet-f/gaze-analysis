import pandas as pd
import numpy as np
import math
from scipy.ndimage import gaussian_filter1d

monitor_width_cm = 47.6
monitor_height_cm = 26.8
resolution = (1920, 1080)
viewer_distance_cm = 70.0
cm_per_pixel_x = monitor_width_cm / resolution[0]
cm_per_pixel_y = monitor_height_cm / resolution[1]
viewing_distance_cm = 70
diag_length_cm = math.sqrt(monitor_width_cm**2 + monitor_height_cm**2)

# 対象の被験者19名分
for subject_id in range(1,20):
    # 各被験者の実験課題3回分
    for experiment_id in range(1, 4):
        
        pose1_df = pd.read_csv(f"./exported_csv/pose_df/pose1_df_id{subject_id:03}-{experiment_id:03}.csv")        
        
        def prepare_pose_df(df):
            df = df.copy()
            df["gx_centered"] = df["mean_x"] - 0.5
            df["gy_centered"] = df["mean_y"] - 0.5
            df["x_cm"] = df["gx_centered"] * resolution[0] * cm_per_pixel_x
            df["y_cm"] = df["gy_centered"] * resolution[1] * cm_per_pixel_y
            df["x_deg"] = np.degrees(np.arctan2(df["x_cm"], viewer_distance_cm))
            df["y_deg"] = np.degrees(np.arctan2(df["y_cm"], viewer_distance_cm))
            df["epoch_sec"] = df["epoch_time"]
            df["is_valid"] = df["validity_sum"] == 2
            return df
        
        def interpolate_missing(df, time_col="epoch_sec", max_gap_ms=100):
            df = df.copy()
            df["valid"] = df["is_valid"]
            df["interp_x"] = np.nan
            df["interp_y"] = np.nan
            df.loc[df["valid"], "interp_x"] = df.loc[df["valid"], "x_deg"]
            df.loc[df["valid"], "interp_y"] = df.loc[df["valid"], "y_deg"]
            df["interp_x"] = df["interp_x"].interpolate(limit_area="inside")
            df["interp_y"] = df["interp_y"].interpolate(limit_area="inside")

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
                    df.loc[block.index, ["interp_x", "interp_y"]] = np.nan
            return df
        
        def apply_gaussian_filter_by_block(df, col_x="interp_x", col_y="interp_y", sigma=1.0):
            df = df.copy()
            df["filtered_x"] = np.nan
            df["filtered_y"] = np.nan
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

        def detect_fixations_ivt(df, velocity_threshold=100, duration_threshold_ms=100):
            fixations = []

            timestamps = df["epoch_sec"].to_numpy()
            xs_deg = df["filtered_x"].to_numpy()
            ys_deg = df["filtered_y"].to_numpy()
            xs_norm = df["mean_x"].to_numpy()
            ys_norm = df["mean_y"].to_numpy()

            # ピクセル換算用
            resolution = (1920, 1080)

            delta_t = np.diff(timestamps)
            delta_x = np.diff(xs_deg)
            delta_y = np.diff(ys_deg)
            safe_delta_t = np.where(delta_t == 0, np.nan, delta_t)
            velocities = np.sqrt(delta_x**2 + delta_y**2) / safe_delta_t
            velocities = np.insert(velocities, 0, 0)
            velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)

            in_fixation = False
            start_idx = 0

            for i in range(len(df)):
                if np.isnan(xs_deg[i]) or np.isnan(ys_deg[i]):
                    if in_fixation:
                        in_fixation = False
                        t_start = timestamps[start_idx]
                        t_end = timestamps[i - 1]
                        duration = (t_end - t_start) * 1000
                        if duration >= duration_threshold_ms:
                            fixations.append({
                                "start_time": t_start,
                                "end_time": t_end,
                                "duration_ms": duration,
                                "x_mean_deg": np.mean(xs_deg[start_idx:i]),
                                "y_mean_deg": np.mean(ys_deg[start_idx:i]),
                                "x_mean_norm": np.mean(xs_norm[start_idx:i]),
                                "y_mean_norm": np.mean(ys_norm[start_idx:i]),
                                "x_mean_px": np.mean(xs_norm[start_idx:i]) * resolution[0],
                                "y_mean_px": np.mean(ys_norm[start_idx:i]) * resolution[1],
                            })
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
                            fixations.append({
                                "start_time": t_start,
                                "end_time": t_end,
                                "duration_ms": duration,
                                "x_mean_deg": np.mean(xs_deg[start_idx:i]),
                                "y_mean_deg": np.mean(ys_deg[start_idx:i]),
                                "x_mean_norm": np.mean(xs_norm[start_idx:i]),
                                "y_mean_norm": np.mean(ys_norm[start_idx:i]),
                                "x_mean_px": np.mean(xs_norm[start_idx:i]) * resolution[0],
                                "y_mean_px": np.mean(ys_norm[start_idx:i]) * resolution[1],
                            })

            if in_fixation:
                t_start = timestamps[start_idx]
                t_end = timestamps[-1]
                duration = (t_end - t_start) * 1000
                if duration >= duration_threshold_ms:
                    fixations.append({
                        "start_time": t_start,
                        "end_time": t_end,
                        "duration_ms": duration,
                        "x_mean_deg": np.mean(xs_deg[start_idx:]),
                        "y_mean_deg": np.mean(ys_deg[start_idx:]),
                        "x_mean_norm": np.mean(xs_norm[start_idx:]),
                        "y_mean_norm": np.mean(ys_norm[start_idx:]),
                        "x_mean_px": np.mean(xs_norm[start_idx:]) * resolution[0],
                        "y_mean_px": np.mean(ys_norm[start_idx:]) * resolution[1],
                    })

            return pd.DataFrame(fixations)
        
        
        def process_pose_df_ivt(df):
            df_prepared = prepare_pose_df(df)
            df_interp = interpolate_missing(df_prepared, time_col="epoch_sec", max_gap_ms=100)
            df_filtered = apply_gaussian_filter_by_block(df_interp, col_x="interp_x", col_y="interp_y", sigma=1.0)
            fix_df = detect_fixations_ivt(df_filtered, velocity_threshold=100, duration_threshold_ms=100)
            return fix_df
        
        fix_df = process_pose_df_ivt(pose1_df)
        fix_df.to_csv(f"./exported_csv/pose_df/pose_fix_df/pose_fix_df_id{subject_id:03}-{experiment_id:03}.csv", index=False)
        
        def evaluate_precision(df):
            total_len = len(df)

            # validity_sum == 2 のデータのみを使う
            valid_df = df.copy()

            valid_df = valid_df.dropna(subset=["x_mean_norm","y_mean_norm"])
            # データ行列
            data = valid_df[["x_mean_norm","y_mean_norm"]].values

            # 重心
            center = np.mean(data, axis=0)

            # 共分散行列と逆行列
            cov = np.cov(data, rowvar=False)
            inv_cov = np.linalg.inv(cov)

            # 差分
            diff = data - center
            left = np.dot(diff, inv_cov)
            mahal_sq = np.sum(left * diff, axis=1)
            mahal_dist = np.sqrt(mahal_sq)

            # σ=1以内
            within_sigma1 = np.sum(mahal_dist <= 1)
            total_valid = len(mahal_dist)
            percent_sigma1 = 100 * within_sigma1 / total_valid

            # 平均マハラノビス距離
            mahal_mean = np.mean(mahal_dist)

            # ユークリッド距離も参考値で計算（正確度）
            euclid_dist = np.linalg.norm(diff, axis=1)
            euclid_mean = np.mean(euclid_dist)

            # ユークリッド距離→cm→視野角
            euclid_mean_cm = euclid_mean * diag_length_cm
            euclid_mean_deg = np.degrees(np.arctan(euclid_mean_cm / viewing_distance_cm))

            # 結果
            return {
                "subject_id": subject_id,
                "task_id": experiment_id,
                "pose_length": total_len,
                "center_x": center[0],
                "center_y": center[1],
                "mahalanobis_mean": mahal_mean,
                "percent_within_sigma1": percent_sigma1,
                "euclid_mean": euclid_mean,
                "euclid_mean_deg": euclid_mean_deg,
            }
            
        summary_list = []
        summary_list.append(evaluate_precision(fix_df))
        summary_df = pd.DataFrame(summary_list)
        
        summary_df.to_csv(f"./exported_csv/pose_df/precision_summary_all.csv", index=False)
        