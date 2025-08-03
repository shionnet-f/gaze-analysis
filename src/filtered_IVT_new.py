import os
import pandas as pd
import numpy as np
import math
from scipy.ndimage import gaussian_filter1d

# === パラメータ設定 ===
monitor_width_cm = 47.6
monitor_height_cm = 26.8
resolution = (1920, 1080)
viewer_distance_cm = 60.0
cm_per_pixel_x = monitor_width_cm / resolution[0]
cm_per_pixel_y = monitor_height_cm / resolution[1]

# === 出力フォルダ作成 ===
os.makedirs("exported_csv/filtered_IVT", exist_ok=True)
os.makedirs("exported_csv/fixation_IVT", exist_ok=True)

# === ガウスフィルタ＋単位変換処理 ===
def filter_norm_from_gxgy(df, sigma=1.0):
    df = df.copy()
    df["valid"] = df["validity_sum"] > 1
    df["interp_x"] = np.nan
    df["interp_y"] = np.nan
    df.loc[df["valid"], "interp_x"] = df["gx"]
    df.loc[df["valid"], "interp_y"] = df["gy"]
    df["interp_x"] = df["interp_x"].interpolate(limit_area="inside")
    df["interp_y"] = df["interp_y"].interpolate(limit_area="inside")

    time_col = "epoch_sec"
    invalid_mask = ~df["valid"]
    group_id = (invalid_mask != invalid_mask.shift()).cumsum()
    for _, block in df[invalid_mask].groupby(group_id):
        if len(block) == 0:
            continue
        t_start = block[time_col].iloc[0]
        t_end = block[time_col].iloc[-1]
        duration_ms = (t_end - t_start) * 1000
        if duration_ms > 100:
            df.loc[block.index, ["interp_x", "interp_y"]] = np.nan

    df["filtered_x"] = np.nan
    df["filtered_y"] = np.nan
    valid_mask = df["interp_x"].notna() & df["interp_y"].notna()
    block_id = (valid_mask != valid_mask.shift()).cumsum()
    for _, block in df[valid_mask].groupby(block_id):
        idx = block.index
        df.loc[idx, "filtered_x"] = gaussian_filter1d(block["interp_x"], sigma=sigma)
        df.loc[idx, "filtered_y"] = gaussian_filter1d(block["interp_y"], sigma=sigma)

    # 単位変換（deg, px）
    df["x_deg"] = np.degrees(np.arctan2((df["filtered_x"] - 0.5) * resolution[0] * cm_per_pixel_x, viewer_distance_cm))
    df["y_deg"] = np.degrees(np.arctan2((df["filtered_y"] - 0.5) * resolution[1] * cm_per_pixel_y, viewer_distance_cm))
    df["x_px"] = df["filtered_x"] * resolution[0]
    df["y_px"] = df["filtered_y"] * resolution[1]
    return df

# === 注視検出（IVT法）===
def detect_fixations_ivt(df, velocity_threshold=20, min_duration_ms=100):
    fixations = []
    timestamps = df["epoch_sec"].to_numpy()
    xs = df["x_deg"].to_numpy()
    ys = df["y_deg"].to_numpy()

    delta_t = np.diff(timestamps)
    delta_x = np.diff(xs)
    delta_y = np.diff(ys)
    safe_delta_t = np.where(delta_t == 0, np.nan, delta_t)
    velocities = np.sqrt(delta_x**2 + delta_y**2) / safe_delta_t
    velocities = np.insert(velocities, 0, 0)
    velocities = np.nan_to_num(velocities)

    in_fixation = False
    start_idx = 0

    for i in range(len(df)):
        if np.isnan(xs[i]) or np.isnan(ys[i]):
            if in_fixation:
                in_fixation = False
                fix = make_fixation(df, xs, ys, timestamps, start_idx, i - 1)
                if fix is not None:
                    fixations.append(fix)
            continue

        if velocities[i] < velocity_threshold:
            if not in_fixation:
                in_fixation = True
                start_idx = i
        else:
            if in_fixation:
                in_fixation = False
                fix = make_fixation(df, xs, ys, timestamps, start_idx, i - 1)
                if fix is not None:
                    fixations.append(fix)

    if in_fixation:
        fix = make_fixation(df, xs, ys, timestamps, start_idx, len(df) - 1)
        if fix is not None:
            fixations.append(fix)

    return pd.DataFrame(fixations)

# === 注視1件の情報を返す ===
def make_fixation(df, xs, ys, timestamps, start_idx, end_idx):
    if end_idx <= start_idx:
        return None  # 空fixationはスキップ

    x_mean_deg = np.mean(xs[start_idx:end_idx + 1])
    y_mean_deg = np.mean(ys[start_idx:end_idx + 1])
    t_start = timestamps[start_idx]
    t_end = timestamps[end_idx]
    duration = (t_end - t_start) * 1000

    x_mean_norm = np.mean(df["filtered_x"].iloc[start_idx:end_idx + 1])
    y_mean_norm = np.mean(df["filtered_y"].iloc[start_idx:end_idx + 1])
    
    x_mean_px = x_mean_norm * resolution[0]
    y_mean_px = y_mean_norm * resolution[1]
    

    return {
        "start_time": t_start,
        "end_time": t_end,
        "duration_ms": duration,
        "x_mean_norm": x_mean_norm,
        "y_mean_norm": y_mean_norm,
        "x_mean_deg": x_mean_deg,
        "y_mean_deg": y_mean_deg,
        "x_mean_px": x_mean_px,
        "y_mean_px": y_mean_px
    }

# === 全被験者・全課題・全試行処理 ===
for subject_id in range(1, 20):
    for experiment_id in range(1, 4):
        eye_path = f"./exported_csv/eye_df_id{subject_id:03}-{experiment_id:03}.csv"
        sampling_path = f"./exported_csv/sampling_df_id{subject_id:03}-{experiment_id:03}.csv"

        if not os.path.exists(eye_path) or not os.path.exists(sampling_path):
            continue

        eye_df = pd.read_csv(eye_path)
        sampling_df = pd.read_csv(sampling_path)

        for _, row in sampling_df.iterrows():
            trial_num = int(row["trial"])
            start_sec = row["start_sec"]
            end_sec = row["end_sec"]

            trial_df = eye_df[(eye_df["epoch_sec"] >= start_sec) & (eye_df["epoch_sec"] <= end_sec)].copy()
            if trial_df.empty:
                continue

            trial_df = filter_norm_from_gxgy(trial_df)

            # filtered_path = f"./exported_csv/filtered_IVT/filtered_df_{subject_id:03}-{experiment_id:03}-{trial_num}.csv"
            # trial_df.to_csv(filtered_path, index=False, float_format="%.6f", encoding="utf-8-sig")

            fix_df = detect_fixations_ivt(trial_df)
            fix_path = f"./exported_csv/fixation_IVT/fix_df_{subject_id:03}-{experiment_id:03}-{trial_num}.csv"
            fix_df.to_csv(fix_path, index=False, float_format="%.3f", encoding="utf-8-sig")

            print(f"✅ {subject_id:03}-{experiment_id:03}-{trial_num} 保存完了")
