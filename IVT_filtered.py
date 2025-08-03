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

# === 出力先フォルダ作成 ===
os.makedirs("exported_csv/filtered_IVT", exist_ok=True)
os.makedirs("exported_csv/fixation_IVT", exist_ok=True)

# === 視線データの準備処理 ===
def prepare_eye_df(df):
    df = df.copy()
    df["gx_centered"] = df["gx"] - 0.5
    df["gy_centered"] = df["gy"] - 0.5
    df["x_cm"] = df["gx_centered"] * resolution[0] * cm_per_pixel_x
    df["y_cm"] = df["gy_centered"] * resolution[1] * cm_per_pixel_y
    df["x_deg"] = np.degrees(np.arctan2(df["x_cm"], viewer_distance_cm))
    df["y_deg"] = np.degrees(np.arctan2(df["y_cm"], viewer_distance_cm))
    df["valid"] = df["validity_sum"] > 1
    return df

# === 線形補間（100ms未満のみ） ===
def interpolate_norm(df, time_col="epoch_sec", max_gap_ms=100):
    df = df.copy()
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

# === ガウスフィルタで平滑化 ===
def filter_norm(df, sigma=1.0):
    df = df.copy()
    df["filtered_x"] = np.nan
    df["filtered_y"] = np.nan
    valid_mask = df["interp_x"].notna() & df["interp_y"].notna()
    block_id = (valid_mask != valid_mask.shift()).cumsum()
    blocks = df[valid_mask].groupby(block_id)
    for _, block in blocks:
        idx = block.index
        smoothed_x = gaussian_filter1d(block["interp_x"], sigma=sigma)
        smoothed_y = gaussian_filter1d(block["interp_y"], sigma=sigma)
        df.loc[idx, "filtered_x"] = smoothed_x
        df.loc[idx, "filtered_y"] = smoothed_y
    return df

# === IVT法による注視検出 ===
def detect_fixations_ivt(df, velocity_threshold=20, min_duration_ms=100):
    fixations = []
    timestamps = df["epoch_sec"].to_numpy()
    xs = df["filtered_x"].to_numpy()
    ys = df["filtered_y"].to_numpy()

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
                t_start = timestamps[start_idx]
                t_end = timestamps[i - 1]
                duration = (t_end - t_start) * 1000
                if duration >= min_duration_ms:
                    x_mean_deg = np.mean(xs[start_idx:i])
                    y_mean_deg = np.mean(ys[start_idx:i])
                    x_mean_px = (x_mean_deg * viewer_distance_cm)
                    y_mean_px = (y_mean_deg * viewer_distance_cm)
                    x_px = (x_mean_px / cm_per_pixel_x) + (resolution[0] / 2)
                    y_px = (y_mean_px / cm_per_pixel_y) + (resolution[1] / 2)
                    fixations.append({
                        "start_time": t_start,
                        "end_time": t_end,
                        "duration_ms": duration,
                        "x_mean_deg": x_mean_deg,
                        "y_mean_deg": y_mean_deg,
                        "x_mean_px": x_px,
                        "y_mean_px": y_px
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
                if duration >= min_duration_ms:
                    x_mean_deg = np.mean(xs[start_idx:i])
                    y_mean_deg = np.mean(ys[start_idx:i])
                    x_mean_px = (x_mean_deg * viewer_distance_cm)
                    y_mean_px = (y_mean_deg * viewer_distance_cm)
                    x_px = (x_mean_px / cm_per_pixel_x) + (resolution[0] / 2)
                    y_px = (y_mean_px / cm_per_pixel_y) + (resolution[1] / 2)
                    fixations.append({
                        "start_time": t_start,
                        "end_time": t_end,
                        "duration_ms": duration,
                        "x_mean_deg": x_mean_deg,
                        "y_mean_deg": y_mean_deg,
                        "x_mean_px": x_px,
                        "y_mean_px": y_px
                    })

    # 最後のfixation
    if in_fixation:
        t_start = timestamps[start_idx]
        t_end = timestamps[-1]
        duration = (t_end - t_start) * 1000
        if duration >= min_duration_ms:
            x_mean_deg = np.mean(xs[start_idx:])
            y_mean_deg = np.mean(ys[start_idx:])
            x_mean_px = (x_mean_deg * viewer_distance_cm)
            y_mean_px = (y_mean_deg * viewer_distance_cm)
            x_px = (x_mean_px / cm_per_pixel_x) + (resolution[0] / 2)
            y_px = (y_mean_px / cm_per_pixel_y) + (resolution[1] / 2)
            fixations.append({
                "start_time": t_start,
                "end_time": t_end,
                "duration_ms": duration,
                "x_mean_deg": x_mean_deg,
                "y_mean_deg": y_mean_deg,
                "x_mean_px": x_px,
                "y_mean_px": y_px
            })

    return pd.DataFrame(fixations)

# === 各被験者・課題・試行ごとに処理 ===
for subject_id in range(1, 20):
    for experiment_id in range(1, 4):
        eye_path = f"./exported_csv/eye_df_id{subject_id:03}-{experiment_id:03}.csv"
        sampling_path = f"./exported_csv/sampling_df_id{subject_id:03}-{experiment_id:03}.csv"

        if not os.path.exists(eye_path) or not os.path.exists(sampling_path):
            print(f"⚠️ ファイルなし: id{subject_id:03}-{experiment_id:03}")
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

            trial_df = prepare_eye_df(trial_df)
            trial_df = interpolate_norm(trial_df)
            trial_df = filter_norm(trial_df)

            # 保存（フィルタ済みデータ）
            filtered_out = f"./exported_csv/filtered_IVT/filtered_df_{subject_id:03}-{experiment_id:03}-{trial_num}.csv"
            trial_df.to_csv(filtered_out, index=False, float_format="%.6f", encoding="utf-8-sig")

            # 注視検出
            # fix_df = detect_fixations_ivt(trial_df)
            # fix_out = f"./exported_csv/fixation_IVT/fix_df_{subject_id:03}-{experiment_id:03}-{trial_num}.csv"
            # fix_df.to_csv(fix_out, index=False, float_format="%.3f", encoding="utf-8-sig")

            # print(f"✅ {subject_id:03}-{experiment_id:03}-{trial_num} 保存完了")
            # print("**************************************")
            # print(fix_df)
            # print(len(fix_df))
            
            """
            # # 背景画像を読み込み
            # img = mpimg.imread(
            #     f"output_aoi/{experiment_id}-{int(trial_num)+1}.jpg"
            # )  # 例: "background.png"

            # # 図の作成
            # fig, ax = plt.subplots()

            # # 背景画像の表示（軸にフィットさせて）
            # ax.imshow(img, extent=[0, 1920, 1080, 0])  # 上下反転（y軸を上→下に）

            # # 散布図の描画（fix_dfは事前に用意）
            # ax.scatter(fix_df["x_px"], fix_df["y_px"], alpha=0.5, c='green', s=10)

            # # 軸設定（アスペクト比保持）
            # ax.set_xlim(0, 1920)
            # ax.set_ylim(1080, 0)  # y軸を反転
            # ax.set_box_aspect(1080 / 1920)  # 縦横比を固定

            # fig.text(0.1,0.05, f"Fixations: {len(fix_df)}", color="black", fontsize=14, bbox=dict(facecolor='white', edgecolor='black'))


            # # ラベルや装飾
            # ax.set_title(f"IVT Fixations in Trial {int(trial_num + 1)}")
            # ax.set_xlabel("X (px)")
            # ax.set_ylabel("Y (px)")
            # ax.grid(True)

            # print(
            #     f"ID{subject_id:03}-{experiment_id:03}の画像{experiment_id}-{int(trial_num + 1)}: {len(fix_df)} fixations detected."
            # )

            # # レイアウト調整＆表示
            # # plt.tight_layout()
            # # plt.show()
            # fig.savefig(f"plotscatter_fixation_IvtFiltered/fixation_id{subject_id:03}-{experiment_id:03}_trial{int(trial_num + 1)}.png",
            # dpi=300, bbox_inches='tight')

            # plt.close(fig)  # ← これを忘れない
            """
