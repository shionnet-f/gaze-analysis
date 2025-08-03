import os
import pandas as pd

# === パス設定 ===
base_input_filtered = "./exported_csv/filtered_IVT"
base_input_fixation = "./exported_csv/fixation_IVT"
sampling_dir = "./exported_csv/sampling_df_plus03_08"
flags_path = "./exported_csv/condition_flags.csv"
precision_path = "./exported_csv/pose_df/pose_fix_df/precision_summary.csv"

output_dir_filtered = "./exported_csv/filtered_IVT_split"
output_dir_fixation = "./exported_csv/fixation_IVT_split"
os.makedirs(output_dir_filtered, exist_ok=True)
os.makedirs(output_dir_fixation, exist_ok=True)

# === 区間定義 ===
intervals = {
    "0.0-2.0": ("start_sec", "start_plus_2.0s"),
    "0.0-0.3": ("start_sec", "start_sec_plus_0.3"),
    "0.3-0.8": ("start_sec_plus_0.3", "start_sec_plus_0.8"),
    "0.8-2.0": ("start_sec_plus_0.8", "start_plus_2.0s"),
}

# === 使用可能な組み合わせを抽出 ===
flags_df = pd.read_csv(flags_path)
precision_df = pd.read_csv(precision_path)

valid_pairs = flags_df[flags_df["all_ok"] == True].merge(
    precision_df[precision_df["precision_bool"] == 1],
    on=["subject_id", "experiment_id"]
)[["subject_id", "experiment_id"]].drop_duplicates()

# === 有効ペアに対して処理を実行 ===
for _, pair in valid_pairs.iterrows():
    subject_id = pair["subject_id"]
    experiment_id = pair["experiment_id"]

    # samplingファイルの読み込み
    sampling_path = os.path.join(
        sampling_dir,
        f"with_timepoints_sampling_df_id{subject_id:03}-{experiment_id:03}_plus03_08.csv"
    )
    if not os.path.exists(sampling_path):
        print(f"⚠️ samplingファイルが見つかりません: {sampling_path}")
        continue

    sampling_df = pd.read_csv(sampling_path)

    for _, row in sampling_df.iterrows():
        trial = int(row["trial"])

        for label, (start_key, end_key) in intervals.items():
            t_start = row[start_key]
            t_end = row[end_key]

            # === filtered_IVT ===
            filtered_file = f"{base_input_filtered}/filtered_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"
            if os.path.exists(filtered_file):
                df_filtered = pd.read_csv(filtered_file)
                df_cut = df_filtered[
                    (df_filtered["epoch_sec"] >= t_start) & (df_filtered["epoch_sec"] <= t_end)
                ]
                out_filtered = f"{output_dir_filtered}/filtered_df_{subject_id:03}-{experiment_id:03}-{trial}_{label}.csv"
                df_cut.to_csv(out_filtered, index=False, encoding="utf-8-sig")

            # === fixation_IVT ===
            fixation_file = f"{base_input_fixation}/fix_df_{subject_id:03}-{experiment_id:03}-{trial}.csv"
            if os.path.exists(fixation_file):
                df_fix = pd.read_csv(fixation_file)
                df_fix_cut = df_fix[
                    (df_fix["start_time"] >= t_start) & (df_fix["end_time"] <= t_end)
                ]
                out_fix = f"{output_dir_fixation}/fix_df_{subject_id:03}-{experiment_id:03}-{trial}_{label}.csv"
                df_fix_cut.to_csv(out_fix, index=False, encoding="utf-8-sig")

    print(f"✅ {subject_id:03}-{experiment_id:03} の処理完了")

print("🎉 条件に合致した全データの切り出しと保存が完了しました。")
