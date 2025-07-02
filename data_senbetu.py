import pandas as pd

# 対象の被験者19名分
for subject_id in range(1,20):
    # 各被験者の実験課題3回分
    for experiment_id in range(1, 4):
        eye_df  = pd.read_csv(f"./exported_csv/eye_df_interp_validity/eye_df_interp_validity{subject_id:03}-{experiment_id:03}.csv")
        sampling_df = pd.read_csv(f"./exported_csv/sampling_df_id{subject_id:03}-{experiment_id:03}.csv")

        # 必須列があるか確認
        required_sampling = ["trial", "samples", "start_sec","end_sec",	"duration_sec","sampling_rate_Hz"]
        required_eye = ['trial', 'total_samples', 'valid_sum_2', 'valid_interp_1_or_2', 'rate_valid_sum_2', 'rate_valid_interp_1_or_2']

        if not all(col in sampling_df.columns for col in required_sampling):
            raise ValueError("sampling_dfに必要な列がありません")

        if not all(col in eye_df.columns for col in required_eye):
            raise ValueError("eye_dfに必要な列がありません")

        # duration判定
        sampling_df["duration_ok"] = sampling_df["duration_sec"] >= 39.9

        # eye_df 有効率条件
        eye_df["validity_ok"] = eye_df["rate_valid_interp_1_or_2"] >= 60

        # 補完率計算
        eye_df["interp_rate(%)"] = (
            (eye_df["valid_interp_1_or_2"] - eye_df["valid_sum_2"])
            / eye_df["valid_interp_1_or_2"]
        ) * 100
        eye_df["interp_ok"] = eye_df["interp_rate(%)"] < 10

        # durationとvalidity/interpをマージ
        merged = pd.merge(
            sampling_df[[ "trial","duration_sec", "duration_ok"]],
            eye_df[[ "trial", "validity_ok", "interp_ok"]],
            on=["trial"],
            how="outer"
        )

        # すべての条件
        merged["all_ok"] = (
            merged["duration_ok"].fillna(False) &
            merged["validity_ok"].fillna(False) &
            merged["interp_ok"].fillna(False)
        )

        # ID列を追加
        merged["subject_id"] = subject_id
        merged["experiment_id"] = experiment_id

        merged = merged[[
            "subject_id",
            "experiment_id",
            "trial",
            "duration_sec",
            "duration_ok",
            "validity_ok",
            "interp_ok",
            "all_ok"
        ]]

        output_file = "./exported_csv/condition_flags.csv"

        # 1ファイルずつ追記
        if subject_id == 1 and experiment_id == 1:
            # 1回目はヘッダーあり
            merged.to_csv(output_file, mode="w", index=False, header=True)
        else:
            # 2回目以降はヘッダーなし
            merged.to_csv(output_file, mode="a", index=False, header=False)
        