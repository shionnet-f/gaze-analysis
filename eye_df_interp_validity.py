import pandas as pd

# 対象の被験者19名分
for subject_id in range(1,20):
    # 各被験者の実験課題3回分
    for experiment_id in range(1, 4):
        
        eye_df_interp = pd.read_csv(f"exported_csv/eye_df_interp/eye_df_interp{subject_id:03}-{experiment_id:03}.csv")

        # trialが0〜7のデータのみ抽出（-1は除外）
        filtered_df = eye_df_interp[eye_df_interp['trial'].between(0, 7)]

        # 各試行ごとの集計
        summary_df = filtered_df.groupby('trial').agg(
            total_samples=('trial', 'count'),
            valid_sum_2=('validity_sum', lambda x: (x == 2).sum()),
            valid_interp_1_or_2=('validity_interp', lambda x: ((x == 1) | (x == 2)).sum())
        ).reset_index()

        summary_df['rate_valid_sum_2'] = (summary_df['valid_sum_2'] * 100 / summary_df['total_samples']).round(2)
        summary_df['rate_valid_interp_1_or_2'] = (summary_df['valid_interp_1_or_2'] * 100 / summary_df['total_samples']).round(2)
        
        # ファイル名（例：補完後のデータを保存）
        output_path = f"exported_csv/eye_df_interp_validity/eye_df_interp_validity{subject_id:03}-{experiment_id:03}.csv"
        
        # 保存（index=Falseでインデックス列を除く）
        summary_df.to_csv(output_path, index=False, float_format="%.6f", encoding="utf-8-sig")

