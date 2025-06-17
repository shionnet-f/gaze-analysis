import pandas as pd
import numpy as np

# 対象の被験者19名分
for subject_id in range(1,20):
    # 各被験者の実験課題3回分
    for experiment_id in range(1, 4):
        
        eye_df = pd.read_csv(f"exported_csv/eye_df_id{subject_id:03}-{experiment_id:03}.csv")

        def interpolate_validity(eye_df, max_gap_duration=0.1):
            
            eye_df = eye_df.copy()

            # 欠損扱いの視線を NaN にする（validity_sum != 2）
            gx_interp = eye_df['gx'].copy()
            gy_interp = eye_df['gy'].copy()
            gx_interp[eye_df['validity_sum'] != 2] = np.nan
            gy_interp[eye_df['validity_sum'] != 2] = np.nan

            # 初期 validity_interp を 2（完全有効）としておく
            validity_interp = np.full(len(eye_df), 2.0)
            validity_interp[eye_df['validity_sum'] != 2] = np.nan  # 一時的に NaN にする

            isnan = gx_interp.isna()
            gap_start = None

            for i in range(len(eye_df)):
                if isnan.iloc[i] and gap_start is None:
                    gap_start = i
                elif not isnan.iloc[i] and gap_start is not None:
                    gap_end = i
                    duration = eye_df['epoch_sec'].iloc[gap_end - 1] - eye_df['epoch_sec'].iloc[gap_start]
                    if duration <= max_gap_duration:
                        # 補完する（視線座標）
                        gx_interp.iloc[gap_start:gap_end] = gx_interp.interpolate(method='linear').iloc[gap_start:gap_end]
                        gy_interp.iloc[gap_start:gap_end] = gy_interp.interpolate(method='linear').iloc[gap_start:gap_end]
                        validity_interp[gap_start:gap_end] = 1  # 補完済み
                    else:
                        validity_interp[gap_start:gap_end] = 0  # 補完せず
                    gap_start = None

            # 最後が欠損で終わる場合
            if gap_start is not None:
                validity_interp[gap_start:] = 0

            # 結果の追加
            eye_df['gx_interp'] = gx_interp
            eye_df['gy_interp'] = gy_interp
            eye_df['validity_interp'] = np.nan_to_num(validity_interp, nan=0).astype(int)

            return eye_df
        
        eye_df = interpolate_validity(eye_df)
        # ファイル名（例：補完後のデータを保存）
        output_path = f"exported_csv/eye_df_interp/eye_df_interp{subject_id:03}-{experiment_id:03}.csv"
        

        # 保存（index=Falseでインデックス列を除く）
        eye_df.to_csv(output_path, index=False, float_format="%.6f", encoding="utf-8-sig")

