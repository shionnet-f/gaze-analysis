import pandas as pd
import matplotlib.pyplot as plt

# 対象の被験者19名分
for subject_id in range(1,20):
    # 各被験者の実験課題3回分
    for experiment_id in range(1, 4):

        eye_df_interp = pd.read_csv(f"exported_csv/eye_df_interp/eye_df_interp{subject_id:03}-{experiment_id:03}.csv")

        # 0〜7の試行のみを対象にフィルタ
        eye_df_interp = eye_df_interp[eye_df_interp['trial'].between(0, 7)].copy()

        # 各試行で時間を正規化（各試行の開始時刻を0に）
        eye_df_interp['time_norm'] = eye_df_interp.groupby('trial')['epoch_sec'].transform(lambda x: x - x.min())

        # 試行番号を取得
        unique_trials = sorted(eye_df_interp['trial'].unique())

        # 各試行ごとに1枚ずつ表示
        for trial_num in unique_trials:
            trial_data = eye_df_interp[eye_df_interp['trial'] == trial_num].copy()

            # バイナリ化：0 → 0 (Invalid), 1 or 2 → 1 (Valid)
            trial_data['valid_binary'] = trial_data['validity_interp'].apply(lambda x: 1 if x in [1, 2] else 0)

            fig, ax = plt.subplots(figsize=(10, 2.5))

            # ステップラインプロット（矩形波）
            ax.step(
                trial_data['time_norm'],
                trial_data['valid_binary'],
                where='post',
                color='blue',
                linewidth=1.5
            )

            # オレンジのハイライト：補完された部分 (validity_interp == 1)
            interp_data = trial_data[trial_data['validity_interp'] == 1]
            for x in interp_data['time_norm']:
                ax.axvspan(x, x + 0.05, color='orange', alpha=0.4)

            # 軸・ラベルの設定
            ax.set_title(f"ID:{subject_id:03}-{experiment_id:03}_Trial{int(trial_num + 1)}")
            ax.set_xlim(0, 40)
            ax.set_ylim(-0.2, 1.2)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Invalid', 'Valid'])
            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Validity")
            ax.grid(True)

            # 保存
            filename = f"plotstep_validity/validity_id{subject_id:03}-{experiment_id:03}_trial{int(trial_num + 1)}.png"
            plt.tight_layout()
            # plt.show()
            plt.savefig(filename)
            plt.close()
