import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
from collections import defaultdict
import japanize_matplotlib

# === パラメータ ===
background_dir = "../face_aoi_project/output_aoi_triangle"

input_dir = "./exported_csv/fixation_IVT_split"  # ← filtered_IVT_split にすれば視線データに対応
output_dir = "./exported_csv/heatmap_fixation"
x_col = "x_mean_norm"  # ← "filtered_x" にすれば視線データ用
y_col = "y_mean_norm"  # ← "filtered_y"

# input_dir = "./exported_csv/filtered_IVT_split"
# output_dir = "./exported_csv/heatmap_filtered"
# x_col = "filtered_x"
# y_col = "filtered_y"

screen_width, screen_height = 1920, 1080
bin_size = 30
interval = "0.0-2.0"

os.makedirs(output_dir, exist_ok=True)

# === データ集約: {(exp_id, trial): [(subj_id, x_px, y_px), ...]} ===
heatmap_data = defaultdict(list)

for fname in os.listdir(input_dir):
    if not fname.endswith(f"{interval}.csv"):
        continue
    match = re.match(r".*_(\d+)-(\d+)-(\d+)_0\.0-2\.0\.csv", fname)
    if not match:
        continue

    subj_id = int(match.group(1))
    exp_id = int(match.group(2))
    trial = int(match.group(3))
    path = os.path.join(input_dir, fname)

    df = pd.read_csv(path)
    if x_col not in df.columns or y_col not in df.columns:
        continue

    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        continue

    x_px = (df[x_col] * screen_width).clip(0, screen_width - 1)
    y_px = (df[y_col] * screen_height).clip(0, screen_height - 1)

    heatmap_data[(exp_id, trial)].extend([(subj_id, x, y) for x, y in zip(x_px, y_px)])

# === ヒートマップ描画 + 件数記録 ===
summary_list = []

for (exp_id, trial), data in heatmap_data.items():
    if not data:
        continue

    # データ展開
    subj_ids = [d[0] for d in data]
    x = [d[1] for d in data]
    y = [d[2] for d in data]

    unique_subjects = len(set(subj_ids))
    summary_list.append({
        "experiment_id": exp_id,
        "trial": trial + 1,  # 表示上は1スタート
        "n_subjects": unique_subjects
    })

    # 2Dヒストグラム
    counts, _, _ = np.histogram2d(
        x, y,
        bins=bin_size,
        range=[[0, screen_width], [0, screen_height]]
    )
    counts = counts / counts.sum()

    # === 背景画像の読み込み ===
    bg_path = os.path.join(background_dir, f"{exp_id}-{trial+1}.jpg")
    if not os.path.exists(bg_path):
        print(f"⚠ 背景画像なし: {bg_path}")
        continue

    img = mpimg.imread(bg_path)
    fig, ax = plt.subplots(figsize=(7, 4))

    # 背景
    ax.imshow(img, extent=[0, screen_width, screen_height, 0], aspect='auto')

    # ヒートマップ
    cax = ax.imshow(
        counts.T,
        extent=[0, screen_width, 0, screen_height],
        origin='lower',
        cmap="jet",
        alpha=0.5,
        vmin=0,
        vmax=0.12
    )

    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label("正規化した注視密度", fontsize=12)

    ax.set_xlim(0, screen_width)
    ax.set_ylim(0, screen_height)
    ax.invert_yaxis()  # 上下をモニタ座標系に合わせる
    ax.set_xlabel("X座標", fontsize=12)
    ax.set_ylabel("Y座標", fontsize=12)

    plt.tight_layout()
    outpath = os.path.join(output_dir, f"heatmap_{exp_id:03}-{trial+1}_normalized.png")
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✅ 保存: {outpath}")

# === 件数のCSV出力 ===
summary_df = pd.DataFrame(summary_list)
summary_df = summary_df.sort_values(by=["experiment_id", "trial"])
summary_df.to_csv(os.path.join(output_dir, "heatmap_subject_counts.csv"), index=False, encoding="utf-8-sig")
print("📄 ヒートマップごとの件数を出力しました。")
