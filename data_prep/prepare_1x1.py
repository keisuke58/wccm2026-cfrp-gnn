#!/usr/bin/env python3
"""1x1 微小欠陥（OOD汎化テスト用）の DSPSS を学習データと同じ形式に前処理する。

学習データ(all_sub_hole_defect_zscore)と同一パイプライン:
  1. 差分:   subtracted = raw_DSPSS - hole_no_defect_original
  2. 外れ値: percentile(1,99) でクリップ（global統計, サンプルから推定）
  3. 正規化: サンプルごとに自前の mean/std で z-score（per-sample）
     ↑ normalize_with_outlier_removal.py の zscore と同一仕様（各サイズ独立処理に整合）

ラベル(19class)は別途 rpt_19class_1x1.py で RPT から生成（class = 挿入層 L-1, 非欠陥=0）。

使い方:
  python prepare_1x1.py \
    --raw_dir   /home/nishioka/CFRP/CFRP_hole/hole_data_inp/Defect_hole_1x1_Random_npy \
    --reference /home/nishioka/CFRP/CFRP_hole/hole_data_inp/basicdata_for_holegnn/hole_no_defect_original.npy \
    --out_dir   /home/nishioka/CFRP/CFRP_hole/hole_data_inp/Defect_hole_1x1_subtracted_zscore \
    --sample_size 1000
"""
import argparse, os, glob
import numpy as np


def global_percentiles(files, reference, sample_size, total_nodes):
    """サンプルした subtracted 値から percentile_1 / percentile_99 を推定。"""
    rng = list(range(len(files)))
    if sample_size and sample_size < len(files):
        step = max(1, len(files) // sample_size)
        rng = rng[::step][:sample_size]
    vals = []
    for i in rng:
        a = np.load(files[i])[:total_nodes].astype(np.float64)
        vals.append(a - reference)
    vals = np.concatenate(vals)
    p1, p99 = np.percentile(vals, 1), np.percentile(vals, 99)
    print(f"[stats] sampled {len(rng)} files, subtracted p1={p1:.4f} p99={p99:.4f} "
          f"(mean={vals.mean():.4f} std={vals.std():.4f})")
    return p1, p99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--sample_size', type=int, default=1000)
    ap.add_argument('--total_nodes', type=int, default=13942)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ref = np.load(args.reference)[:args.total_nodes].astype(np.float64)
    files = sorted(glob.glob(os.path.join(args.raw_dir, '*.npy')))
    print(f"[1x1] {len(files)} raw files | reference shape {ref.shape}")
    assert ref.shape[0] == args.total_nodes, f"reference {ref.shape} != {args.total_nodes}"

    p1, p99 = global_percentiles(files, ref, args.sample_size, args.total_nodes)

    n_ok = 0
    for f in files:
        a = np.load(f)[:args.total_nodes].astype(np.float64)
        if a.shape[0] != args.total_nodes:
            print(f"  skip shape {a.shape}: {os.path.basename(f)}"); continue
        sub = a - ref                       # 1) 差分
        sub = np.clip(sub, p1, p99)          # 2) percentile クリップ
        s = sub.std()                        # 3) per-sample z-score
        out = (sub - sub.mean()) / s if s > 0 else sub * 0.0
        np.save(os.path.join(args.out_dir, os.path.basename(f)), out.astype(np.float32))
        n_ok += 1
    print(f"[1x1] wrote {n_ok}/{len(files)} -> {args.out_dir}")


if __name__ == '__main__':
    main()
