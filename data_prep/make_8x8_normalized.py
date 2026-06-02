#!/usr/bin/env python3
"""8x8 plain（非差分）正規化データを 2x2/4x4 と同じレシピで生成する。

2x2/4x4 の `Defect_hole_*_Region1_21_npy_normalized` は
**グローバル min-max → [0,1]**（normalization_statistics.csv で確認: normalized∈[~1e-5, ~0.99998]）。
各サイズ独立に global min/max を計算して適用する方式。本スクリプトは raw 8x8 にこれを適用し
`Defect_hole_8x8_Region1_21_npy_normalized` を作る（差分なし＝plain）。

入力: DSPSS_8x8/Defect_hole_8x8_original/*.npy（生DSPSS, 13942）
出力: Defect_hole_8x8_Region1_21_npy_normalized/*.npy + normalization_statistics.csv
"""
import argparse, os, glob
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--total_nodes', type=int, default=13942)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.raw_dir, '*.npy')))
    print(f"[8x8norm] {len(files)} raw files")

    # 1) global min/max（全ファイル走査）
    gmin, gmax = np.inf, -np.inf
    for f in files:
        a = np.load(f)[:args.total_nodes]
        gmin = min(gmin, float(a.min())); gmax = max(gmax, float(a.max()))
    rng = gmax - gmin
    print(f"[8x8norm] global min={gmin:.4f} max={gmax:.4f} range={rng:.4f}")
    assert rng > 0

    # 2) min-max 正規化 + 統計csv
    csv = os.path.join(args.out_dir, 'normalization_statistics.csv')
    with open(csv, 'w') as cf:
        cf.write('filename,original_min,original_max,normalized_min,normalized_max,normalized_mean,normalized_std\n')
        n_ok = 0
        for f in files:
            a = np.load(f)[:args.total_nodes].astype(np.float64)
            out = (a - gmin) / rng
            np.save(os.path.join(args.out_dir, os.path.basename(f)), out.astype(np.float32))
            cf.write(f"{os.path.basename(f)},{a.min():.6g},{a.max():.6g},"
                     f"{out.min():.6g},{out.max():.6g},{out.mean():.6g},{out.std():.6g}\n")
            n_ok += 1
    print(f"[8x8norm] wrote {n_ok}/{len(files)} -> {args.out_dir}")


if __name__ == '__main__':
    main()
