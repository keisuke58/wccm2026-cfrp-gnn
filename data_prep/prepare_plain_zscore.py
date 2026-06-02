#!/usr/bin/env python3
"""plain（差分なし）zscore データを、差分版と同一の分割・同一の正規化で作る。

公平な「差分 vs plain」アブレーション用。差分版(all_sub_hole_defect_zscore)との違いは
**欠陥なし基準の引き算をしない**点のみ。正規化は同じ per-sample zscore（percentileクリップ付き）。

復元トリック: vancouver には差分前 all_sub_hole_defect(= raw − 実基準) と分割情報があるので
  raw = all_sub_hole_defect/<file> + 実基準(real_hole_no_defect_original.npy)
で生DSPSSを復元 → クロスサーバー転送不要・同一ファイル分割を保証。

出力: all_hole_defect_zscore/{train,val,test}/<同名.npy>
"""
import argparse, os, glob
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sub_dir', default='/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect',
                    help='flat subtracted npy dir (= raw - real_ref)')
    ap.add_argument('--ref', default='/home/nishioka/GNN/GNN_hole_2026/real_hole_no_defect_original.npy')
    ap.add_argument('--split_base', default='/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore',
                    help='dir with train/val/test subfolders defining the file split')
    ap.add_argument('--out_base', default='/home/nishioka/GNN/GNN_hole_2026/all_hole_defect_zscore')
    ap.add_argument('--total_nodes', type=int, default=13942)
    ap.add_argument('--sample_size', type=int, default=1000)
    args = ap.parse_args()

    ref = np.load(args.ref)[:args.total_nodes].astype(np.float64)
    print(f"[plain] ref shape {ref.shape} mean {ref.mean():.3f}")

    def raw_of(fname):
        sub = np.load(os.path.join(args.sub_dir, fname))[:args.total_nodes].astype(np.float64)
        return sub + ref  # 生DSPSS復元

    # global percentile(1,99) を train の raw サンプルから推定
    train_files = sorted(os.listdir(os.path.join(args.split_base, 'train')))
    step = max(1, len(train_files) // args.sample_size)
    sample = train_files[::step][:args.sample_size]
    vals = np.concatenate([raw_of(f) for f in sample])
    p1, p99 = np.percentile(vals, 1), np.percentile(vals, 99)
    print(f"[plain] raw sampled {len(sample)} files: p1={p1:.3f} p99={p99:.3f} "
          f"(mean={vals.mean():.3f} std={vals.std():.3f})")

    total = 0
    for split in ('train', 'val', 'test'):
        files = sorted(os.listdir(os.path.join(args.split_base, split)))
        out_dir = os.path.join(args.out_base, split)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for f in files:
            if not f.endswith('.npy'):
                continue
            try:
                raw = raw_of(f)
            except FileNotFoundError:
                print(f"  [skip missing sub] {f}"); continue
            raw = np.clip(raw, p1, p99)              # percentile クリップ
            s = raw.std()
            out = (raw - raw.mean()) / s if s > 0 else raw * 0.0  # per-sample zscore
            np.save(os.path.join(out_dir, f), out.astype(np.float32))
            n += 1
        print(f"[plain] {split}: wrote {n} -> {out_dir}")
        total += n
    print(f"[plain] done. total {total} files in {args.out_base}")


if __name__ == '__main__':
    main()
