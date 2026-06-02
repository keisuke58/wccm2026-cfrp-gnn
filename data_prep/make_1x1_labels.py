#!/usr/bin/env python3
"""1x1 微小欠陥の 19class ラベルを生成（OOD評価用）。

学習データ(2x2/4x4)と同一の canonical ロジックを使用:
  - rpt_19class_2x2.py (inner, L<=10):  processed = [n - 7125*(L-1) - 1  for n if n <= 7125*L]
  - rpt_19class_outer_2x2.py (outer, L>=11): processed = [n - 7125*L - 1 for n if 7125*(L+1) <= n]
  - 共通: labels[node][L-1] = 1, 非欠陥=class0, hole 308要素を np.delete → 13942ノード
L値はファイル名から取得し inner/outer を自動切替する（1x1専用スクリプトの座標規約問題を避ける）。
"""
import argparse, os, re
from glob import glob
import numpy as np

TARGET = "Field Output reported at element nodes for region: CFRP_INTERSTAGE_HOLE-1.Region_22"
NUM_CLASSES = 19
TOTAL_NODES = 7125 * 2  # 14250 (穴除去前)


def hole_elements():
    num_rows = 57
    total_elements = 125 * 57
    cfg = [(26, 51, 7, 7), (26, 77, 7, 15)]  # (row_start, col_start, size_rows, size_cols)
    hs = []
    for offset in (0, total_elements):
        for (rs, cs, sr, sc) in cfg:
            for c in range(cs, cs + sc):
                for r in range(rs, rs + sr):
                    hs.append(c * num_rows + (num_rows - r + 1) + offset)
    return hs


HOLE = hole_elements()


def extract_defect_nodes(file_path):
    nodes = set()
    found = False
    with open(file_path) as f:
        for line in f:
            if TARGET in line:
                found = True
                continue
            if found:
                m = re.search(r"^\s*\d+\s+(\d+)\s+[\d.E+-]+\s+[\d.E+-]+\s+[\d.E+-]+", line)
                if m:
                    nodes.add(int(m.group(1)))
    return sorted(nodes)


def make_label(file_path, L):
    nodes = extract_defect_nodes(file_path)
    if L <= 10:  # inner
        processed = [n - 7125 * (L - 1) - 1 for n in nodes if n <= 7125 * L]
    else:        # outer
        processed = [n - 7125 * L - 1 for n in nodes if 7125 * (L + 1) <= n]
    labels = np.zeros((TOTAL_NODES, NUM_CLASSES), dtype=np.int8)
    for node in processed:
        if 0 <= node < TOTAL_NODES:
            labels[node][L - 1] = 1
    return np.delete(labels, HOLE, axis=0), len(processed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rpt_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob(os.path.join(args.rpt_dir, '*.rpt')))
    print(f"[labels] {len(files)} RPT files")
    n_ok, n_empty = 0, 0
    for fp in files:
        name = os.path.basename(fp)
        m = re.search(r"L(\d+)", name)
        if not m:
            continue
        L = int(m.group(1))
        try:
            lab, k = make_label(fp, L)
        except Exception as e:
            print(f"  ERR {name}: {e}"); continue
        if k == 0:
            n_empty += 1
        if lab.shape[0] != 13942:
            print(f"  ⚠ shape {lab.shape} (expect 13942): {name}"); continue
        out = os.path.join(args.out_dir, f"{os.path.splitext(name)[0]}_19label.npy")
        np.save(out, lab.astype(np.int8))
        n_ok += 1
    print(f"[labels] wrote {n_ok}/{len(files)} (defect-empty={n_empty}) -> {args.out_dir}")


if __name__ == '__main__':
    main()
