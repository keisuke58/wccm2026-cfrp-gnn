#!/bin/bash
# =============================================================================
# WCCM2026 CFRP×GNN  ablation runner
# 差分正規化 vs plain正規化 × 幾何特徴 on/off ＋ 新規正則化（GATv2/ノイズ/平滑化/ミラー）
#
# 前提（vancouver サーバー）:
#   - 差分版データ: /home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore/{train,val,test}
#   - plain版データ: $PLAIN_BASE（生DSPSS→zscore。data_prep/ で別途生成しておく）
#   - ラベル:        /home/nishioka/GNN/GNN_hole_2026/all_19class_label
#   - ミラーperm:    mirror_perm.npy（make_mirror_perm.py で生成）
# 4GPU torchrun。各設定は順番に実行（GPUメモリ衝突回避）。
# =============================================================================
set -u
cd "$(dirname "$0")/.."                  # repo root
TRAIN=train.py
NPROC=${NPROC:-4}
EPOCHS=${EPOCHS:-2000}

# NCCL タイムアウト対策
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

DIFF_BASE="/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore"     # 差分正規化
PLAIN_BASE="${PLAIN_BASE:-/home/nishioka/GNN/GNN_hole_2026/all_hole_defect_zscore}"  # plain（要生成）
MIRROR_PERM="${MIRROR_PERM:-mirror_perm.npy}"

run () {  # $1=tag  $2...=extra flags
  local tag="$1"; shift
  local ts; ts=$(date +"%Y%m%d_%H%M%S")
  local log="ablation_${tag}_${ts}.log"
  echo "=== [$tag] $(date) ==="
  echo "flags: $*"
  torchrun --nproc_per_node=${NPROC} ${TRAIN} \
      --epochs ${EPOCHS} --use_onecycle --dropout 0.1 --edge_drop_prob 0.01 \
      --batch_size 128 --hidden_channels 32 \
      --num_workers 4 --cudnn_benchmark \
      "$@" 2>&1 | tee "${log}"
  echo "log -> ${log}"
}

# --- A. 入力正規化 × 幾何特徴（教授論点: 差分 vs plain）---
run diff_base        --data_base "${DIFF_BASE}"
run diff_geo         --data_base "${DIFF_BASE}" --extra_geo_features
run plain_base       --data_base "${PLAIN_BASE}"
run plain_geo        --data_base "${PLAIN_BASE}" --extra_geo_features   # ← 基準不要で差分に肉薄するか

# --- B. 新規正則化（過学習 0.73->0.66 と ノイズ頑健化を攻める）---
run diff_gatv2       --data_base "${DIFF_BASE}" --conv_type gatv2
run diff_noise       --data_base "${DIFF_BASE}" --train_noise_std 0.1 --train_noise_curriculum
run diff_smooth      --data_base "${DIFF_BASE}" --no_logit_adjust --label_smoothing 0.05
run diff_mirror      --data_base "${DIFF_BASE}" --mirror_augment --mirror_perm_path "${MIRROR_PERM}"

# --- C. 全部入り（最有力候補）---
run diff_all  --data_base "${DIFF_BASE}" --conv_type gatv2 --extra_geo_features \
              --train_noise_std 0.1 --train_noise_curriculum \
              --mirror_augment --mirror_perm_path "${MIRROR_PERM}"
run plain_all --data_base "${PLAIN_BASE}" --conv_type gatv2 --extra_geo_features \
              --train_noise_std 0.1 --train_noise_curriculum \
              --mirror_augment --mirror_perm_path "${MIRROR_PERM}"

# --- D. Conv zoo（GATv2 以外のアーキ比較。差分ベースで横並び）---
#   transformer/pna は --edge_geo_features で座標差分エッジ特徴 [dx,dy,dz,dist] を併用すると
#   左右非対称（論文の弱点②）に効きうる。gine は常時エッジ特徴を使用。
run diff_sage         --data_base "${DIFF_BASE}" --conv_type sage
run diff_resgated     --data_base "${DIFF_BASE}" --conv_type resgated
run diff_gin          --data_base "${DIFF_BASE}" --conv_type gin
run diff_gine         --data_base "${DIFF_BASE}" --conv_type gine
run diff_transformer  --data_base "${DIFF_BASE}" --conv_type transformer
run diff_transformer_edge --data_base "${DIFF_BASE}" --conv_type transformer --edge_geo_features
run diff_pna          --data_base "${DIFF_BASE}" --conv_type pna
run diff_pna_edge     --data_base "${DIFF_BASE}" --conv_type pna --edge_geo_features
# MeshGraphNet（Encode-Process-Decode＋エッジ更新。メッシュ応力場SOTA構造）。
run diff_mgn          --data_base "${DIFF_BASE}" --conv_type meshgraphnet --mgn_blocks 10

# --- E. plain正規化アーム（基準不要化＝教授論点。差分アームと横並び比較）---
#   PLAIN_BASE は all_hole_defect_zscore（生DSPSS→zscore, 実在確認済 2026-06-02）。
run plain_gat             --data_base "${PLAIN_BASE}" --conv_type gat
run plain_gat_geo         --data_base "${PLAIN_BASE}" --conv_type gat --extra_geo_features
run plain_gatv2           --data_base "${PLAIN_BASE}" --conv_type gatv2
run plain_sage            --data_base "${PLAIN_BASE}" --conv_type sage
run plain_transformer_edge --data_base "${PLAIN_BASE}" --conv_type transformer --edge_geo_features
run plain_pna_edge        --data_base "${PLAIN_BASE}" --conv_type pna --edge_geo_features
run plain_pna_edge_geo    --data_base "${PLAIN_BASE}" --conv_type pna --edge_geo_features --extra_geo_features
run plain_mgn             --data_base "${PLAIN_BASE}" --conv_type meshgraphnet --mgn_blocks 10

# --- F. mirror拡張アーム（左右非対称＝論文の弱点②。差分ベース）---
#   MIRROR_PERM は make_mirror_perm.py で生成（反転軸=周方向z, 57分割の左右軸）:
#     python make_mirror_perm.py --fixed <normalized_x_2layer.npy> \
#            --reflect <normalized_z_2layer.npy> --out mirror_perm.npy
#   座標元: /home/nishioka/GNN/GNN_hole/GNN_hole_data/（2026版学習もこの座標を使用）。
#   生成時 coverage=1.000 / involution=1.000 を確認すること（厳密な左右対合）。
run mirror_gat            --data_base "${DIFF_BASE}" --conv_type gat   --mirror_augment --mirror_perm_path "${MIRROR_PERM}"
run mirror_gatv2_geo      --data_base "${DIFF_BASE}" --conv_type gatv2 --extra_geo_features --mirror_augment --mirror_perm_path "${MIRROR_PERM}"
run mirror_pna_edge       --data_base "${DIFF_BASE}" --conv_type pna --edge_geo_features --mirror_augment --mirror_perm_path "${MIRROR_PERM}"
run mirror_transformer_edge --data_base "${DIFF_BASE}" --conv_type transformer --edge_geo_features --mirror_augment --mirror_perm_path "${MIRROR_PERM}"

echo "=== ablation done. group-purged 正直評価は各runに --group_purge_eval を足す ==="
