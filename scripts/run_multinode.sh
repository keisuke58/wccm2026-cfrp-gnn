#!/bin/bash
# =============================================================================
# WCCM2026 CFRP×GNN  マルチノード分散学習起動スクリプト
#
# 対象ホスト:
#   NODE0 = vancouver01 (192.168.10.121, 4× RTX 4090 24GB)  — master
#   NODE1 = stuttgart01 (192.168.10.111, 4× RTX 3090 24GB)  — worker
#   合計 8GPU / world_size=8
#
# 使い方（ローカルPC から実行）:
#   bash scripts/run_multinode.sh
#
# または各ノードで手動実行:
#   NODE0 側: LAUNCH_ROLE=master bash scripts/run_multinode.sh
#   NODE1 側: LAUNCH_ROLE=worker bash scripts/run_multinode.sh
#
# 前提:
#   - ssh config に vancouver01 / stuttgart01 が ProxyJump copaam で定義済み
#   - NFS ホームが全ノードで共有 (/home/nishioka)
#   - miniconda3 は NFS 上の /home/nishioka/miniconda3
# =============================================================================
set -euo pipefail

# ---- 設定 ----
MASTER_ADDR="192.168.10.121"          # vancouver01 の IP
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES=2
NPROC_PER_NODE=4                      # 各ノードの GPU 数
WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))  # 8

PYTHON="/home/nishioka/miniconda3/bin/python3.12"
TORCHRUN="/home/nishioka/miniconda3/bin/torchrun"
REPO="/home/nishioka/git/wccm2026-cfrp-gnn"
TRAIN="${REPO}/train.py"

# ---- 実験設定（変更可能） ----
CONV_TYPE="${CONV_TYPE:-hybridmgn}"
HIDDEN="${HIDDEN:-32}"
EPOCHS="${EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"        # 8GPU で batch_size=32 → 実効 256
DATA_BASE="${DATA_BASE:-/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore}"

TS=$(date +"%Y%m%d_%H%M%S")

# ---- 共通環境変数 ----
ENV_VARS=(
  "LD_LIBRARY_PATH=/home/nishioka/miniconda3/lib:\${LD_LIBRARY_PATH:-}"
  "MPLBACKEND=Agg"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  "NCCL_ASYNC_ERROR_HANDLING=1"
  "NCCL_BLOCKING_WAIT=0"
  "NCCL_DEBUG=WARN"
  "NCCL_TIMEOUT=1800"
  "NCCL_IB_DISABLE=1"                 # IB 未使用環境では無効化
  "NCCL_SOCKET_IFNAME=eth0"           # ネットワーク IF（環境に合わせて変更）
  "OMP_NUM_THREADS=2"
)
EXPORT_STR=$(printf "export %s; " "${ENV_VARS[@]}")

# ---- torchrun 共通フラグ ----
COMMON_TRAIN_ARGS=(
  "--conv_type ${CONV_TYPE}"
  "--hidden_channels ${HIDDEN}"
  "--epochs ${EPOCHS}"
  "--batch_size ${BATCH_SIZE}"
  "--use_onecycle"
  "--dropout 0.1"
  "--edge_drop_prob 0.01"
  "--data_base ${DATA_BASE}"
  "--num_workers 4"
  "--use_amp"
  "--cudnn_benchmark"
)

# ---- ログ出力 ----
LOG_DIR="${REPO}/logs_multinode"
mkdir -p "${LOG_DIR}"
LOG0="${LOG_DIR}/multinode_${TS}_node0.log"
LOG1="${LOG_DIR}/multinode_${TS}_node1.log"

echo "================================================================"
echo "  WCCM2026 Multinode Launch"
echo "  MASTER: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Nodes: ${NNODES} | GPUs/node: ${NPROC_PER_NODE} | World: ${WORLD_SIZE}"
echo "  conv_type=${CONV_TYPE} hidden=${HIDDEN} epochs=${EPOCHS} bs=${BATCH_SIZE}"
echo "================================================================"

# ---- NODE 0 (vancouver01) ----
echo "[$(date)] Launching NODE 0 on vancouver01 ..."
ssh -o StrictHostKeyChecking=no vancouver01 "
  ${EXPORT_STR}
  export PATH=/home/nishioka/miniconda3/bin:\$PATH
  nohup ${TORCHRUN} \
    --nnodes=${NNODES} \
    --node_rank=0 \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${TRAIN} ${COMMON_TRAIN_ARGS[*]} \
    > ${LOG0} 2>&1 &
  echo \$! > /tmp/wccm_node0_${TS}.pid
  echo 'Node0 PID: '\$(cat /tmp/wccm_node0_${TS}.pid)
" &

sleep 3

# ---- NODE 1 (stuttgart01) ----
echo "[$(date)] Launching NODE 1 on stuttgart01 ..."
ssh -o StrictHostKeyChecking=no stuttgart01 "
  ${EXPORT_STR}
  export PATH=/home/nishioka/miniconda3/bin:\$PATH
  nohup ${TORCHRUN} \
    --nnodes=${NNODES} \
    --node_rank=1 \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${TRAIN} ${COMMON_TRAIN_ARGS[*]} \
    > ${LOG1} 2>&1 &
  echo \$! > /tmp/wccm_node1_${TS}.pid
  echo 'Node1 PID: '\$(cat /tmp/wccm_node1_${TS}.pid)
"

echo ""
echo "================================================================"
echo "  Jobs submitted!"
echo "  NODE0 log: ${LOG0}"
echo "  NODE1 log: ${LOG1}"
echo ""
echo "  Monitor: ssh vancouver01 'tail -f ${LOG0}'"
echo "  Stop:    ssh vancouver01 'kill \$(cat /tmp/wccm_node0_${TS}.pid)'"
echo "           ssh stuttgart01 'kill \$(cat /tmp/wccm_node1_${TS}.pid)'"
echo "================================================================"
