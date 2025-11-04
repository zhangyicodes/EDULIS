#!/usr/bin/env bash

#export TORCH_DISTRIBUTED_DEBUG=INFO
#CONFIG="condinst_r50_fpn_ms-poly-90k_coco_instance.py"
#GPUS=2
#NNODES=${NNODES:-1}
#NODE_RANK=${NODE_RANK:-0}
#PORT=${PORT:-29502}
#MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
#
##CHECKPOINT="/media/ubuntu/DATA2T/mmdetection-main/work_dirs/paper2/edge_unfolding_mask_num_4/iter_50000.pth"
#
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch \
#    --nnodes=$NNODES \
#    --node_rank=$NODE_RANK \
#    --master_addr=$MASTER_ADDR \
#    --nproc_per_node=$GPUS \
#    --master_port=$PORT \
#    $(dirname "$0")/train.py \
#    --config "/media/ubuntu/DATA2T/mmdetection-main/configs/condinst/${CONFIG}" \
#    --launcher pytorch \
#    --work-dir "/media/ubuntu/DATA2T/mmdetection-main/work_dirs/paper2/condinst" ${@:3}
##    --resume ${CHECKPOINT} \



export TORCH_DISTRIBUTED_DEBUG=INFO
CONFIG="condinst_r50_fpn_ms-poly-90k_coco_instance.py"
GPUS=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

#CHECKPOINT="/media/ubuntu/DATA2T/mmdetection-main/work_dirs/paper2/edge_unfolding_mask_num_4/iter_50000.pth"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config "/media/ubuntu/DATA2T/mmdetection-main/configs/condinst/${CONFIG}" \
    --launcher pytorch \
    --work-dir "/media/ubuntu/DATA2T/mmdetection-main/work_dirs/paper2/edulis_wouifpnaux" ${@:3}
#    --resume ${CHECKPOINT} \

