#!/usr/bin/env bash

# SceneFlow training with Depth Anything V3 Small as the monocular feature extractor.
# Requires DA3 small weights under checkpoints/da3-small or DEPTH_ANYTHING_3_DA3_SMALL_DIR.

CHECKPOINT_DIR=checkpoints/defomstereo_da3s_corr_noagg && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=9991 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 \
--name defomstereo_da3s_corr \
--batch_size 4  \
--num_workers 2  \
--train_datasets sceneflow \
--train_folds 1 \
--num_steps 260000 \
--mixed_precision \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder da3s \
--lr 0.00026 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
