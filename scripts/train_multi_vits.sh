#!/usr/bin/env bash

# trained on 2 x 24GB 3090/4090 GPUs

CHECKPOINT_DIR=checkpoints/defomstereo_vits_multi && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=9990 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 \
--name defomstereo_vits_multi \
--batch_size 4  \
--num_workers 4  \
--train_datasets  sceneflow sintel_stereo falling_things vkitti2 instereo2k crestereo \
--train_folds 1 50 1 1 10 1 \
--num_steps 300000 \
--mixed_precision \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vits \
--lr 0.0003 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


