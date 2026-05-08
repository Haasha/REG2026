#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2

python /mnt/research/data_slow/miccai_challenge/metric_zhengyang/scripts_final/evaluate.py \
    --ground-truth /mnt/research/data_slow/miccai_challenge/metric_zhengyang/scripts_final/utils_metric_A/REG^2_CoT_bladder_v1.json \
    --predictions /mnt/research/data_slow/miccai_challenge/metric_zhengyang/scripts_final/utils_metric_A/REG^2_CoT_bladder_v1.json \
    --visual-json /mnt/research/data_slow/miccai_challenge/metric_zhengyang/scripts_final/utils_metric_B/roi_question_answer_pairs.json \
    --visual-mapping-txt /mnt/research/data_slow/miccai_challenge/metric_zhengyang/scripts_final/utils_metric_B/anonymous_rois_mapping.txt \
    --output /mnt/research/data_slow/miccai_challenge/metric_zhengyang/scripts_final/evaluation_output.json