#!/bin/sh

for SEED in 1 2 3;
do
    for i in 1 2 4 6 8 16 32;
    do
        CUDA_VISIBLE_DEVICES=3 python ../experiments/lora_forget.py experiment.name=lora_vit_forget_"$i" strategy.lora_rank=$i strategy.lora_alpha=$((2*i)) deploy=cvc_serv model=timvit1k experiment.seed=$SEED
    done
done
