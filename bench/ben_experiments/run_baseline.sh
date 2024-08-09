#!/bin/bash

model_names=("Meta-Llama-3.1-8B-Instruct" "Meta-Llama-3-8B-Instruct")

particles=(1 5 10)

start=1

end=1

schema=all

DEVICE=0

if [ ! -z "$1" ]; then
    DEVICE=$1
fi


for model_name in "${model_names[@]}"; do
    out_dir="full_results/character/$model_name/greedy-baseline"

    if [ ! -d "$out_dir" ]; then
        mkdir -p "$out_dir"
    fi

    for n_particles in "${particles[@]}"; do
        for run in $(seq $start $end); do
            exp_name="${model_name}-baseline-${run}"

            CUDA_VISIBLE_DEVICES=$DEVICE python run_baseline.py \
                --method greedy \
                --exp-name "${model_name}-baseline" \
                --particles "$n_particles" \
                --model-name "meta-llama/$model_name" \
                --out-dir "results" \
                --schema $schema

            echo "done for ${exp_name} with ${n_particles} particles"
        done
    done
done