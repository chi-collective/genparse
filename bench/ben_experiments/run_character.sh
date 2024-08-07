#!/bin/bash

model_names=("Meta-Llama-3.1-8B-Instruct" "Meta-Llama-3-8B-Instruct")

particles=(1 5 10)

start=1

end=1

schema=all

for model_name in "${model_names[@]}"; do
    out_dir="full_results/character/$model_name"

    if [ ! -d "$out_dir" ]; then
        mkdir -p "$out_dir"
    fi

    for n_particles in "${particles[@]}"; do
        for run in $(seq $start $end); do
            exp_name="${model_name}-smc-${run}"

            CUDA_VISIBLE_DEVICES=1 python run_genparse.py \
                --particles "$n_particles" \
                --proposal character \
                --exp-name $exp_name \
                --model-name "meta-llama/$model_name" \
                --out-dir $out_dir \
                --schema $schema 

            echo "done for ${exp_name} with ${n_particles} particles"
        done
    done
done