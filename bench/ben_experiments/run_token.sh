#!/bin/bash

# List of model names
model_names=("Meta-Llama-3.1-8B-Instruct") #"Meta-Llama-3-8B-Instruct"

# List of particles
particles=(1 10 20 50)

schema=concert_singer,pets_1,museum_visit,employee_hire_evaluation,tvshow

CUDA_DEVICES=0

for model_name in "${model_names[@]}"; do
    for n_particles in "${particles[@]}"; do
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python run_genparse.py \
            --particles "$n_particles" \
            --proposal token \
            --K 10 \
            --exp-name  "${model_name}-improper-weight" \
            --model-name "meta-llama/$model_name" \
            --out-dir "results" \
            --schema $schema \
            --improper-weights

        echo "smc done for ${model_name} with ${n_particles} particles"

    done
done
