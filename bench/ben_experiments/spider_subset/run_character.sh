#!/bin/bash

# List of model names
model_names=("Meta-Llama-3.1-8B-Instruct" "Meta-Llama-3-8B-Instruct")

# List of particles
particles=(1 10 20 50 100)

schema=concert_singer,pets_1,museum_visit,employee_hire_evaluation,tvshow

for model_name in "${model_names[@]}"; do
    for n_particles in "${particles[@]}"; do
        python run_baseline.py \
            --method sampling \
            --exp-name "${model_name}-baseline" \
            --particles "$n_particles" \
            --model-name "meta-llama/$model_name" \
            --out-dir "results" \
            --schema $schema

        echo "baseline done for ${model_name} with ${n_particles} particles"

        CUDA_VISIBLE_DEVICES=0 python run_genparse.py \
            --particles "$n_particles" \
            --proposal character \
            --exp-name  "${model_name}-smc-new" \
            --model-name "meta-llama/$model_name" \
            --out-dir "results" \
            --schema $schema

        echo "smc done for ${model_name} with ${n_particles} particles"

    done
done