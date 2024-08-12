#!/bin/bash

model_names=("Meta-Llama-3.1-8B-Instruct")

particles=(10 5 1)

start=1

end=1

schema=all

DEVICE=0
METHOD=smc

function show_help() {
    echo "Usage: $0 [--device DEVICE] [--method METHOD]"
    echo ""
    echo "Arguments:"
    echo "  --device DEVICE   The CUDA device ID to use (default: 0)"
    echo "  --method METHOD   The method to use, either 'smc' or 'improper-weights' (default: 'smc')"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            show_help
            ;;
    esac
done

if [ "$METHOD" != "smc" ] && [ "$METHOD" != "improper-weights" ]; then
    echo "Error: METHOD must be 'smc' or 'improper-weights'."
    show_help
fi

for model_name in "${model_names[@]}"; do
    out_dir="full_results/$model_name/token/$METHOD"

    if [ ! -d "$out_dir" ]; then
        mkdir -p "$out_dir"
    fi

    for n_particles in "${particles[@]}"; do
        for run in $(seq $start $end); do
            exp_name="${model_name}-$METHOD-$run"

            CUDA_VISIBLE_DEVICES=$DEVICE python scripts/run_genparse.py \
                --particles $n_particles \
                --proposal token \
                --K 10 \
                --exp-name $exp_name \
                --model-name "meta-llama/$model_name" \
                --out-dir $out_dir \
                --schema $schema \
                --verbosity 0

            echo "done for ${exp_name} with ${n_particles} particles"
        done
    done
done