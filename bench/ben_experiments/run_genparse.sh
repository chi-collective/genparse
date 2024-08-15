#!/bin/bash

model_names=("Meta-Llama-3.1-8B-Instruct")
particles=(10 5 1)
start=1
end=1
schema=all
DEVICE=0
METHOD=smc
PROPOSAL=character
ess_thresholds=(0.5)
K=0

function show_help() {
    echo "Usage: $0 [--device DEVICE] [--method METHOD] [--particles PARTICLES] [--models MODELS] [--proposal PROPOSAL] [--K K] [--ess-thresholds ESS_THRESHOLDS]"
    echo ""
    echo "Arguments:"
    echo "  --device DEVICE                  The CUDA device ID to use (default: 0)"
    echo "  --method METHOD                  The method to use, either 'smc' or 'local' (default: 'smc')"
    echo "  --particles PARTICLES            A comma-separated list of particle numbers (default: '10,5,1')"
    echo "  --models MODELS                  A comma-separated list of model names (default: 'Meta-Llama-3.1-8B-Instruct')"
    echo "  --proposal PROPOSAL              Proposal distribution to use, either 'token' or 'character' (default: 'character')"
    echo "  --K K                            Parameter for token proposal (default: 0). Must be 0 if PROPOSAL is 'character'."
    echo "  --ess-thresholds ESS_THRESHOLDS  A comma-separated list of ESS threshold values (default: '0.5')."
    exit 1
}

function is_numeric() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            if ! is_numeric "$2"; then
                echo "Error: DEVICE must be a numeric value."
                show_help
            fi
            DEVICE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            if [ "$METHOD" != "smc" ] && [ "$METHOD" != "local" ]; then
                echo "Error: METHOD must be 'smc' or 'local'."
                show_help
            fi
            shift 2
            ;;
        --particles)
            IFS=',' read -r -a particles <<< "$2"
            for n in "${particles[@]}"; do
                if ! is_numeric "$n"; then
                    echo "Error: PARTICLES must be a comma-separated list of numeric values."
                    show_help
                fi
            done
            shift 2
            ;;
        --models)
            IFS=',' read -r -a model_names <<< "$2"
            if [ -z "$model_names" ]; then
                echo "Error: MODELS cannot be empty."
                show_help
            fi
            shift 2
            ;;
        --proposal)
            if [ "$2" != "token" ] && [ "$2" != "character" ]; then
                echo "Error: PROPOSAL must be either 'token' or 'character'."
                show_help
            fi
            PROPOSAL="$2"
            shift 2
            ;;
        --K)
            if [ "$PROPOSAL" == "character" ] && [ "$2" -ne 0 ]; then
                echo "Error: K must be 0 when PROPOSAL is 'character'."
                show_help
            fi
            if [ "$2" != "None" ] && ! is_numeric "$2" && [ "$2" != "0" ]; then
                echo "Error: K must be None or an integer when PROPOSAL is 'token'."
                show_help
            fi
            K="$2"
            shift 2
            ;;
        --ess-thresholds)
            IFS=',' read -r -a ess_thresholds <<< "$2"
            for ess in "${ess_thresholds[@]}"; do
                if ! is_numeric "$ess"; then
                    echo "Error: ESS_THRESHOLD must be a comma-separated list of numeric values."
                    show_help
                fi
            done
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            show_help
            ;;
    esac
done

if [ "$PROPOSAL" == "character" ] && [ "$K" -ne 0 ]; then
    echo "Error: K must be 0 when PROPOSAL is 'character'."
    exit 1
fi

for model_name in "${model_names[@]}"; do
    out_dir="results/$model_name/$PROPOSAL/$METHOD"

    if [ ! -d "$out_dir" ]; then
        mkdir -p "$out_dir"
    fi

    for n_particles in "${particles[@]}"; do
        for ess_threshold in "${ess_thresholds[@]}"; do
            for run in $(seq $start $end); do
                exp_name="${model_name}-$METHOD-${run}-${ess_threshold}"

                CUDA_VISIBLE_DEVICES=$DEVICE python scripts/run_genparse.py \
                    --particles $n_particles \
                    --proposal $PROPOSAL \
                    --K $K \
                    --exp-name $exp_name \
                    --model-name "meta-llama/$model_name" \
                    --out-dir $out_dir \
                    --schema $schema \
                    --verbosity 0 \
                    --ess-threshold $ess_threshold \
                    $( [ "$METHOD" == "local" ] && echo "--local-poe" )

                echo "Done ${exp_name} with ${n_particles} particles and ESS threshold ${ess_threshold}"
            done
        done
    done
done
