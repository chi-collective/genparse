#!/bin/bash

for proposal in character token
do
    for model in codellama
    do
        for particles in 10
        do
            for inference in smc-steer
            do
                for max_tokens in 120
                do
                    for script in benchmark/benchmark_vllm_inference.py benchmark/benchmark_inference.py
                    do
                        if [ "$inference" == "smc-steer" ]; then
                            for nbeam in 3
                            do
                                command="python $script --model $model --proposal $proposal --particles $particles --inference $inference --max-tokens $max_tokens --n-beam $nbeam"
                            done
                        else
                            command="python $script --model $model --proposal $proposal --particles $particles --inference $inference --max-tokens $max_tokens"                    
                        fi
                        echo Running: $command
                        export CUDA_VISIBLE_DEVICES=0
                        $command
                    done
                done
            done
        done
    done
done
