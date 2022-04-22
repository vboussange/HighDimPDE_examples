#!/bin/bash

examples=("rep_mut" "nonlocal_comp" "nonlocal_sinegordon" "fisherkpp_neumann" "allencahn_neumann")
for i in {0..4}; do
    echo "lauching script for ${examples["$i"]}"
    # ($HOME/utils/julia-1.7.2/bin/julia --project=. --threads 1 "$pathsim.jl" $SEED $c &> "stdout/${namesim}_$c.out") &
     ($HOME/utils/julia-1.7.2/bin/julia --project=. --threads=10 run_example_files_arg_parse.jl $i ${examples["$i"]} &> stdout/${examples["$i"]}.stdout) &
done
wait
echo "computation over"
date