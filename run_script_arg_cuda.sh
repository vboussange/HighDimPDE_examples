#!/bin/bash
# Args : 
# 1: name of script without .jl
# 2: cuda_device
script=$1

echo "lauching script $script"
date
# Launching on GPU zero
($HOME/utils/julia-1.7.2/bin/julia --project=. --threads=20 $script.jl $2 &> stdout/$script.stdout) 
wait
echo "computation over"
date