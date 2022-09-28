#!/bin/bash

script=$1

echo "lauching script $script"
# Launching on GPU zero
($HOME/utils/julia-1.7.2/bin/julia --project=. --threads=20 $script.jl 0 &> stdout/$script.stdout) 
wait
echo "computation over"
date