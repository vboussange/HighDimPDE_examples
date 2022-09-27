#!/bin/bash

echo "lauching script"
# Launching on GPU zero
($HOME/utils/julia-1.7.2/bin/julia --project=. --threads=20 run_param_explo.jl 0 &> stdout/run_param_explo.stdout) 
wait
echo "computation over"
date