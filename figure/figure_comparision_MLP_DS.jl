#=
Comparing time taken for replicator mutator, 
with replicator mutator example

For now, replace by script "complexity_vs_error.jl"
=#

cd(@__DIR__)
using HighDimPDE
using PyPlot
using JLD2
using UnPack
using DataFrames

path_results = "../results/2022-09-28/"

dict_results_MLP =load(path_results*"/explo_param_MLP_T=0.5/dict_results_MLP_param_explo.jld2")
dict_results_DS =load(path_results*"/explo_param_DS_T=0.2/dict_results_MLP_param_explo.jld2")

fig, axs = subplots(1,3, figsize = (8,2))