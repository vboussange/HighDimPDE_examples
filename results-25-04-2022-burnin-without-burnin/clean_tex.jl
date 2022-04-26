cd(@__DIR__)
using Latexify
using CSV

file = "Table1-fisher-kpp_neumann.tex"
open(file, "w") do io
    read(io, String)
end

