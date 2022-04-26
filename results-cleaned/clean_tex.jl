#=
Read tables without burnin results
and replaces numerical values by nice latexified numbers.

Outputs in `clean_tables`.
=#
cd(@__DIR__)
using Latexify
using CSV
using Glob
dir_tables = "25-04-2022/"
isdir(dir_tables*"clean_tables") ? nothing : mkdir(dir_tables*"clean_tables")
files = glob(dir_tables*"Table*.tex")

for file in files
    name = basename(file)
    tex_string = read(file, String)
    tex_string = replace(tex_string, ".0000000" => "", 
                                    "0.1000000" => "\\nicefrac{1}{10}", 
                                    "0.2000000" => "\\nicefrac{1}{5}",
                                    "0.5000000" => "\\nicefrac{1}{2}", )
    println(tex_string)
    open(dir_tables*"clean_tables/$name", "w") do io
        write(io,tex_string);
    end
end
