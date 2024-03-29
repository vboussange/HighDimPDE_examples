#=
Plotting reflection 

=#
cd(@__DIR__)
using PyPlot
using HighDimPDE
using LinearAlgebra
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] =14

"""
    function reflect_once(a::T, b::T, s::T, e::T) where T <: Vector

Reflects once, providing b and c 
"""
function reflect_once(a::T, b::T, s::T, e::T) where T <: Vector
    r = 2; n = zeros(size(a))
    # first checking if b is in the hypercube
    all((a .>= s) .& (a .<= e)) ? nothing : error("a = $a not in hypercube")
    size(a) == size(b) ? nothing : error("a not same dim as b")
    for i in 1:length(a)
        if b[i] < s[i]
            rtemp = (a[i] - s[i]) / (a[i] - b[i])
            if rtemp < r
                r = rtemp
                n .= 0
                n[i] = -1
            end
        elseif  b[i] > e[i]
            rtemp =  (e[i] - a[i]) / (b[i]- a[i])
            if rtemp < r
                r = rtemp
                n .= 0
                n[i] = 1
            end
        end
    end
    c = a + r * ( b - a )
    a = c
    b = b - 2 * n * ( dot(b-c,n))
    r = 2;
    for i in 1:length(a)
        if b[i] < s[i]
            rtemp = (a[i] - s[i]) / (a[i] - b[i])
            if rtemp < r
                r = rtemp
                n .= 0
                n[i] = -1
            end
        elseif  b[i] > e[i]
            rtemp =  (e[i] - a[i]) / (b[i]- a[i])
            if rtemp < r
                r = rtemp
                n .= 0
                n[i] = 1
            end
        end
    end
    return b,c
end

# Plotting on half plane figure
fig, ax = plt.subplots(1)
ax.axis("off")
d = 2
X0 = [-0.9,-0.3]
X1 = X0 .+ (1.2, -1)
X11, c = reflect_once(X0, X1, fill(-1.,d), fill(1.,d))

# half plane
ax.plot([-1,1],[-1,-1], c = "black", linewidth=3.) # half plane
display(fig)

ax.plot([X0[1],X1[1]],[X0[2],X1[2]], c = "tab:blue", ) # original segment
display(fig)
ax.plot([c[1],X11[1]],[c[2],X11[2]], c = "tab:blue") # reflected segment
display(fig)
# projection y axis
ax.plot([X1[1],X11[1]],[X1[2],X11[2]], c = "tab:red", linestyle = "dashed")
display(fig)
# projection x axis
ax.plot([c[1],X1[1]],[X1[2],X1[2]], c = "tab:green", linestyle = "dashed")
display(fig)
# normal vector
ax.annotate("", xytext=(c[1],-1), xy=(c[1],-1.2), arrowprops=Dict("arrowstyle"=>"->, head_width=0.3", "color"=>"tab:red"))
# ax.arrow(c[1],-1.,0,-0.1,color="tab:red", head_length=0.1,head_width=0.1) 
display(fig)
# n<n,b-c>
ax.annotate("", xytext=(c[1],c[2]), xy=(c[1],X1[2]), arrowprops=Dict("arrowstyle"=>"->, head_width=0.3", 
                                                                "color"=>"tab:red",
                                                                "linestyle"=>"dashed"), 
                                                                )
display(fig)

# plotting dots
ax.scatter([X0[1]],[X0[2]], c = "tab:blue", zorder = 10)
ax.annotate(L"a",([X0[1]],[X0[2]]), (5e0, 5e0), textcoords="offset points")
display(fig)
ax.scatter([X1[1]],[X1[2]], c = "tab:blue", zorder = 10)
ax.annotate(L"b",(X1[1],X1[2]), (5e0, 5e0), textcoords="offset points")
display(fig)
ax.scatter([X11[1]],[X11[2]], c = "tab:blue", zorder = 10)
ax.annotate(L"b - 2a \langle a, b-c \rangle",(X11[1],X11[2]), (5e0, 5e0), textcoords="offset points")
display(fig)
ax.scatter([c[1]],[c[2]], c = "tab:blue", zorder = 10)
ax.annotate(L"c",(c[1],c[2]), (0, 1.5e1), textcoords="offset points")
display(fig)
ax.annotate(L"n",(c[1],c[2]), (-1.5e1, -4e1), textcoords="offset points", c="tab:red")
display(fig)
ax.annotate(L"n\langle n, b-c\rangle",(c[1],c[2]), (-8e1, -7.e1), textcoords="offset points", c="tab:red")
display(fig)
fig.savefig("reflection_half_plane.pdf", dpi=300)



# Plotting on squared fig
fig, ax = plt.subplots(1)
ax.axis("off")
d = 2
X0 = [-0.9,-0.3]
X1 = X0 .+ (3.2, -3.4)

# square plane
ax.plot([-1,1],[-1,-1], c = "black", linewidth=3.) # half plane
ax.plot([-1,1],[1,1], c = "black", linewidth=3.) # half plane
ax.plot([-1,-1],[-1,1], c = "black", linewidth=3.) # half plane
ax.plot([1,1],[-1,1], c = "black", linewidth=3.) # half plane
display(fig)

ax.scatter([X0[1]],[X0[2]], c = "tab:blue", zorder = 10)
ax.annotate(L"a", (X0[1],X0[2]),(-5e0, -2e1), textcoords="offset points",)
display(fig)

coord_refs = []
for r in 1:3
    X11, c = reflect_once(X0, X1, fill(-1.,d), fill(1.,d))

    ax.plot([X0[1],X1[1]],[X0[2],X1[2]], c = "tab:blue", ) # original segment
    display(fig)

    ax.scatter([X1[1]],[X1[2]], c = "tab:blue", zorder = 10)
    push!(coord_refs, (X1[1],X1[2]))
    X0 = c
    X1 = X11
end

ax.annotate(L"b = P(\mathcal{R}_0(a,b))", coord_refs[1], (5e0, 5e0), textcoords="offset points",)
display(fig)
ax.annotate(L"P(\mathcal{R}_1(a,b))", coord_refs[2], (5e0, 5e0), textcoords="offset points",)
display(fig)
ax.annotate(L"P(\mathcal{R}_2(a,b))", coord_refs[3], (5e0, 5e0), textcoords="offset points",)
display(fig)

ax.plot([c[1],X1[1]],[c[2],X1[2]], c = "tab:blue", ) # original segment
display(fig)
ax.scatter([X1[1]],[X1[2]], c = "tab:blue", zorder = 10)
ax.annotate(L"P(\mathcal{R}_4(a,b)) = R(a,b)", (X1[1],X1[2]), (-4e1, -1.5e1), textcoords="offset points",)
display(fig)

# plotting dots
# ax.scatter([X0[1]],[X0[2]], c = "tab:blue", zorder = 10)

display(fig)
fig.savefig("reflection_cube.pdf", dpi=300, bbox_inches="tight")