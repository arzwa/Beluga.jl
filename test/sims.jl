using Pkg; Pkg.activate("/home/arzwa/julia-dev/Beluga/")
using Test, DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, PhyloTree
df = CSV.read("test/data/N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end


function set_randrates!(model, d)
    for (i, n) in model.nodes
        x = rand(d)
        n[:λ] = exp(x[1])
        n[:μ] = exp(x[2])
        @show i, n[:λ, :μ]
    end
end


begin
    d = MvNormal(log.([2., 2.]), 0.1I)
    m, y = DuplicationLossWGDModel(nw, df[1:2,:], 2., 2., 0.9)
    set_randrates!(m, d)
    p = PArray()
    σ = 1
    wgds = []
    for i=1:8
        n, t = Beluga.randpos(m)
        q = rand(Beta(1.5,4))
        wgdnode = insertwgd!(m, n, t, q)
        child = nonwgdchild(wgdnode)
        push!(wgds, (child.i, Beluga.parentdist(child, wgdnode), q))
    end
    display(wgds)
end


df = rand(m, 1500)
clade1 = [:bvu, :sly, :ugi, :cqu]
clade2 = setdiff(names(df), clade1)
df = df[sum.(eachrow(df[clade1])) .!= 0, :]
@show size(df)
df = df[sum.(eachrow(df[clade2])) .!= 0, :]
@show size(df)
df = df[1:1000,:]
CSV.write("../../rjumpwgd/data/sims/model1_8wgd_N=1000.csv", df)
open("../../rjumpwgd/data/sims/model1_8wgd_params.csv", "w") do f
    write(f,"nonwgdchild,t,q\n")
    write(f, join([join(x, ",") for x in wgds], "\n"))
end
