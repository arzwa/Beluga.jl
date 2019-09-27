
# attempt to get Turing working with distributed logpdf
using Distributed
using PhyloTrees
using CSV
using DataFrames
#addprocs(2)
@everywhere using DistributedArrays
@everywhere using Beluga
@everywhere using Turing
#Turing.setadbackend(:reverse_diff)

s = SpeciesTree("test/data/tree1.nw")
df = CSV.read("test/data/counts1.tsv", delim="\t"); deletecols!(df, :Orthogroup)
n = length(s.tree.nodes)
X = profile(s, df)
m = maximum(X)

@model dlmodel(x, ::Type{TV}=Vector{Float64}) where TV = begin
    θ ~ MvLogNormal([log(1.), log(1.)], [1. 0.9; 0.9 1.])
    λ ~ MvLogNormal(repeat([log(θ[1])], n), ones(n))
    μ ~ MvLogNormal(repeat([log(θ[2])], n), ones(n))
    η ~ Beta(10, 1)
    x ~ DuplicationLoss(s, λ, μ, η, m)
end


d = dlmodel(X)
chain = sample(d, HMC(1000, 0.05, 10));
