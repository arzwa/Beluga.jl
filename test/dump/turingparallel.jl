
# attempt to get Turing working with distributed logpdf
using Distributed
using PhyloTrees
using CSV
using DataFrames
addprocs(2)
@everywhere using DistributedArrays
@everywhere using Beluga
@everywhere using Turing
#Turing.setadbackend(:reverse_diff)

s = SpeciesTree(readtree("test/data/tree1.nw"))
df = CSV.read("test/data/counts1.csv", delim="\t"); deletecols!(df, :Orthogroup)
d = DLModel(s, 0.2, 0.3)
n = length(d.b)
X = profile(s, df)

@model dlmodel(x, ::Type{TV}=Vector{Float64}) where TV = begin
    θ ~ MvLogNormal([log(1.), log(1.)], [1. 0.9; 0.9 1.])
    λ ~ MvLogNormal(repeat([log(θ[1])], n), ones(n))
    μ ~ MvLogNormal(repeat([log(θ[2])], n), ones(n))
    η ~ Beta(10, 1)
    x ~ DLModel(s, λ, μ, η)
end

@model dlmodel(x) = begin
    λ = zeros(Real, n)
    μ = zeros(Real, n)
    for i=1:n
        λ[i] ~ LogNormal(log(0.2), 0.2)
        μ[i] ~ LogNormal(log(0.2), 0.2)
    end
    η ~ Beta(10, 1)
    x ~ DLModel(s, λ, μ, η)
end

D = distribute(X, dist=(length(workers()),1))
m = dlmodel(D)
chn = sample(m, HMC(1000, 0.1, 5))

# KissThreading and Mohammed's branch (broken)
using Turing
using Beluga
using PhyloTrees
using CSV
using DataFrames
@model dlmodel(x)= begin
    θ ~ MvLogNormal([log(1.), log(1.)], [1. 0.9; 0.9 1.])
    λ ~ MvLogNormal(repeat([log(θ[1])], n), ones(n))
    μ ~ MvLogNormal(repeat([log(θ[2])], n), ones(n))
    η ~ Beta(10, 1)
    for i=1:length(x)
        x[i] ~ DLModel(s, λ, μ, η)
    end
end

s = SpeciesTree(readtree("test/data/tree1.nw"))
df = CSV.read("test/data/counts1.csv", delim="\t"); deletecols!(df, :Orthogroup)
d = DLModel(s, 0.2, 0.3)
n = length(d.b)
X = profile(s, df)
m = dlmodel([X[i,:] for i=1:size(X)[1]])
chn = sample(m, HMC(1000, 0.1, 5))

# ForwardDiff dual tag issue
using ForwardDiff
function gradient(d::DLModel, x::AbstractMatrix{Int64}, cond=:oib)
    v = Beluga.asvector(d)
    f = (u) -> logpdf(DLModel(d, u), x, cond)
    g = ForwardDiff.gradient(f, v)
    return g[:, 1]
end
