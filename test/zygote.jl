using Distributed
using PhyloTrees
using CSV
using DataFrames
using Zygote
addprocs(2)
@everywhere using DistributedArrays
@everywhere using Beluga

s = SpeciesTree(readtree("test/data/tree1.nw"))
df = CSV.read("test/data/counts1.csv", delim="\t"); deletecols!(df, :Orthogroup)
d = DLModel(s, 0.2, 0.3)
n = length(d.b)
X = profile(s, df)
D = distribute(X, dist=(length(workers()),1))

function gradient(d::DLModel, X::AbstractMatrix{Int64}, cond=:oib)
    v = Beluga.asvector(d)
    f = (u) -> logpdf(DLModel(d, u), X, cond)
    g = Zygote.gradient(f, v)
    return g
end

gradient(d, X)
