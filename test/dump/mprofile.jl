using Test
using Beluga, DataFrames, PhyloTrees, DistributedArrays, Distributions, CSV
import Beluga: AbstractProfile, minfs, Chain
include("../../src/mprofile.jl")

s = SpeciesTree("test/data/plants1.nw")
df = CSV.read("test/data/plants1-100.tsv", delim=",")
deletecols!(df, :Orthogroup)

K = 4
p, m = MixtureProfile(df, s, K)

d = [DuplicationLoss(s, rand(19), rand(19), 1/1.5, m) for i=1:K]
for i = 1:K
    logpdf!(d[i], p, i, -1)
    set_L!(p, i)
end

@test logpdf!(d[1], p, 1, -1) ==
    logpdf!(d[2], p, 2, -1) == logpdf!(d[3], p, 3, -1)

@test logpdf!(d[1], p, 1, -1) == logpdf!(d, p, -1)

logpdf_allother!(d[1], p, 1, -1)
