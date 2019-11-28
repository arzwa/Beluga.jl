# using Distributed
# addprocs(2)
# @everywhere using Pkg
# @everywhere Pkg.activate("/home/arzwa/dev/Beluga/")
# @everywhere using Beluga, PhyloTree

using Pkg
Pkg.activate("/home/arzwa/dev/Beluga/")
using Beluga, PhyloTree
using Test, DataFrames, CSV, Distributions, LinearAlgebra
# using Plots, StatsPlots


# branch model
begin
    df = CSV.read("test/data/plants1-100.tsv", delim=",")
    # df = CSV.read("test/data/N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
    # df = CSV.read("../../rjumpwgd/data/sims/model1_8wgd_N=1000.csv", delim=",")
    df = df[1:25,:]
    nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
    d, y = DuplicationLossWGDModel(nw, df, 1., 1., 0.9, Beluga.BelugaBranch)
    p = Profile(y)
    # p = PArray()
    prior = IidRevJumpPrior(
        Σ₀=[0.1 0.099 ; 0.099 0.1],
        X₀=Normal(),
        πK=Beluga.UpperBoundedGeometric(0.3, 15),
        πq=Beta(1,1),
        πη=0.9)
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain, rjump=(1., 10., 0.001))

    # \delta = 0.001: 2574/5410
    # \delta = 0.01 : 2660/5483
    # \delta = 0.1  : 1520/5441
end

rjmcmc!(chain, 5500, show=10, trace=1)
