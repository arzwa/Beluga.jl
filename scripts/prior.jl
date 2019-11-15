using Pkg; Pkg.activate("/home/arzwa/julia-dev/Beluga/")
using DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, PhyloTree, Parameters


# branch model
begin
    nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
    d = DuplicationLossWGDModel(nw, 1., 1., 0.9, Beluga.BelugaBranch)
    p = PArray()
    prior = IidRevJumpPrior(
        Σ₀=[5 4.5 ; 4.5 5],
        X₀=MvNormal(log.([2,2]), I),
        πK=Geometric(0.2),
        πq=Beta(1,1),
        πη=Beta(3,1))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain)
end

rjmcmc!(chain, 11000, trace=1, show=100)
