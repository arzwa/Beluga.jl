using Pkg; Pkg.activate("/home/arzwa/dev/Beluga/")
using DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, Parameters


# branch model
begin
    nw = open("test/data/plants2.nw", "r") do f ; readline(f); end
    d, p = DLWGD(nw, 1., 1., 0.9)
    prior = IidRevJumpPrior(
        Σ₀=[1 0.5 ; .5 1],
        X₀=MvNormal(log.([1,1]), I),
        πK=Beluga.UpperBoundedGeometric(0.1, 15),
        πq=Beta(1,1),
        πη=Beta(3,1),
        Tl=treelength(d))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain, qkernel=Beta(1,1))
end

rjmcmc!(chain, 21000, trace=1, show=100)
