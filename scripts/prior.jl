using Pkg; Pkg.activate("/home/arzwa/dev/Beluga/")
using DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, Parameters


# branch model
begin
    nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
    d, p = DLWGD(nw, 1., 1., 0.9)
    prior = IidRevJumpPrior(
        Σ₀=[5 4.5 ; 4.5 5],
        X₀=MvNormal(log.([2,2]), I),
        πK=Beluga.UpperBoundedGeometric(0.2, 15),
        πq=Beta(1,1),
        πη=Beta(3,1))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain, rjump=(1., 20., 0.01))
end

rjmcmc!(chain, 21000, trace=1, show=100)
