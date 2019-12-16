using Pkg; Pkg.activate("/home/arzwa/dev/Beluga/")
using DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, Parameters, LinearAlgebra
using HypothesisTests

nw = open("test/data/dicots/plants2.nw", "r") do f ; readline(f); end
d, p = DLWGD(nw, 1., 1., 0.9)

prior1 = IidRevJumpPrior(
    Σ₀=[1 0.5 ; 0.5 1],
    X₀=MvNormal(log.([1,1]), I),
    πK=DiscreteUniform(0, 10),
    πq=Beta(1,1),
    πη=Beta(3,1),
    Tl=treelength(d))

prior2 = IidRevJumpPrior(
    Σ₀=[1 0.0 ; 0.0 1],
    X₀=MvNormal(log.([1,1]), [0.5 0.45 ; 0.45 0.5]),
    πK=Geometric(0.3),
    πq=Beta(1,1),
    πη=Beta(3,1),
    Tl=treelength(d))

prior3 = IidRevJumpPrior(
    Σ₀=[1 0.5 ; 0.5 1],
    X₀=MvNormal(log.([1,1]), I),
    πK=DiscreteUniform(0, 10),
    πq=Beta(1,3),
    πη=Beta(3,1),
    Tl=treelength(d))

prior4 = IidRevJumpPrior(
    Σ₀=[1 0.0 ; 0.0 1],
    X₀=MvNormal(log.([1,1]), [0.5 0.45 ; 0.45 0.5]),
    πK=Geometric(0.3),
    πq=Beta(1,1),
    πη=Beta(3,1),
    Tl=treelength(d))

priors = [prior1, prior2, prior3, prior4]

function main(priors)
    for prior in priors
        nw = open("test/data/dicots/plants2.nw", "r") do f ; readline(f); end
        d, p = DLWGD(nw, 1., 1., 0.9)
        chain = RevJumpChain(data=p, model=d, prior=prior)
        init!(chain, qkernel=Beta(1,1), λkernel=Exponential())
        rjmcmc!(chain, 22000, trace=2, show=100)
    end
end
