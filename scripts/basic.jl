using Pkg; Pkg.activate("/home/arzwa/dev/Beluga/")
using DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, Parameters


# branch model
begin
    ddir = "test/data"
    nw = open(joinpath(ddir, "plants1.nw"), "r") do f ; readline(f); end
    df = CSV.read(joinpath(ddir, "plants1-10.tsv"), delim=",")
    d, p = DLWGD(nw, df, 1., 1., 0.9)
    prior = IidRevJumpPrior(
        Σ₀=[0.5 0 ; 0 0.5],
        X₀=MvNormal(log.(ones(2)), I),
        πK=Beluga.UpperBoundedGeometric(0.1, 20),
        πq=Beta(1,1),
        πη=Beta(3,1))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain, rjump=(1., 20., 1e-10))
end

rjmcmc!(chain, 5000, trace=1, show=100)
