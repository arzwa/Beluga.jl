using Pkg; Pkg.activate("/home/arzwa/dev/Beluga/")
using DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, Parameters


# branch model
begin
    ddir = "test/data"
    nw = open(joinpath(ddir, "plants2.nw"), "r") do f ; readline(f); end
    df = CSV.read(joinpath(ddir, "dicots-f01-100.csv"), delim=",")
    d, p = DLWGD(nw, df, 1., 1., 0.9)
    prior = IidRevJumpPrior(
        Σ₀=[0.5 0 ; 0 0.5],
        X₀=MvNormal(log.(ones(2)), I),
        πK=Beluga.UpperBoundedGeometric(0.1, 20),
        πq=Beta(1,1),
        πη=Beta(3,1))
    chain = RevJumpChain(data=deepcopy(p), model=deepcopy(d), prior=deepcopy(prior))
end

init!(chain, qkernel=Beta(1,10), λdrop=(δ=0.01, ti=25, stop=0))
rjmcmc!(chain, 1000, trace=1, show=10)

#
# ks = []
# for delta in [1e-4, 1e-3, 1e-2, 1e-1], b in 5:5:40
#     @show delta, b
#     chain = RevJumpChain(data=deepcopy(p), model=deepcopy(d), prior=deepcopy(prior))
#     init!(chain, qkernel=Beta(1,b), λdrop=(δ=delta, ti=25, stop=0))
#     rjmcmc!(chain, 1000, trace=1, show=10)
#     @show chain.props[0][2]
#     push!(ks, deepcopy(chain.props))
# end
#
#
# (delta, b, round(ap[i], digits=2)) = (0.0001, 5, 0.28)
# (delta, b, round(ap[i], digits=2)) = (0.0001, 10, 0.41)
# (delta, b, round(ap[i], digits=2)) = (0.0001, 15, 0.46)
# (delta, b, round(ap[i], digits=2)) = (0.0001, 20, 0.54)
# (delta, b, round(ap[i], digits=2)) = (0.0001, 25, 0.59)
# (delta, b, round(ap[i], digits=2)) = (0.0001, 30, 0.63)
# (delta, b, round(ap[i], digits=2)) = (0.0001, 35, 0.66)
# (delta, b, round(ap[i], digits=2)) = (0.0001, 40, 0.66)
# (delta, b, round(ap[i], digits=2)) = (0.001, 5, 0.24)
# (delta, b, round(ap[i], digits=2)) = (0.001, 10, 0.43)
# (delta, b, round(ap[i], digits=2)) = (0.001, 15, 0.47)
# (delta, b, round(ap[i], digits=2)) = (0.001, 20, 0.52)
# (delta, b, round(ap[i], digits=2)) = (0.001, 25, 0.59)
# (delta, b, round(ap[i], digits=2)) = (0.001, 30, 0.64)
# (delta, b, round(ap[i], digits=2)) = (0.001, 35, 0.65)
# (delta, b, round(ap[i], digits=2)) = (0.001, 40, 0.67)
# (delta, b, round(ap[i], digits=2)) = (0.01, 5, 0.26)
# (delta, b, round(ap[i], digits=2)) = (0.01, 10, 0.38)
# (delta, b, round(ap[i], digits=2)) = (0.01, 15, 0.47)
# (delta, b, round(ap[i], digits=2)) = (0.01, 20, 0.54)

function f()
    i = 1
    for delta in [1e-4, 1e-3, 1e-2, 1e-1], b in 5:5:40
        @show delta, b, round(ap[i], digits=2)
        i += 1
    end
end
