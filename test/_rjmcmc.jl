# using Distributed
# addprocs(2)
# @everywhere using Pkg
# @everywhere Pkg.activate("/home/arzwa/dev/Beluga/")
# @everywhere using Beluga
# using CSV, DataFrames, Distributions, LinearAlgebra

# using Pkg
Pkg.activate("/home/arzwa/dev/Beluga/")
using Beluga
using Test, DataFrames, CSV, Distributions, LinearAlgebra


# branch model
begin
    df = CSV.read("test/data/plants1-100.tsv", delim=",")
    # df = CSV.read("test/data/N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
    # df = CSV.read("../../rjumpwgd/data/sims/model1_8wgd_N=1000.csv", delim=",")
    nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
    d, p = DLWGD(nw, df, 1., 1., 0.9, Beluga.Branch)
    # d, p = DLWGD(nw, 1., 1., 0.9, Beluga.Branch)
    prior = IidRevJumpPrior(
        Σ₀=[0.1 0.099 ; 0.099 0.1],
        X₀=MvNormal([0., 0.], I),
        πK=Beluga.UpperBoundedGeometric(0.3, 15),
        πq=Beta(1,1),
        πη=Beta(3,1))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain, rjump=(1., 10., 0.001))

    # \delta = 0.001: 2574/5410
    # \delta = 0.01 : 2660/5483
    # \delta = 0.1  : 1520/5441
end

rjmcmc!(chain, 5500, show=10, trace=1)


#= 2 (3?) workers
julia> for i=1:10; @time logpdf!(d, p); end
  0.010894 seconds (912 allocations: 266.953 KiB)
  0.011553 seconds (913 allocations: 267.438 KiB)
  0.016060 seconds (912 allocations: 266.953 KiB)
  0.022446 seconds (915 allocations: 267.063 KiB)
  0.011193 seconds (911 allocations: 266.938 KiB)
  0.021619 seconds (915 allocations: 267.469 KiB)
  0.012829 seconds (912 allocations: 266.953 KiB)
  0.016304 seconds (911 allocations: 266.938 KiB)
  0.020149 seconds (912 allocations: 266.953 KiB)
  0.012285 seconds (915 allocations: 267.469 KiB)

julia> for i=1:10; @time logpdf!(d, p); end
    0.016945 seconds (57.77 k allocations: 12.998 MiB)
    0.017028 seconds (57.77 k allocations: 12.998 MiB)
    0.039958 seconds (57.77 k allocations: 12.998 MiB, 36.61% gc time)
    0.015804 seconds (57.77 k allocations: 12.998 MiB)
    0.018057 seconds (57.77 k allocations: 12.998 MiB)
    0.022800 seconds (57.77 k allocations: 12.998 MiB)
    0.042096 seconds (57.77 k allocations: 12.998 MiB, 35.99% gc time)
    0.019602 seconds (57.77 k allocations: 12.998 MiB)
    0.024842 seconds (57.77 k allocations: 12.998 MiB)
    0.021387 seconds (57.77 k allocations: 12.998 MiB)
 =#
