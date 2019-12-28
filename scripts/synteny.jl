using Pkg; Pkg.activate("/home/arzwa/dev/Beluga/")
using DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, Parameters

base = "/home/arzwa/research/synteny/data/tao2018"
nw = readline(joinpath(base, "monocots.nw"))
df = CSV.read(joinpath(base, "monocots-f0-100.csv"), delim=",")

begin
    d, p = DLWGD(nw, df, exp(-8.5), 1., 0.9)
    Beluga.addrandwgds!(d, p, 10, Beta())
    prior = Beluga.SyntenyRevJumpPrior(
        lλ=Uniform(-9,-8),
        Tl=treelength(d))
    kernel = Beluga.SimpleKernel(qkernel=Beta(1,3))
    chain = RevJumpChain(data=p, prior=prior, kernel=kernel, model=d)
    init!(chain)
end

rjmcmc!(chain, 1000, trace=1, show=10)
