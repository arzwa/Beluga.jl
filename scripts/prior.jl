using Pkg; Pkg.activate("/home/arzwa/dev/Beluga/")
using DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, Parameters, LinearAlgebra
using Plots


# branch model
begin
    nw = open("test/data/dicots/plants2.nw", "r") do f ; readline(f); end
    d, p = DLWGD(nw, 1., 1., 0.9)
    prior = IidRevJumpPrior(
        Σ₀=[1 0.5 ; 0.5 1],
        X₀=MvNormal(log.([1,1]), I),
        # πK=Beluga.UpperBoundedGeometric(0.2, 10),
        # πK=DiscreteUniform(0,10),
        πK=DiscreteUniform(0, 15),
        πq=Beta(1,1),  # I have issues when not uniform (why?)
        πη=Beta(3,1),
        Tl=treelength(d))
        #πE=LogNormal(log(1), 0.1))
    chain = RevJumpChain(data=p, model=d, prior=prior)
    init!(chain, qkernel=Beta(1,3), λkernel=Exponential())
end

for i=1:100
    rjmcmc!(chain, 1000, trace=2, show=100)
    p = bar(Beluga.freqmap(chain.trace[!,:k]))
    plot!(0:20, pdf(prior.πK, 0:20), linewidth=2)
    display(p)
end
# scatter(Beluga.freqmap(chain.trace[!,:k]))
