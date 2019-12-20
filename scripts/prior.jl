using Pkg; Pkg.activate("/home/arzwa/dev/Beluga/")
using DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, Parameters, LinearAlgebra
using Plots, StatsPlots


# branch model
begin
    nw = open("test/data/dicots/plants2.nw", "r") do f ; readline(f); end
    d, p = DLWGD(nw, 1., 1., 0.9)
    prior = IidRevJumpPrior(
        Σ₀=[1 0.0 ; 0.0 1],
        X₀=MvNormal(log.([3,3]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,1),  # I have issues when not uniform (why?)
        πη=Beta(3,1),
        Tl=treelength(d),
        πE=Normal(1, 0.1))
    # kernel = Beluga.SimpleKernel(qkernel=Beta(1,1))
    # kernel = Beluga.DropKernel(qkernel=Beta(1,5))
    kernel = Beluga.BranchKernel(qkernel=Beta(1,3))
    chain = RevJumpChain(data=p, model=d, prior=prior, kernel=kernel)
    init!(chain)
end

for i=1:100
    rjmcmc!(chain, 1000, trace=5, show=100)
    kmax = maximum(chain.trace[50:end,:k])
    p1 = bar(Beluga.freqmap(chain.trace[50:end,:k]), color=:white)
    plot!(0:kmax, pdf(prior.πK, 0:kmax), linewidth=2, color=:black)
    p2 = histogram(log.(chain.trace[50:end,:λ1]), normalize=true, color=:white)
    plot!(p2, Normal(prior.X₀.μ[1], sqrt(prior.X₀.Σ.mat[1,1])), linewidth=2, color=:black)
    p3 = histogram(log.(chain.trace[50:end,:μ1]), normalize=true, color=:white)
    plot!(p3, Normal(prior.X₀.μ[2], sqrt(prior.X₀.Σ.mat[2,2])), linewidth=2, color=:black)
    p4 = histogram(chain.trace[50:end,:η1], normalize=true, color=:white)
    plot!(p4, prior.πη, linewidth=2, color=:black)
    display(plot(p1,p2,p3,p4, legend=false, grid=false))
end
# scatter(Beluga.freqmap(chain.trace[!,:k]))
