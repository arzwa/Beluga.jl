
Random.seed!(143)

@testset "rjMCMC prior" begin
    nw = readline("data/plants1c.nw")
    d, p = DLWGD(nw, 1., 1., 0.9)
    prior = IRRevJumpPrior(
        Σ₀=[1 0.0 ; 0.0 1],
        X₀=MvNormal(log.([3,3]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,1),
        πη=Beta(3,1),
        Tl=Beluga.treelength(d))
    kernel = Beluga.BranchKernel(qkernel=Beta(1,3))
    chain = RevJumpChain(data=p, model=d, prior=prior, kernel=kernel)
    init!(chain)
    rjmcmc!(chain, 2000, trace=5, show=100)

    for f in [mean, var]
        @test abs(f(prior.πK) - f(chain.trace[!,:k])) < 1.
        @test abs(f(prior.X₀)[1] - f(log.(chain.trace[!,:λ1]))) < 0.1
        @test abs(f(prior.X₀)[2] - f(log.(chain.trace[!,:μ1]))) < 0.1
        @test abs(f(prior.πη)[1] - f(chain.trace[!,:η1])) < 0.1
    end
end
