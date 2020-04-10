
Random.seed!(143)

@testset "rjMCMC prior sampling, MWG" begin
    nw = readline(joinpath(@__DIR__, "data/plants1c.nw"))
    d, p = DLWGD(nw, 1., 1., 0.9)
    prior = IRRevJumpPrior(
        Ψ=[1 0.0 ; 0.0 1],
        X₀=MvNormal(log.([3,3]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,1),
        πη=Beta(3,1),
        Tl=treelength(d))
    kernel = BranchKernel(qkernel=Beta(1,3))
    chain = RevJumpChain(data=p, model=d, prior=prior, kernel=kernel)
    init!(chain)
    @time rjmcmc!(chain, 5000, trace=5, show=100)

    for f in [mean, var]
        @test abs(f(prior.πK) - f(chain.trace[!,:k])) < 1.
        @test abs(f(prior.X₀)[1] - f(log.(chain.trace[!,:λ1]))) < 0.1
        @test abs(f(prior.X₀)[2] - f(log.(chain.trace[!,:μ1]))) < 0.1
        @test abs(f(prior.πη)[1] - f(chain.trace[!,:η1])) < 0.1
    end
end

Random.seed!(143)

@testset "rjMCMC prior sampling AMM" begin
    nw = readline(joinpath(@__DIR__, "data/plants1c.nw"))
    d, p = DLWGD(nw, 2., 2., 0.9)
    prior = IRRevJumpPrior(
        Ψ=[1 0.0 ; 0.0 1],
        X₀=MvNormal(log.([3,3]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,1),
        πη=Beta(3,1),
        Tl=treelength(d))
    kernel = BranchKernel(qkernel=Beta(1,3))
    props = Beluga.AMMProposals(2*Beluga.ne(d)+2, σ=5, β=0.1)
    chain = RevJumpChain(data=p, model=d, prior=prior, kernel=kernel, props=props)
    init!(chain)
    @time rjmcmc!(chain, 10000, trace=5, show=100)

    for f in [mean, var]
        @test abs(f(prior.πK) - f(chain.trace[!,:k])) < 1.5
        @test abs(f(prior.X₀)[1] - f(log.(chain.trace[!,:λ1]))) < 0.4
        @test abs(f(prior.X₀)[2] - f(log.(chain.trace[!,:μ1]))) < 0.4
        @test abs(f(prior.πη)[1] - f(chain.trace[!,:η1])) < 0.1
    end
end
