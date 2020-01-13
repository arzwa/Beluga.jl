
Random.seed!(143)

@testset "rjMCMC prior sampling, MWG" begin
    nw = readline("data/plants1c.nw")
    d, p = DLWGD(nw, 1., 1., 0.9)
    prior = IRRevJumpPrior(
        Ψ=[1 0.0 ; 0.0 1],
        X₀=MvNormal(log.([3,3]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,1),
        πη=Beta(3,1),
        Tl=Beluga.treelength(d))
    kernel = Beluga.BranchKernel(qkernel=Beta(1,3))
    chain = RevJumpChain(data=p, model=d, prior=prior, kernel=kernel)
    init!(chain)
    @time rjmcmc!(chain, 2000, trace=5, show=100)

    for f in [mean, var]
        @test abs(f(prior.πK) - f(chain.trace[!,:k])) < 1.
        @test abs(f(prior.X₀)[1] - f(log.(chain.trace[!,:λ1]))) < 0.1
        @test abs(f(prior.X₀)[2] - f(log.(chain.trace[!,:μ1]))) < 0.1
        @test abs(f(prior.πη)[1] - f(chain.trace[!,:η1])) < 0.1
    end
end

Random.seed!(143)

@testset "rjMCMC prior sampling AMM" begin
    nw = readline("test/data/plants1c.nw")
    d, p = DLWGD(nw, 2., 2., 0.9)
    prior = IRRevJumpPrior(
        Ψ=[1 0.0 ; 0.0 1],
        X₀=MvNormal(log.([3,3]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,1),
        πη=Beta(3,1),
        Tl=Beluga.treelength(d))
    kernel = Beluga.BranchKernel(qkernel=Beta(1,3))
    props = Beluga.AMMProposals(2*Beluga.ne(d)+2, σ=5, β=0.1)
    chain = RevJumpChain(data=p, model=d, prior=prior, kernel=kernel, props=props)
    init!(chain)
    @time rjmcmc!(chain, 100000, trace=5, show=100)

    for f in [mean, var]
        @test abs(f(prior.πK) - f(chain.trace[!,:k])) < 1.5
        @test abs(f(prior.X₀)[1] - f(log.(chain.trace[!,:λ1]))) < 0.2
        @test abs(f(prior.X₀)[2] - f(log.(chain.trace[!,:μ1]))) < 0.2
        @test abs(f(prior.πη)[1] - f(chain.trace[!,:η1])) < 0.1
    end
end

begin
    nw = readline("example/9dicots/dicots.nw")
    df = CSV.read("example/9dicots/dicots-f01-25.csv")
    d, p = DLWGD(nw, df, 2., 2., 0.9)
    # d, p = DLWGD(nw, 2., 2., 0.9)
    prior = IRRevJumpPrior(
        Ψ=[1 0.0 ; 0.0 1],
        X₀=MvNormal(log.([1,1]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,3),
        πη=Beta(3,1),
        Tl=Beluga.treelength(d))
    kernel = Beluga.BranchKernel(qkernel=Beta(1,3))
    props = Beluga.AMMProposals(2*Beluga.ne(d)+2, σ=0.1, β=0.05)
    chain = RevJumpChain(data=p, model=d, prior=prior,
        kernel=kernel, props=props, da=false)
    init!(chain)
    for i=1:100
        @time rjmcmc!(chain, 100, trace=1, show=10)
        p = plot([plot(chain.trace[!,Symbol("λ$i")]) for i=1:4:16]...,
                 legend=false, grid=false)
        display(p)
    end
end

begin
    nw = readline("example/9dicots/dicots.nw")
    df = CSV.read("example/9dicots/dicots-f01-1000.csv")
    d, p = DLWGD(nw, df, 2., 2., 0.9)
    prior = IRRevJumpPrior(
        Ψ=[1 0.0 ; 0.0 1],
        X₀=MvNormal(log.([1,1]), [0.2 0.; 0. 0.2]),
        πK=DiscreteUniform(0, 10),
        πq=Beta(1,1),
        πη=Beta(3,1),
        Tl=Beluga.treelength(d))
    kernel = Beluga.BranchKernel(qkernel=Beta(1,3))
    chain2 = RevJumpChain(data=p, model=d, prior=prior, kernel=kernel, da=true)
    init!(chain2)
    for i=1:100
        @time rjmcmc!(chain2, 100, trace=1, show=10)
        p = plot([plot(chain2.trace[!,Symbol("λ$i")]) for i=1:4:16]...,
                 legend=false, grid=false)
        display(p)
    end
end

#     DA: 143.103991 seconds (501.90 M allocations: 107.977 GiB, 10.66% gc time)
#  no DA: 189.500442 seconds (796.78 M allocations: 169.154 GiB, 10.63% gc time)
# DA+AMM: 103.014576 seconds (350.40 M allocations: 72.692 GiB, 15.11% gc time)
# BUT: pjump(DA) << pjump(no-DA) and average prior DA > no-DA
