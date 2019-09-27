using Random, Test, Distributions
Random.seed!(6789)

geomean(x) = exp(mean(log.(x)))

prior2 = Beluga.IIDRatesPrior(
    Exponential(0.1),
    MvLogNormal(log.([0.5, 0.5]), [.5 0.45 ; 0.45 .5]),
    Beta(1,1),
    Beta(8,2))

n = 2000
s = SpeciesTree("data/plants1.nw")
p = Profile()  # prior alone
chain = DLChain(p, prior2, s, 2)

@info "Running short chain ($n iters), sampling from prior"
chn = mcmc!(chain, n, show_trace=false)

@testset "Sampling from prior (iid rates)" begin
    trace = chain.trace
    for i=1:19
        x = trace[Symbol("λ$i")]
        @test isapprox(std(x), 0.5, atol=0.05)
        @test isapprox(geomean(x), 0.5, atol=0.05)
        x = trace[Symbol("μ$i")]
        @test isapprox(std(x), 0.5, atol=0.05)
        @test isapprox(geomean(x), 0.5, atol=0.05)
    end
    x = trace[:ν]
    @test isapprox(mean(x), mean(prior2.dν), atol=0.08) # XXX tol is quite high
    @test isapprox(std(x), std(prior2.dν), atol=0.08)   # XXX tol is quite high
    x = trace[:η]
    @test isapprox(mean(x), mean(prior2.dη), atol=0.01)
    @test isapprox(std(x), std(prior2.dη), atol=0.01)
end
