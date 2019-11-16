# Simulations to investigate the impact of across family variation. Ideally
# both acros family variation as the only source of variation, and a combination
# of lineage × family variation
using Pkg; Pkg.activate("/home/arzwa/julia-dev/Beluga/")
using Test, DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, PhyloTree, Parameters, DrWatson


function randmodel(nw, λ, μ, η, dist::Distribution, nt=BelugaBranch)
    r = rand(dist)
    DuplicationLossWGDModel(nw, λ*r, μ*r, η, nt)
end

function simulate(nw, simprior, N, clade1)
    @unpack pθ, pη, pr = simprior
    λ, μ = exp.(rand(pθ))
    η = rand(pη)
    α = rand(pr)
    pα = Gamma(α,1/α)
    @assert mean(pα) == 1
    df = vcat([rand(randmodel(nw, λ, μ, η, pα)) for i=1:2N]...)
    clade2 = setdiff(names(df), clade1)
    df = df[sum.(eachrow(df[clade1])) .!= 0, :]
    df = df[sum.(eachrow(df[clade2])) .!= 0, :]
    df[1:N,:], (α=α, η=η, λ=λ, μ=μ)
end

function inference(nw, df, prior, n=6000, nt=BelugaBranch)
    model, y = DuplicationLossWGDModel(nw, df, 1., 1., 0.9, nt)
    data = Profile(y)
    chain = RevJumpChain(data=data, model=deepcopy(model), prior=prior)
    init!(chain)
    mcmc!(chain, n, trace=1, show=10)
    Beluga.posterior_Σ!(chain, model)
    chain
end

function output(chain, x, burnin=1000)
    @unpack α, λ, μ, η = x
    trace = chain.trace[burnin:end,:]

    df = describe(trace,
        :mean=>mean, :gmean=>(x)->exp(mean(log.(x))),
        :std =>std,  :gstd =>(x)->exp(std(log.(x))),
        :q025=>(x)->quantile(x, .025),
        :q05 =>(x)->quantile(x, .05),
        :q50 =>(x)->quantile(x, .50),
        :q95 =>(x)->quantile(x, .95),
        :q975=>(x)->quantile(x, .975))
    push!(df, [:α ; α .+ zeros(length(names(df))-1)])
    push!(df, [:λ ; λ .+ zeros(length(names(df))-1)])
    push!(df, [:μ ; μ .+ zeros(length(names(df))-1)])
    push!(df, [:η ; η .+ zeros(length(names(df))-1)])
end


@with_kw struct FamilyVarPrior
    pθ = MvNormal(ones(2), [1 0.95 ; 0.95 1])
    pη = Beta(5,1)
    pr = Gamma(50,1/50)
end

r = 1.
nw = open("test/data/plants2.nw", "r") do f ; readline(f); end
clade1 = [:bvu, :sly, :ugi, :cqu]
simprior = FamilyVarPrior()
infprior = IidRevJumpPrior(
    Σ₀=[5 0 ; 0 5],
    X₀=MvNormal(log.([r,r]), I),
    πq=Beta(1,1),
    πη=Beta(3,1))

df, params = simulate(nw, simprior, 100, clade1)
c = inference(nw, df, infprior)
out = output(c, params)
