# Simulations to investigate the impact of across family variation. Ideally
# both acros family variation as the only source of variation, and a combination
# of lineage × family variation
using Pkg; Pkg.activate("/home/arzwa/julia-dev/Beluga/")
using Test, DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, PhyloTree, Parameters, DrWatson


function randmodel(nw, rates::Matrix, η, dist::Distribution, nt=BelugaBranch)
    r = rand(dist)
    model = DuplicationLossWGDModel(nw, 1., 1., η, nt)
    Beluga.setrates!(model, r.*rates)
    model
end

# Family × lineage variation
function simulate(nw, rateprior, pr, N, clade1)
    α = rand(pr)
    pα = Gamma(α,1/α)
    @assert mean(pα) ≈ 1

    η = rand(rateprior.πη)
    Σ = rand(InverseWishart(3, rateprior.Σ₀))
    X = rand(rateprior.X₀)
    rates = exp.(rand(MvNormal(X, Σ), Beluga.ne(m)+1))

    df = vcat([rand(randmodel(nw, rates, η, pα)) for i=1:2N]...)
    clade2 = setdiff(names(df), clade1)
    df = df[sum.(eachrow(df[clade1])) .!= 0, :]
    df = df[sum.(eachrow(df[clade2])) .!= 0, :]
    df[1:N,:], (α=α, Σ=Σ, rates=rates, η=η)
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
    @unpack α, Σ, rates, η = x
    trace = chain.trace[burnin:end,:]
    λs = Dict(Symbol("λ$i")=>rates[1,i] for i in 1:size(rates)[2])
    μs = Dict(Symbol("μ$i")=>rates[1,i] for i in 1:size(rates)[2])
    ss = Dict(:var=>Σ[1,1], :cov=>Σ[1,2], :η1=>η)
    d = merge(λs, μs, ss)
    df = DataFrame(:variable=>collect(keys(sort(d))),
        :trueval=>collect(values(sort(d))))
    df = join(df, describe(trace,
        :mean=>mean, :gmean=>(x)->exp(mean(log.(x))),
        :std =>std,  :gstd =>(x)->exp(std(log.(x))),
        :q025=>(x)->quantile(x, .025),
        :q05 =>(x)->quantile(x, .05),
        :q50 =>(x)->quantile(x, .50),
        :q95 =>(x)->quantile(x, .95),
        :q975=>(x)->quantile(x, .975)), on=:variable)
    push!(df, [:α ; α .+ zeros(length(names(df))-1)])
    df[:gmean][isnothing.(df[:gmean])] .= NaN
    df[:gstd][ isnothing.(df[:gstd])]  .= NaN
    df
end



# nw = open("data/plants2.nw", "r") do f ; readline(f); end
nw = open("test/data/plants2.nw", "r") do f ; readline(f); end

clade1 = [:bvu, :sly, :ugi, :cqu]

# base model and rates Distributions
m = DuplicationLossWGDModel(nw, 2., 2., 0.9, BelugaBranch)
params = (r=1, α=10, σ=0.5, cv=0.95, σ0=0.5, qa=1, qb=1, ηa=5, ηb=1, pk=0.2, N=50, n=5500, burnin=500)
@unpack r, α, σ, cv, σ0, ηa, ηb, qa, qb, pk, N, n, burnin = params
pr = Gamma(α,1/α)

simprior = IidRevJumpPrior(
    Σ₀=[σ cv*σ ; cv*σ σ],
    X₀=MvNormal(log.([r,r]), [σ0 cv*σ0 ; cv*σ0 σ0]),
    πq=Beta(qa,qb),
    πη=Beta(ηa,ηb),
    πK=Geometric(pk))

infprior = IidRevJumpPrior(
    Σ₀=[1 0.5 ; 0.5 1],
    X₀=MvNormal(log.([r,r]), I),
    πq=Beta(1,1),
    πη=Beta(3,1))

d, x = simulate(nw, simprior, pr, N, clade1)
c = inference(nw, d, infprior, n)
out = output(c, x, burnin)
# CSV.write("$(savename(params)).$(ARGS[1]).csv", out)
