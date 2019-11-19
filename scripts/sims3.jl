# These should be simulations that simulate from a particular prior, do
# inference with reversible jump, and evaluate both rate estimates and WGD
# inferences using the branch-bayes factors.
using Pkg; Pkg.activate("/home/arzwa/julia-dev/Beluga/")
using Test, DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, PhyloTree, Parameters, DrWatson

function randmodel(m, prior::IidRevJumpPrior)
    model = deepcopy(m)
    model[1][:η] = rand(prior.πη)
    k = rand(prior.πK)
    wgds = Dict()
    for i=1:k
        n, t = Beluga.randpos(model)
        q = rand(prior.πq)
        wgdnode = insertwgd!(model, n, t, q)
        child = nonwgdchild(wgdnode)
        wgds[wgdnode.i] = (child.i, q)
    end
    Σ = rand(InverseWishart(3, prior.Σ₀))
    X = rand(prior.X₀)
    rates = exp.(rand(MvNormal(X, Σ), Beluga.ne(m)+1))
    Beluga.setrates!(model, rates)
    (model=model, Σ=Σ, η=model[1][:η], rates=rates, wgds=wgds)
end

function simulate(m, N, clade1)
    df = rand(m, 2N)  # HACK
    clade2 = setdiff(names(df), clade1)
    df = df[sum.(eachrow(df[clade1])) .!= 0, :]
    df = df[sum.(eachrow(df[clade2])) .!= 0, :]
    df[1:N,:]
end

function inference(nw, df, wgds, prior, n=6000, nt=BelugaBranch)
    model, y = DuplicationLossWGDModel(nw, df, 1., 1., 0.9, nt)
    data = Profile(y)
    chain = RevJumpChain(data=data, model=deepcopy(model), prior=prior)
    init!(chain)
    rjmcmc!(chain, n, trace=1, show=10)
    Beluga.posterior_Σ!(chain, model)
    chain
end

function output(chain, x, burnin=1000)
    @unpack model, rates, wgds, Σ, η = x
    trace = chain.trace[burnin:end,:]
    qs = Dict(Symbol("q$i")=>model[i][:q] for (i,wgd) in sort(wgds))
    λs = Dict(Symbol("λ$i")=>rates[1,i] for i in 1:size(rates)[2])
    μs = Dict(Symbol("μ$i")=>rates[1,i] for i in 1:size(rates)[2])
    ss = Dict(:var=>Σ[1,1], :cov=>Σ[1,2], :η1=>η)
    d = merge(qs, λs, μs, ss)
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
    df[:gmean][isnothing.(df[:gmean])] .= NaN
    df[:gstd][ isnothing.(df[:gstd])]  .= NaN
    df
end



# script
nw = open("test/data/plants2.nw", "r") do f ; readline(f); end
clade1 = [:bvu, :sly, :ugi, :cqu]

# base model and rates Distributions
m = DuplicationLossWGDModel(nw, 2., 2., 0.9, BelugaBranch)

params = (r=1, σ=0.1, cv=0.95, σ0=1,
    qa=1, qb=1, ηa=5, ηb=1,
    pk=0.2, N=10, n=5500, burnin=500)

@unpack r, σ, cv, σ0, ηa, ηb, qa, qb, pk, N, n, burnin = params

simprior = IidRevJumpPrior(
    Σ₀=[σ cv*σ ; cv*σ σ],
    X₀=MvNormal(log.([r,r]), [σ0 cv*σ0 ; cv*σ0 σ0]),
    πq=Beta(qa,qb),
    πη=Beta(ηa,ηb),
    πK=Geometric(pk))

infprior = IidRevJumpPrior(
    Σ₀=[5 0 ; 0 5],
    X₀=MvNormal(log.([r,r]), I),
    πq=Beta(1,1),
    πη=Beta(3,1))

x = randmodel(m, simprior)
d = simulate(x.model, N, clade1)
c = inference(nw, d, x.wgds, infprior, n)
out = output(c, x, burnin)

bfs = branch_bayesfactors(chain)
CSV.write("$(savename(params)).$(ARGS[1]).csv", out)
CSV.write("$(savename(params)).$(ARGS[1]).bfs.csv", out)
