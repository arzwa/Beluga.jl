using Pkg; Pkg.activate("/home/arzwa/julia-dev/Beluga/")
using Test, DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, PhyloTree, Parameters, DrWatson


function randmodel(m, prior)
    model = deepcopy(m)
    k = rand(prior.πK)
    wgds = Dict()
    for i=1:k
        n, t = Beluga.randpos(model)
        wgdnode = insertwgd!(model, n, t, 0.5)
        child = nonwgdchild(wgdnode)
        wgds[wgdnode.i] = (n.i, t)
    end
    @unpack model, Σ = rand(prior, model)
    (model=model, Σ=Σ, rates=Beluga.branchrates(model), wgds=wgds)
end

function simulate(m, N, clade1)
    df = rand(m, 2N)  # HACK
    clade2 = setdiff(names(df), clade1)
    df = df[sum.(eachrow(df[clade1])) .!= 0, :]
    df = df[sum.(eachrow(df[clade2])) .!= 0, :]
    df[1:N,:]
end

function inference(m, nw, df, wgds, prior, n=6000)
    model, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9)
    data = Profile(y)
    chain = RevJumpChain(data=data, model=deepcopy(model), prior=prior)
    for (i, wgd) in sort(wgds)
        wgdnode = insertwgd!(chain.model, chain.model[wgd[1]], wgd[2], rand())
        extend!(chain.data, wgd[1])
    end
    init!(chain)
    mcmc!(chain, n, trace=1, show=10)
    Beluga.posterior_Σ!(chain, model)
    chain
end

function output(chain, x, burnin=1000)
    @unpack model, rates, wgds, Σ = x
    trace = chain.trace[burnin:end,:]
    qs = Dict(Symbol("q$i")=>model[i][:q] for (i,wgd) in sort(wgds))
    λs = Dict(Symbol("λ$i")=>rates[1,i] for i in 1:size(rates)[2])
    μs = Dict(Symbol("μ$i")=>rates[1,i] for i in 1:size(rates)[2])
    ss = Dict(:var=>Σ[1,1], :cov=>Σ[1,2])
    d = merge(qs, λs, μs, ss)
    df = DataFrame(:variable=>collect(keys(sort(d))),
        :trueval=>collect(values(sort(d))))
    join(df, describe(trace,
        :mean=>mean, :std=>std,
        :q025=>(x)->quantile(x, .025),
        :q05 =>(x)->quantile(x, .05),
        :q95 =>(x)->quantile(x, .95),
        :q975=>(x)->quantile(x, .975)), on=:variable)
end


# script
df = CSV.read("test/data/N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
clade1 = [:bvu, :sly, :ugi, :cqu]

# base model and rates Distributions
m, y = DuplicationLossWGDModel(nw, df[1:2,:], 2., 2., 0.9, BelugaBranch)
params = (r=2, σ=2, σ0=1, qa=1, qb=1, ηa=5, ηb=1, pk=0.2, N=10, n=100, burnin=20)

@unpack r, σ, σ0, ηa, ηb, qa, qb, pk, N, n, burnin = params

simprior = CoevolRevJumpPrior(
    Σ₀=[σ 0.75σ ; 0.75σ σ],
    X₀=MvNormal(log.([r,r]), [σ0 0.95σ0 ; 0.95σ0 σ0]),
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
c = inference(x.model, nw, d, x.wgds, infprior, n)
out = output(c, x, burnin)
# CSV.write("$(savename(params)).$(ARGS[1]).csv", out)
