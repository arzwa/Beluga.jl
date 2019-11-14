using Pkg; Pkg.activate("/home/arzwa/julia-dev/Beluga/")
using Test, DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, PhyloTree, Parameters, DrWatson


function randmodel(m, θ_dist, q_dist, k_dist)
    model = deepcopy(m)
    rates = exp.(rand(θ_dist, length(model)))
    Beluga.setrates!(model, rates)
    k = rand(k_dist)
    wgds = Dict()
    for i=1:k
        n, t = Beluga.randpos(model)
        q = rand(q_dist)
        wgdnode = insertwgd!(model, n, t, q)
        child = nonwgdchild(wgdnode)
        wgds[wgdnode.i] = (n.i, t, q)
    end
    (model=model, rates=rates, wgds=wgds)
end

function simulate(m, N, clade1)
    df = rand(m, 2N)  # HACK
    clade2 = setdiff(names(df), clade1)
    df = df[sum.(eachrow(df[clade1])) .!= 0, :]
    df = df[sum.(eachrow(df[clade2])) .!= 0, :]
    df[1:N,:]
end

function inference(m, nw, df, wgds, n=6000)
    model, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9)
    data = Profile(y)
    chain = RevJumpChain(data=data, model=model, prior=prior)
    for (i, wgd) in sort(wgds)
        wgdnode = insertwgd!(chain.model, chain.model[wgd[1]], wgd[2], rand())
        extend!(chain.data, wgd[1])
    end
    init!(chain)
    mcmc!(chain, n, trace=1, show=1)
    chain
end

function output(chain, x, burnin=1000)
    @unpack rates, wgds = x
    trace = chain.trace[burnin:end,:]
    qs = Dict(Symbol("q$i")=>wgd[end] for (i,wgd) in sort(wgds))
    λs = Dict(Symbol("λ$i")=>rates[1,i] for i in 1:size(rates)[2])
    μs = Dict(Symbol("μ$i")=>rates[1,i] for i in 1:size(rates)[2])
    d = merge(qs, λs, μs)
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
m, y = DuplicationLossWGDModel(nw, df[1:2,:], 2., 2., 0.9)
params = (r=2, σ=0.2, qa=1, qb=1, pk=0.2, N=100, n=1000, burnin=200)

@unpack r, σ, qa, qb, pk, N, n, burnin = params
θ_dist = MvNormal(log.([r, r]), σ*I)
q_dist = Beta(qa, qb)
k_dist = Geometric(pk)
prior  = IidRevJumpPrior(
    Σ₀=[5 0 ; 0 5],
    X₀=MvNormal(log.([r,r]), I),
    πq=Beta(1,1),
    πη=Beta(3,1))

x = randmodel(m, θ_dist, q_dist, k_dist)
d = simulate(x.model, N, clade1)
c = inference(x.model, nw, d, x.wgds, n)
out = output(c, x, burnin)
CSV.write("$(savename(params)).$(ARGS[1]).csv", out)
