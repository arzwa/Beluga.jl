using Distributed
# addprocs(2)
@everywhere using Pkg
@everywhere Pkg.activate("/home/arzwa/dev/Beluga/")
@everywhere using Beluga, PhyloTree
using Test, DataFrames, CSV, Distributions, LinearAlgebra
using Plots, StatsPlots

df = CSV.read("test/data/N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
df = df[1:50,:]
nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end


# branch model
begin
    df = CSV.read("test/data/plants1-100.tsv", delim=",")
    # df = CSV.read("test/data/N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
    # df = CSV.read("../../rjumpwgd/data/sims/model1_8wgd_N=1000.csv", delim=",")
    df = df[1:20,:]
    nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
    d, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9, Beluga.BelugaBranch)
    p = Profile(y)
    # p = PArray()
    prior = IidRevJumpPrior(
        Σ₀=[1 0.9 ; 0.9 1],
        X₀=MvNormal(log.([2,2]), I),
        πK=Beluga.UpperBoundedGeometric(0.2, 15),
        πq=Beta(1,1),
        πη=0.9)
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain, rjump=(1., 10., 0.01))
end

rjmcmc!(chain, 5500, show=10, trace=1)


function mcmcmove!(chain)
    chain.state[:gen] += 1
    rand() < 0.5 ? move_rmwgd!(chain) : move_addwgd!(chain)
    move!(chain)
    trace!(chain)
end


for i=1:11000
    mcmcmove!(chain)
    if i%10 == 0
        println("↘ ", join(ro.(Vector(chain.trace[end,1:9])), ", "), " ⋯")
    end
    @assert length(chain.model) == length(postwalk(chain.model[1]))
    # if i%15 == 0
    #     display(postwalk(chain.model[1]))
    # end
end

l  = [x[id(chain.model[i], :λ)] for x in trace[1001:end], i=1:19]
m  = [x[id(chain.model[i], :μ)] for x in trace[1001:end], i=1:19]
k  = [x[:k] for x in trace[1001:end]]
e  = [x[:η1] for x in trace[1001:end]]
A = [k e l m]
CSV.write("/home/arzwa/bayesian_models_gfe/data/rjmcmc/dlwgd_coevol_prior.csv", chain.trace)

p = plot(); for i=1:size(l)[2]; density!(log.(l[:,i])); end; p
p = plot(); for i=1:size(m)[2]; density!(log.(m[:,i])); end; p

p = Profile()
logpdf!(p.Lp, p.xp, chain.model)

for (i,n) in chain.model.nodes
    x = [x[id(n, :λ)] for x in trace[100:end]] ; @show i, mean(x)
    x = [x[id(n, :μ)] for x in trace[100:end]] ; @show i, mean(x)
end

for (i, n) in chain.model.nodes
    @show i, n.x.θ
end

move_addwgd!(chain)
move_rmwgd!(chain)
