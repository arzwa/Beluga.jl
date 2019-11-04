using Test, DataFrames, CSV, Distributions, Random
using AdaptiveMCMC
using Beluga
using Parameters, LinearAlgebra, StatsBase
import Distributions: logpdf
import Beluga: iswgdafter, iswgd, nonwgdparent, parentdist, isroot, set!, rev!
import Beluga: ModelNode, update!, logpdfroot, branchrates, isawgd, iswgd
import Beluga: extend!, shrink!
import AdaptiveMCMC: ProposalKernel

df = CSV.read("test/data/N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
df = df[1:25,:]
nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
ro(x, d=3) = round(x, digits=d)

begin
    d, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9)
    p = Profile(y)
    prior = RevJumpPrior(Σ₀=[100 99 ; 99 100], X₀=MvNormal(log.([2,2]), I))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain)
    trace = State[]
end

for i=1:1000
    move_rmwgd!(chain)
    move_addwgd!(chain)
    move!(chain)
    chain.state[:gen] += 1
    # push!(trace, branchrates(chain.model))
    push!(trace,deepcopy(chain.state))
    if i%10 == 0
        @unpack state = chain
        x = [state[:gen], ro(state[:logp]), ro(state[:logπ]), state[:k]]
        println("⋅ ", join(x, ", "))
    end
end

logpdf!(p[1].Lp, p[1].xp, chain.model)

for (i,n) in chain.model.nodes
    x = [x[id(n, :λ)] for x in trace[100:end]] ; @show i, mean(x)
    x = [x[id(n, :μ)] for x in trace[100:end]] ; @show i, mean(x)
end

for (i, n) in chain.model.nodes
    @show i, n.x.θ
end

move_addwgd!(chain)
move_rmwgd!(chain)
