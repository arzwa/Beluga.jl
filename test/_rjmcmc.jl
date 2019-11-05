using Test, DataFrames, CSV, Distributions, Random, DistributedArrays
using AdaptiveMCMC
using Beluga
using Parameters, LinearAlgebra, StatsBase
import Distributions: logpdf
import Beluga: iswgdafter, iswgd, nonwgdparent, parentdist, isroot, set!, rev!
import Beluga: ModelNode, update!, logpdfroot, branchrates, isawgd, iswgd
import Beluga: extend!, shrink!, ne, reindex!
import AdaptiveMCMC: ProposalKernel
include("../src/_rjmcmc.jl")

df = CSV.read("test/data/N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
df = CSV.read("test/data/plants1-100.tsv", delim=",")
df = df[1:100,:]
nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
ro(x, d=3) = round(x, digits=d)


# data
begin
    d, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9)
    p = Profile(y)
    prior = RevJumpPrior(Σ₀=[500 0 ; 0 500], X₀=MvNormal(log.([2,2]), I))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain)
    trace = State[]
end


# with WGD, no reversible jump
begin
    d, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9)
    p = Profile(y)
    prior = RevJumpPrior(Σ₀=[500 0. ; 0. 500], X₀=MvNormal(log.([2,2]), I), πq=Beta(2,2))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    wgdnode = insertwgd!(chain.model, chain.model[12], 0.03, 0.5)
    extend!(chain.data, 12)
    init!(chain)
    trace = State[]
end


# sample from the prior
# NOTE: for the reversible jump chain under the geometric prior, a removal step
# will always be accepted when it is not acccompanied by a change in parameters
# it is therefore best not to do such a step every iteration?
begin
    d, y = DuplicationLossWGDModel(nw, df[1:2,:], exp(randn()), exp(randn()), 0.9)
    p = PArray()
    prior = RevJumpPrior(Σ₀=[500 0. ; 0. 500],
                         X₀=MvNormal(log.([2,2]), I),
                         πK=Geometric(0.1))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain)
    trace = State[]
end


# sample from prior with WGD, no reversible jump
begin
    d, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9)
    p = PArray()
    # p = Profile(y)
    prior = RevJumpPrior(Σ₀=[500 0. ; 0. 500], X₀=MvNormal(log.([2,2]), I), πq=Beta(2,2))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    wgdnode = insertwgd!(chain.model, chain.model[12], 0.03, 0.5)
    # extend!(chain.data, 12)
    init!(chain)
    trace = State[]
end

for i=1:11000
    chain.state[:gen] += 1
    # rand() < 0.5 ? move_rmwgd!(chain) : move_addwgd!(chain)
    move!(chain)
    trace!(chain)
    if i%1 == 0
        println("↘ ", join(ro.(Vector(chain.trace[end,1:9])), ", "), " ⋯")
    end
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
