using Pkg; Pkg.activate("/home/arzwa/julia-dev/Beluga/")
using Test, DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, PhyloTree

df = CSV.read("test/data/N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
df = df[1:50,:]
nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end

# data
begin
    d, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9)
    p = Profile(y)
    prior = RevJumpPrior(Σ₀=[500 0 ; 0 500], X₀=MvNormal(log.([2,2]), I))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain)
end


# sample from the prior
# NOTE: for the reversible jump chain under the geometric prior, a removal step
# will always be accepted when it is not acccompanied by a change in parameters
# it is therefore best not to do such a step every iteration?
begin
    df = CSV.read("test/data/plants1-100.tsv", delim=",")
    d, y = DuplicationLossWGDModel(nw, df[1:50,:], exp(randn()), exp(randn()), 0.9)
    p = Profile(y)
    prior = CoevolRevJumpPrior(
        Σ₀=[100 0. ; 0. 100],
        X₀=MvNormal(log.([2,2]), I),
        πK=Poisson(5),
        πη=Beta(3,1))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain)
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
end

# sample with WGD
begin
    df = CSV.read("test/data/plants1-100.tsv", delim=",")
    # df = CSV.read("test/data/N=250_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
    df = df[1:100,:]
    nw = open("test/data/plants1c.nw", "r") do f ; readline(f); end
    d, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9)
    p = Profile(y)
    p = PArray()
    prior = CoevolRevJumpPrior(
        Σ₀=[100 0. ; 0. 100],
        X₀=MvNormal(log.([2,2]), I),
        πK=Geometric(0.1),
        πq=Beta(1,1))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    # wgdnode = insertwgd!(chain.model, chain.model[12], 0.03, 0.5)
    # extend!(chain.data, 12)
    # wgdnode = insertwgd!(chain.model, chain.model[16], 0.03, 0.5)
    # extend!(chain.data, 16)
    init!(chain)
end

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
        Σ₀=[5 4.5 ; 4.5 5],
        X₀=MvNormal(log.([2,2]), I),
        πK=Geometric(0.1),
        πq=Beta(1,1),
        πη=0.9)
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain)
end




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
