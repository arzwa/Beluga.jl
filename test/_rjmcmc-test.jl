@everywhere using Pkg
@everywhere Pkg.activate("/home/arzwa/julia-dev/Beluga/")
@everywhere using Beluga
using Test, DataFrames, CSV, Distributions, LinearAlgebra

ro(x, d=3) = round(x, digits=d)

function mcmc!(chain, n)
    for i=1:n
        chain.state[:gen] += 1
        rand() < 0.5 ? move_rmwgd!(chain) : move_addwgd!(chain)
        move!(chain)
        trace!(chain)
        if i%10 == 0
            println("↘ ", join(ro.(vec(chain)[1:14]), ", "), " ⋯")
        end
    end
end


begin
    nw = "((vvi:0.11000000000,((egr:0.10521364740,(ath:0.06820000000,cpa:0.06820000000):0.03701364740):0.00078635260,(mtr:0.1000000000,ptr:0.1000000000):0.006000000):0.00400000000):0.00702839392,((bvu:0.04330000000,cqu:0.04330000000):0.06820651982,(ugi:0.08350000000,sly:0.08350000000):0.02800651982):0.00552187411);"
    df = CSV.read("test/data/N=1000_tree=plants1c.nw_η=0.9_λ=2_μ=2.csv", delim=",")
    d, y = DuplicationLossWGDModel(nw, df, exp(randn()), exp(randn()), 0.9)
    p = Profile(y)
    prior = RevJumpPrior(Σ₀=[100 0 ; 0 100], X₀=MvNormal(log.([2,2]), I))
    chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
    init!(chain)
end

mcmc!(chain, 11000)
branchrates!(chain)
CSV.write("rjmcmc-test.trace.csv", chain.trace)
