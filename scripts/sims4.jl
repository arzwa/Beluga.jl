# These should be simulations that simulate from a particular prior, do
# inference with reversible jump, and evaluate both rate estimates and WGD
# inferences using the branch-bayes factors.
using Pkg; Pkg.activate("/home/arzwa/rjumpwgd")
using DataFrames, CSV, Distributions, LinearAlgebra
using Beluga, Parameters, DrWatson, JLD


# Configuration
outdir = "/tmp/sims4test"
simid  = "1"  # ARGS[1]
nw     = open("test/data/plants2.nw", "r") do f ; readline(f); end
clade1 = [:bvu, :sly, :ugi, :cqu]
params = (
    r=1.5, σ=0.1, cv=0.99, σ0=0.5,
    qa=2, qb=3, ηa=5, ηb=1, qaa=1, qbb=1,
    s=.5, ss=0.4, pk=0.1, N=10, n=210, burnin=10)
@unpack r, σ, cv, σ0, s, ss, ηa, ηb, qa, qaa, qb, qbb, pk, N, n, burnin = params


# priors
simprior = IidRevJumpPrior(
    Σ₀=[σ cv*σ ; cv*σ σ],
    X₀=MvNormal(log.([r,r]), [σ0 cv*σ0 ; cv*σ0 σ0]),
    πq=Beta(qa,qb),
    πη=Beta(ηa,ηb),
    πK=Binomial(10, 0.5))

infprior = IidRevJumpPrior(
    Σ₀=[s ss ; ss s],
    X₀=MvNormal(log.([r,r]), [s ss; ss s]),
    πq=Beta(qaa,qbb),
    πη=Beta(3,1),
    πK=UpperBoundedGeometric(pk, 10))


# methods
function simulate(model, N, clade1)
    clade2 = [v for (k,v) in  model.leaves if !(v in clade1)]
    rand(model, N, [clade1,clade2])
end

function inference(nw, df, wgds, prior, n=6000, nt=Branch)
    model, data = DLWGD(nw, df, 1., 1., 0.9, nt)
    chain = RevJumpChain(data=data, model=deepcopy(model), prior=prior)
    init!(chain)
    rjmcmc!(chain, n, trace=1, show=10)
    posterior_Σ!(chain)
    posterior_E!(chain)
    chain
end

function output(chain, x, burnin=1000)
    @unpack model, rates, wgds, Σ, η = x
    trace = chain.trace[burnin:end,:]
    qs = Dict(Symbol("q$i")=>model[i][:q] for (i,wgd) in sort(wgds))
    λs = Dict(Symbol("λ$i")=>rates[1,i] for i in 1:size(rates)[2])
    μs = Dict(Symbol("μ$i")=>rates[2,i] for i in 1:size(rates)[2])
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
    df[isnothing.(df[!,:gmean]),:gmean] .= NaN
    df[isnothing.(df[!,:gstd]), :gstd]  .= NaN
    df
end

function main()
    isdir(outdir) ? outdir : mkdir(outdir)
    m, p = DLWGD(nw, 2., 2., 0.9, Branch)
    x = rand(simprior, m)
    d = simulate(x.model, N, clade1)
    c = inference(nw, d, x.wgds, infprior, n)
    out = output(c, x, burnin)
    bfs = branch_bayesfactors(c, burnin)
    CSV.write(joinpath(outdir,
        "$(savename(params)).$(simid).bfs.csv"), bfs)
    CSV.write(joinpath(outdir,
        "$(savename(params)).$(simid).trace.csv"), c.trace)
    JLD.save( joinpath(outdir,
        "$(savename(params)).$(simid).wgdtrace.jld"), "wgds", get_wgdtrace(c))
    return c
end

chain = main()
