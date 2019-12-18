using Distributed
@everywhere env_dir = "."
@everywhere using Pkg; @everywhere Pkg.activate(env_dir)
using DataFrames, CSV, Distributions, Parameters, JLD
@everywhere using Beluga


# Configuration ________________________________________________________________
config = (
    treefile = "test/data/sim100/plants2.nw",
    trace    = "test/data/sim100/set6_c10_counts.csv",
    outdir   = ARGS[1],
    clade1   = [:bvu, :sly, :ugi, :cqu]
    rj       = true,
    niter    = 11000,
    burnin   = 1000,
    saveiter = 2500,
    ppsiter  = 2500,
    theta0   = 1.5, sigma0 = 0.5, cov0 = 0.45,
    sigma    = 1.0, cov    = 0.0,
    qa       = 1.0, qb     = 3.0,
    etaa     = 3.0, etab   = 1.0,
    pk       = DiscreteUniform(0, 20),
    kernel   = Beluga.DropKernel(qkernel=Beta(1,3)),
    expected = Normal(1, 0.1),
)

# script _______________________________________________________________________
@unpack treefile, trace, outdir = config
@unpack niter, burnin, saveiter, ppsiter = config
@unpack theta0, sigma0, cov0, cov, sigma = config
@unpack etaa, etab, qa, qb, pk, kernel, expected = config
isdir(outdir) ? nothing : mkdir(outdir)
@info "config" config
open(joinpath(outdir, "config.txt"), "w") do f; write(f, string(config)); end

trace = CSV.read(trace)
nw = open(treefile, "r") do f ; readline(f); end
df = CSV.read(datafile, delim=",")

# prior
prior = IidRevJumpPrior(
    Σ₀=[sigma cov ; cov sigma],
    X₀=MvNormal(log.([theta0, theta0]), [sigma0 cov0 ; cov0 sigma0]),
    πK=pk,
    πq=Beta(qa,qb),
    πη=Beta(etaa,etab),
    Tl=treelength(d),
    πE=expected)


# methods ______________________________________________________________________
function simulate(model, N, clade1)
    clade2 = [v for (k,v) in  model.leaves if !(v in clade1)]
    rand(model, N, [clade1,clade2])
end

function rj_inference(nw, df, prior, kernel, n=6000, nt=Branch)
    r = rand(prior.X₀)
    η = rand(prior.πη)
    model, data = DLWGD(nw, df, r[1], r[2], η, nt)
    chain = RevJumpChain(data=data, model=model, prior=prior, kernel=kernel)
    init!(chain)
    rjmcmc!(chain, n, trace=1, show=10, rjstart=0)
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

function write_wgds(fname, wgds)
    open(fname, "w") do f
        write(f, "wgdnode,branch,q\n")
        for (k,v) in wgds
            write(f, "$k,$(v[1]),$(v[2])\n")
        end
    end
end


# run simulation
function main()
    isdir(outdir) ? outdir : mkdir(outdir)
    m, p = DLWGD(nw, 2., 2., 0.9, Branch)
    x = rand(simprior, m)
    d = simulate(x.model, N, clade1)
    c = rj ? rj_inference(nw, d, infprior, n) :
             fixed_inference(nw, d, x.wgds, infprior, n)
    out = output(c, x, burnin)
    bfs = branch_bayesfactors(c, burnin)
    write_wgds(joinpath(outdir,
        "$(savename(params)).$(simid).wgds.csv"), x.wgds)
    CSV.write(joinpath(outdir,
        "$(savename(params)).$(simid).sim.csv"), out)
    CSV.write(joinpath(outdir,
        "$(savename(params)).$(simid).counts.csv"), d)
    CSV.write(joinpath(outdir,
        "$(savename(params)).$(simid).bfs.csv"), bfs)
    CSV.write(joinpath(outdir,
        "$(savename(params)).$(simid).trace.csv"), c.trace)
    JLD.save( joinpath(outdir,
        "$(savename(params)).$(simid).wgdtrace.jld"), "wgds", get_wgdtrace(c))
    if pp
        pps = PostPredSim(c, d, 1000, burnin=burnin)
        JLD.save(joinpath(outdir,
            "$(savename(params)).$(simid).pps.jld"), "pps", pps)
    end
    return c, out
end

chain, out = main()
