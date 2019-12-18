using Distributed
@everywhere env_dir = "."
@everywhere using Pkg; @everywhere Pkg.activate(env_dir)
using DataFrames, CSV, Distributions, Parameters, JLD
@everywhere using Beluga


# Configuration ________________________________________________________________
config = (
    treefile = "test/data/sim100/plants2.nw",
    trace    = "/home/arzwa/rjumpwgd/data/dicots/ormcmc_2222860/trace.csv",
    outdir   = length(ARGS) > 0 ? ARGS[1] : "/tmp/sims",
    clade1   = [:bvu, :sly, :ugi, :cqu],
    N        = 100,
    rj       = true,
    pp       = true,
    niter    = 2500,
    burnin   = 500,
    theta0   = 1.5, sigma0 = 0.5, cov0 = 0.45,
    sigma    = 1.0, cov    = 0.0,
    qa       = 1.0, qb     = 3.0,
    etaa     = 3.0, etab   = 1.0,
    pk       = DiscreteUniform(0, 20),
    ksim     = Binomial(10, 0.5),
    qsim     = Beta(2,3),
    kernel   = Beluga.DropKernel(qkernel=Beta(1,3)),
    expected = nothing
)

# methods ______________________________________________________________________
function setwgds!(trace)
    wgds = Meta.parse.(trace[!,:wgds])
    trace[!,:wgds] = eval.(wgds)
end

function addrandwgds!(model, πk, πq)
    Beluga.removewgds!(model)
    k = rand(πk)
    wgds = Dict()
    for i=1:k
        n, t = Beluga.randpos(model)
        q = rand(πq)
        wgdnode = insertwgd!(model, n, t, q)
        child = Beluga.nonwgdchild(wgdnode)
        wgds[wgdnode.i] = (child.i, q)
    end
    wgds
end

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

function output(chain, model, wgds, burnin=1000)
    trace = chain.trace[burnin:end,:]
    rates = Beluga.getrates(model)
    η = model[1][:η]
    qs = Dict(Symbol("q$i")=>model[i][:q] for (i,wgd) in sort(wgds))
    λs = Dict(Symbol("λ$i")=>rates[1,i] for i in 1:size(rates)[2])
    μs = Dict(Symbol("μ$i")=>rates[2,i] for i in 1:size(rates)[2])
    d = merge(qs, λs, μs)
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
function main(config)
    @unpack treefile, trace, outdir, clade1  = config
    @unpack niter, burnin, N, rj, pp = config
    @unpack theta0, sigma0, cov0, cov, sigma, ksim, qsim = config
    @unpack etaa, etab, qa, qb, pk, kernel, expected = config

    isdir(outdir) ? nothing : mkdir(outdir)
    @info "config" config
    open(joinpath(outdir, "config.txt"), "w") do f; write(f, string(config)); end

    trace = CSV.read(trace)
    setwgds!(trace)
    nw = open(treefile, "r") do f ; readline(f); end
    m, p  = DLWGD(nw, 2., 2., 0.9, Branch)

    # prior
    prior = IidRevJumpPrior(
        Σ₀=[sigma cov ; cov sigma],
        X₀=MvNormal(log.([theta0, theta0]), [sigma0 cov0 ; cov0 sigma0]),
        πK=pk,
        πq=Beta(qa,qb),
        πη=Beta(etaa,etab),
        Tl=treelength(m),
        πE=expected)

    model = m(rand(trace))
    @info "rates" Beluga.getrates(model)
    wgds  = addrandwgds!(model, ksim, qsim)
    @info "WGDs" wgds Beluga.getwgds(model)
    data  = simulate(model, N, clade1)
    @info "Simulated data" data

    c = rj ? rj_inference(nw, data, prior, kernel, niter) :
             fixed_inference(nw, data, wgds, infprior, n)
    wgdtrace = get_wgdtrace(c)

    out = output(c, model, wgds, burnin)
    bfs = branch_bayesfactors(c, burnin)
    write_wgds(joinpath(outdir, "wgds.csv"), wgds)
    CSV.write(joinpath(outdir, "sim.csv"), out)
    CSV.write(joinpath(outdir, "counts.csv"), data)
    CSV.write(joinpath(outdir, "bfs.csv"), bfs)
    CSV.write(joinpath(outdir, "trace.csv"), c.trace)
    JLD.save( joinpath(outdir, "wgdtrace.jld"), "wgds", wgdtrace)
    if pp
        @info "Doing posterior predicive simulations (PPS)"
        pps = PostPredSim(c, data, 1000, burnin=burnin)
        JLD.save(joinpath(outdir, "pps.jld"), "pps", pps)
        @info "PPS results" pp_pvalues(pps);
    end
    return c, out, wgds
end

chain, out = main(config)
