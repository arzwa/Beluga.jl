# This should be a script for non-julia users to run the rjMCMC as in my paper
env_dir = "/home/arzwa/dev/Beluga/"
using Pkg; Pkg.activate(env_dir)
using DataFrames, CSV, Distributions, Parameters, JLD
using Beluga

# configuration ________________________________________________________________
# this could end up in an argparse ind of thing
treefile = "test/data/plants2.nw"
datafile = "test/data/dicots-f01-100.csv"
outdir   = "/tmp/irmcmc"
niter    = Inf
burnin   = 1000
saveiter = 2500
ppsiter  = saveiter
params   = (
    theta0 = 1.5, sigma0 = 0.5, cov0 = 0.0,
    sigma  = 0.5, cov    = 0.45,
    qa     = 1.0, qb     = 1.0,
    etaa   = 3.0, etab   = 1.0,
    pk     = 0.0, bnd    =  20,
    qka    = 1.0, qkb    =  5,
    lak    = 1.0
)

# script _______________________________________________________________________
isdir(outdir) ? nothing : mkdir(outdir)
@info "params" params
@unpack theta0, sigma0, cov0, cov, sigma = params
@unpack etaa, etab, qa, qb, pk, bnd, qka, qkb, lak = params
nw = open(treefile, "r") do f ; readline(f); end
df = CSV.read(datafile, delim=",")
d, p = DLWGD(nw, df, theta0, theta0, 0.9, Branch)

# prior
prior = IidRevJumpPrior(
    Σ₀=[sigma cov ; cov sigma],
    X₀=MvNormal(log.([theta0, theta0]), [sigma0 cov0 ; cov0 sigma0]),
    πK=pk == 0 ? DiscreteUniform(0, bnd) : UpperBoundedGeometric(p, bnd),
    πq=Beta(qa,qb),
    πη=Beta(etaa,etab),
    Tl=treelength(d))

chain = RevJumpChain(data=p, model=deepcopy(d), prior=prior)
Beluga.init!(chain, qkernel=Beta(qka, qkb), λkernel=Exponential(lak))

function main(chain, outdir, niter, burnin, saveiter, ppsiter)
    gen = 0
    while gen < niter
        rjmcmc!(chain, saveiter, show=10, trace=1, rjstart=0)
        gen += saveiter
        posterior_Σ!(chain)
        posterior_E!(chain)
        @info "Saving trace (gen = $gen)"
        CSV.write(joinpath(outdir, "trace.csv"), chain.trace)
        JLD.save(joinpath(outdir, "wgdtrace.jld"), "wgds", get_wgdtrace(chain))
        @info "Computing Bayes factors"
        bfs = branch_bayesfactors(chain, burnin)
        CSV.write(joinpath(outdir, "bfs.csv"), bfs)
        if gen % ppsiter == 0
            @info "Doing posterior predicive simulations (PPS)"
            pps = PostPredSim(chain, df, 1000, burnin=burnin)
            JLD.save(joinpath(outdir,"pps.jld"), "pps", pps)
            @info "PPS results" pp_pvalues(pps);
        end
    end
    return chain
end

chain = main(chain, outdir, niter, burnin, saveiter, ppsiter)
