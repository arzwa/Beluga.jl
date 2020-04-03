# # WGD inference using reversible-jump MCMC and gene count data

# Load Beluga and required packages:
using Beluga, CSV, Distributions, Random
Random.seed!(23031964)

# Or, if you are running a julia session with multiple processes (when you started julia with `-p` option, or manually added workers using `addprocs`, see [julia docs](https://docs.julialang.org/en/v1/manual/parallel-computing/index.html#Parallel-Computing-1)),
# run:

# ```julia
# using CSV, Distributions
# @everywhere using Beluga  # if julia is running in a parallel environment
# ```

# Then get some data (for instance from the `example` directory of the git repo)
nw = readline(joinpath(@__DIR__, "../../example/9dicots/9dicots.nw"))
df = CSV.read(joinpath(@__DIR__, "../../example/9dicots/9dicots-f01-25.csv"), type=Int)
model, data = DLWGD(nw, df, 1.0, 1.2, 0.9)

# `model` now refers to the duplication-loss and WGD model (with no WGDs for now), `data` refers to the phylogenetic profile matrix. The model was initialized with all duplication and loss rates set to 1 and 1.2 respectively. You can check this easily:
getrates(model)

# or to get the full parameter vector:
asvector(model)

# Now you can easily compute log-likelihoods (and gradients thereof)
logpdf!(model, data)

# so we can do likelihood-based inference (either maximum-likelihood or Bayesian). For the kind of problems tackled here, the only viable option however is Bayesian inference.

# We proceed by specifying the hierarchical prior on the gene family evolutionary process. There is no DSL available (à la Turing.jl, Mamba.jl, Soss.jl or Stan) but we use a fairly flexible prior struct. Here is an exaple for the (recommended) bivariate independent rates (IR) prior:
prior = IRRevJumpPrior(
    Ψ=[1 0. ; 0. 1],
    X₀=MvNormal([0., 0.], [1 0. ; 0. 1]),
    πK=DiscreteUniform(0,20),
    πq=Beta(1,1),
    πη=Beta(3,1),
    equal=true,
    Tl=treelength(model))

# `Ψ` is the prior covariance matrix for the Inverse-Wishart distribution.
# `X₀` is the multivariate Normal prior on the mean duplication and loss rates.
# `πK` is the prior on the number of WGDs (i.e. the model indicator). `πq` is
# the Beta prior on the retention rates (iid). `πη` is the hyperprior on the
# parameter of the geometric distribution on the number of ancestral lineages
# at the root. `Tl` is the tree length (and is used for the prior on the WGD
# ages).

# To sample across model-space (i.e. where we infer the number and locations of
# WGDs), we need the reversible jump algorithm. There are several reversible-jump
# kernels implemented. The simplest is the aptly named `SimpleKernel`, which
# introduces new WGDs with a random retention rate drawn from a Beta distribution.
kernel = SimpleKernel(qkernel=Beta(1,3))

# We can then construct a chain
chain = RevJumpChain(data=data, model=model, prior=prior, kernel=kernel)
init!(chain)

# and sample from it:
rjmcmc!(chain, 1000, show=50)

# This will log a part of the trace to stdout every `show` iterations, so that
# we're able to monitor a bit whether everything looks sensible. Of course in
# reality you would sample way longer than `n=1000` iterations, but since this
# page has to be generated in decent time using a single CPU I'll keep it to 1000
# here.

# Now the computer has done Bayesian inference, and we have to do our part. We
# can analyze the trace (in `chain.trace`), write it to a file, compute statistics, etc.
# Here are some trace plots:
using Plots, DataFrames, LaTeXStrings
burnin=100
p1 = plot(chain.trace[burnin:end,:λ1], label=L"\lambda_1")
plot!(chain.trace[burnin:end,:μ1], label=L"\mu_1")
p2 = plot(chain.trace[burnin:end,:λ8], label=L"\lambda_8")
plot!(chain.trace[burnin:end,:μ8], label=L"\mu_8")
p3 = plot(chain.trace[!,:k], label=L"k")
p4 = plot(chain.trace[!,:η1], label=L"\eta")
plot(p1,p2,p3,p4, grid=false, layout=(2,2))

# Clearly, we should sample way longer to get decent estimates for the duplication
# rates (`λ`), loss rates (`μ`) and number of WGDs (`k`). Note how `η` is quite
# well-sampled already.

# We can also check the effective sample size (ESS) of the model indicator
# variable, for that we will use the method of Heck et al. (2019) implemented in
# the module [`DiscreteMarkovFit.jl`](https://github.com/arzwa/DiscreteMarkovFit.jl):
using DiscreteMarkovFit

# We'll discard a burnin of 100 iterations
d = ObservedBirthDeathChain(Array(chain.trace[100:end,:k]))
out = DiscreteMarkovFit.sample(d, 10000)

# This shows the effective sample size for the number of WGDs and the associated
# posterior probabilities. The maximum a posteriori (MAP) number of WGDs here is three.
# When doing a serious analysis, one should aim for higher ESS values of course.

# We can also compute Bayes factors to get an idea of the number of WGDs for each
# branch in the species tree.
bfs = bayesfactors(chain, burnin=100);

# This suggests strong support for WGD in quinoa (`cqu`), for which we know
# the genome shows a strong signature of an ancestral WGD.  Note that
# we already detect these WGDs using a mere 25 gene families as data!

# A plot of the posterior probabilities for the number of WGDs on each branch
# is a nice way to summarize the rjMCMC output:
plots = [bar(g[!,:k], g[!,:p1], color=:white,
            title=join(string.(g[1,:clade]), ", "))
            for g in groupby(bfs, :branch)]
xlabel!.(plots[end-3:end], L"k")
ylabel!.(plots[1:4:end], L"P(k|X)")
plot(plots..., grid=false, legend=false,
    ylim=(0,1), xlim=(-0.5,3.5),
    yticks=[0, 0.5, 1], xticks=0:3,
    title_loc=:left, titlefont=8)
