# # Sample from the prior using rjMCMC

# Running an MCMC algorithm without data should result in a sample from the prior.
# This is commonly used in complicated Bayesian inference settings such as those
# arising in phylogenetics to verify correct implemetation of an MCMC algorithm.
# Of course, this cannot idicate probems in the likelihood implementation! (which
# can also be quite complicated in phylogenetic applications, at least in my humble
# opinion).
using Distributions, Beluga, Plots, StatsPlots, LaTeXStrings, Random
Random.seed!(190894)

# Obtain the species tree
nw = readline(joinpath(@__DIR__, "../../example/9dicots/9dicots.nw"))
d, p = DLWGD(nw, 1., 1., 0.9)

# Calling `DLWGD` without a data frame with gene counts will result in a 'mock'
# phylogenetic profile for which the likelihood defaults to 1 (or log-likelihood
# to 0).
logpdf!(d, p)

# The prior is specified here
prior = IRRevJumpPrior(
    Ψ=[1 0.0 ; 0.0 1],
    X₀=MvNormal(log.([3,3]), [0.2 0.; 0. 0.2]),
    πK=DiscreteUniform(0, 10),
    πq=Beta(1,3),
    πη=Beta(3,1),
    Tl=treelength(d))
# `Ψ` is the prior covariance matrix for the Inverse-Wishart distribution.
# `X₀` is the multivariate Normal prior on the mean duplication and loss rates.
# `πK` is the prior on the number of WGDs (i.e. the model indicator). `πq` is
# the Beta prior on the retention rates (iid). `πη` is the hyperprior on the
# parameter of the geometric distribution on the number of ancestral lineages
# at the root. `Tl` is the tree length (and is used for the prior on the WGD
# ages).

# We have to set the rjMCMC kernel,
kernel = SimpleKernel(qkernel=Beta(1,3))

# construct the chain
chain = RevJumpChain(data=p, model=d, prior=prior, kernel=kernel)
init!(chain)

# and sample
@time rjmcmc!(chain, 5000, trace=5, show=1000)

# Now plot some stuff
p1 = bar(Beluga.freqmap(chain.trace[!,:k]), color=:white, title=L"k")
plot!(p1, prior.πK, color=:black, marker=nothing)
p2 = stephist(log.(chain.trace[!,:λ1]), fill=true, color=:black, alpha=0.2, normalize=true)
plot!(p2, Normal(log(3),√0.2), color=:black, linewidth=2, title=L"\lambda")
stephist!(log.(chain.trace[!,:λ9]), fill=true, color=:firebrick, alpha=0.2, normalize=true)
p3 = stephist(log.(chain.trace[!,:μ1]), fill=true, color=:black, alpha=0.2, normalize=true)
plot!(p3, Normal(log(3),√0.2), color=:black, linewidth=2, title=L"\mu")
stephist!(log.(chain.trace[!,:μ4]), fill=true, color=:firebrick, alpha=0.2, normalize=true)
p4 = stephist(chain.trace[!,:η1], color=:black, fill=true, alpha=0.2, normalize=true)
plot!(prior.πη, linewidth=2, color=:black, title=L"\eta")
plot(p1, p2, p3, p4, layout=(2,2), legend=false, grid=false, title_loc=:left)

# Note that the prior for duplication and loss rates can have very heavy tails
# under the molecular clock priors (see the distributions in red as examples).
