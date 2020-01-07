# Beluga.jl

Beluga is a julia library for the statistical analysis of gene family evolution
using phylogenetic birth-death processes. It's somewhat related to
[Whale.jl](https://arzwa.github.io/Whale.jl/dev/index.html) as it implements
models of duplication loss and whole-genome duplication, but employs gene count
data instead of gene trees. The library implements the MCMC sampler of
Zwaenepoel & Van de Peer (2019) as well as the reversible-jump MCMC sampler of
Zwaenepoel & Van de Peer (2020, *in preparation*).


## Data preparation

To perform analyses with Beluga, you will need  

1. An ultrametric species tree (time tree)
2. A *phylogenetic profile matrix*. If you have a bunch of protein fasta files
   for a set of species of interest, this can be easily obtained using e.g.
   [OrthoFinder](https://github.com/davidemms/OrthoFinder).

!!! note
    If you only need the phylogenetic profile matrix from OrthoFinder, be
    sure to use the `-og` flag to stop the OrthoFinder pipeline after orthogroup
    inference. The phylogenetic profile matrix can be found in the
    `Orthogroups.GeneCount.tsv` file.

## Reversible-jump MCMC for the inference of WGDs in a phylogeny

Load Beluga and required packages:

```@example rjexample
using Beluga, CSV, Distributions
```

Or, if you are running a julia session with multiple processes (when you started
julia with `-p` option, or manually added workers using `addprocs`, see [julia docs](https://docs.julialang.org/en/v1/manual/parallel-computing/index.html#Parallel-Computing-1)), run

```julia
using CSV, Distributions
@everywhere using Beluga  # if julia is running in a parallel environment
```

Then get some data (for instance from the `example` directory of the git repo)

```@example rjexample
nw = readline("../../example/dicots/dicots.nw")
df = CSV.read("../../example/dicots/dicots-f01-25.csv")
model, data = DLWGD(nw, df, 1.0, 1.2, 0.9)
```

`model` now refers to the duplication-loss and WGD model (with no WGDs for now),
`data` refers to the phylogenetic profile matrix. The model was initialized with
all duplication and loss rates set to 1 and 1.2 respectively. You can check
this easily:


```@example rjexample
getrates(model)
```

or to get the full parameter vector:

```@example rjexample
asvector(model)
```

and you can easily compute likelihoods (and gradients)

```@example rjexample
logpdf!(model, data)
```

Now we proceed to the hierarchical model. There is no fancy DSL available (à la
Turing.jl, Mamba.jl or Soss.jl) but we use a fairly flexible prior struct.
Here is an exaple for the (recommended) independent rates (IR) prior:

```@example rjexample
prior = IRRevJumpPrior(
    # prior covariance matrix (Inverse Wishart prior)
    Σ₀=[1 0. ; 0. 1],

    # multivariate prior on the mean duplication and loss rates
    X₀=MvNormal([0., 0.], [1 0. ; 0. 1]),  

    # prior on the number of WGDs (can be any discrete distribution)
    πK=DiscreteUniform(0,20),

    # prior on the WGD retention rate (should be `Beta`)
    πq=Beta(1,1),

    # prior on η
    πη=Beta(3,1),

    # tree length (determines prior on WGD times)
    Tl=treelength(model))
```

We can then construct a chain and run it

```@example rjexample
chain = RevJumpChain(data=data, model=model, prior=prior)
init!(chain)
rjmcmc!(chain, 100, show=10)
```

This will log a part of the trace to stdout every `show` iterations, so that
we're able to monitor a bit whether everything looks sensible. Of course in
reality you would sample way longer than n=100 iterations.

Now we can analyze the trace (in `chain.trace`), write it to a file, etc. We
can also compute Bayes factors to get an idea of the number of WGDs for each
branch in the species tree.

```@example rjexample
bayesfactors(chain, burnin=10)
```

There are some plotting recipes included in the BelugaPlots package. You may
want to try out the following:

```julia
using BelugaPlots, Plots, StatsPlots
BelugaPlots.traceplot(chain.trace, burnin=10)
BelugaPlots.traceplot(chain.trace, burnin=10, st=density)
BelugaPlots.bfplot(bayesfactors(chain, burnin=10))
BelugaPlots.ktraceplot(chain.trace, burnin=10)
posteriorE!(chain)
BelugaPlots.eplot(chain.trace)
```

To obtain a tree with the marginal posterior mean duplication and loss rate
estimates, you can try the following (note this is not likely to give you a
very nice image unless your taxon IDs are three letter codes)

```julia
BelugaPlots.doubletree(chain, burnin=10)
```
