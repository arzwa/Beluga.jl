[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://arzwa.github.io/Beluga.jl/dev)
[![](https://travis-ci.com/arzwa/Beluga.jl.svg?branch=master)](https://travis-ci.com/arzwa/Beluga.jl)

Copyright (C) 2020 Arthur Zwaenepoel

VIB/UGent center for plant systems biology - [Bioinformatics & evolutionary genomics group](http://bioinformatics.psb.ugent.be/beg/)

# Beluga

## Dependencies

Beluga requires unregistered dependencies, to install Beluga, fire up a julia
session, hit `]` and add the following:

```
(v1.3) pkg> add https://github.com/arzwa/NewickTree.jl
(v1.3) pkg> add https://github.com/arzwa/AdaptiveMCMC.jl
(v1.3) pkg> add https://github.com/arzwa/Beluga.jl
```

## Usage

```julia
using Beluga, CSV, DataFrames, Parameters

# get some data
tree = readline("example/dicots/dicots.nw")
df = CSV.read("example/dicots/dicots-f01-25.csv")

# construct model and profile
λ, μ, η = 1.0, 0.92, 0.66
model, profile = DLWGD(tree, df, λ, μ, η)  

# compute log-likelihood
l = logpdf!(model, profile)

# get model parameters as vector = [λ1, ..., μ1, ..., q1, ..., η])
v = asvector(model)

# construct new model based on the old one and a new parameter vector
x = rand(length(v))
model = model(x)

# compute log likelihood again
l = logpdf!(model, profile)

# change parameters at node 5
update!(model[5], (λ=1.5, μ=1.2))

# change η parameter at root
update!(model[1], :η, 0.91)

# recompute likelihood efficiently starting from node 5
l = logpdf!(model[5], profile)

# gradient
g = gradient(model, profile)

# add a WGD node above node 6 at a distance 0.1 with q=0.25
addwgd!(model, model[6], 0.1, 0.25)
extend!(profile, 6);

# compute the log-likelihood, now for the model with the WGD
logpdf!(model, profile)

# simulate a random phylogenetic profile under the model
rand(model)

# simulate a data set of 100 profiles
rand(model, 100)

# independent rates prior (check & adapt default settings!)
prior = IRRevJumpPrior()
logpdf(prior, model)

# sample random model from prior
@unpack model, Σ = rand(prior, model)

# reversible jump chain
model, profile = DLWGD(tree, df, λ, μ, η)  
chain = RevJumpChain(data=profile, model=model, prior=prior)

# run chain (fixed dimension - no reversible jump)
init!(chain)
mcmc!(chain, 100, show=1)

# run chain (variable dimensions - with reversible jump)
rjmcmc!(chain, 100, show=1)

```

**Notes:**

- Gradients require NaN-safe mode enabled in ForwardDiff: The
following should work for most people:

```
sed -i 's/NANSAFE_MODE_ENABLED = false/NANSAFE_MODE_ENABLED = true/g' ~/.julia/packages/ForwardDiff/*/src/prelude.jl
```

## Citation

Beluga.jl is developed by Arthur Zwaenepoel at the VIB/UGent center for plant
systems biology (bioinformatics and evolutionary genomics group). A preprint on the
reversible-jump sampler for WGD inference implemented in this library can be found
[at BioRXiv](https://www.biorxiv.org/content/early/2020/01/25/2020.01.24.917997).

```
@article {zwaenepoel2020,
	author = {Zwaenepoel, Arthur and Van de Peer, Yves},
	title = {Model-based detection of whole-genome duplications in a phylogeny},
	elocation-id = {2020.01.24.917997},
	year = {2020},
	doi = {10.1101/2020.01.24.917997},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/01/25/2020.01.24.917997},
	eprint = {https://www.biorxiv.org/content/early/2020/01/25/2020.01.24.917997.full.pdf},
	journal = {bioRxiv}
}
```
