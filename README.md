# Beluga

(name is still WIP, I'm not that convinced)

## Usage

```julia
using Beluga, CSV, DataFrames, Parameters

# get some data
datadir = "test/data"

tree = open(joinpath(datadir, "plants1.nw"), "r") do f ; readline(f); end
df = CSV.read(joinpath(datadir, "plants1-100.tsv"), delim=",")

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
insertwgd!(model, model[6], 0.1, 0.25)
extend!(profile, 6);

# compute the log-likelihood, now for the model with the WGD
logpdf!(model, profile)

# simulate a random phylogenetic profile under the model
rand(model)

# simulate an data set of 100 profiles
rand(model, 100)

# independent rates prior (check & adapt default settings!)
prior = IidRevJumpPrior()
logpdf(prior, model)

# sample random model from prior
@unpack model, Σ = rand(prior, model)

# reversible jump chain
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
