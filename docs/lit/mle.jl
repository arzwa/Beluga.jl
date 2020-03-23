# # Maximum likelihood estimation for the DLWGD model using `Beluga.jl`

# We'll need the folowing packages loaded:
using Beluga, CSV, DataFrames, Optim

# First let's get a random data set
nw = "(D:0.1803,(C:0.1206,(B:0.0706,A:0.0706):0.0499):0.0597);"
m, _ = DLWGD(nw, 1.0, 1.5, 0.8)
df = rand(m, 1000, condition=Beluga.rootclades(m))
first(df, 5)

# This illustrates how to simulate data from a DLWGD model instance. We created a
# DLWGD model for the species tree in the Newick string `nw` with constant rates
# of duplication (1.0) and loss (1.5) across the tree and a geometric prior distribution
# on the number of lineages at the root with mean 1.25 (`η = 0.8`). We then simulated
# 1000 gene family profiles subject to the condition that there is at least one gene
# observed at the leaves in each clade stemming from the root.

# No we set up some stuff to do maximum likelihood estimation for the constant rates
# model. But first we'll have to get the proper data structure for the profiles
model, data = DLWGD(nw, df)

# Note that we already got a model object above, however some internals of the model
# are dependent on the data to which it will be applied (due to the algorithm of
# Csuros & Miklos).

# Now for some functions needed for the ML estimation
n = Beluga.ne(model) + 1  # number of nodes = number of edges + 1
fullvec(θ, η=0.8, n=n) = [repeat([exp(θ[1])], n); repeat([exp(θ[2])], n) ; η]
f(θ) = -logpdf!(model(fullvec(θ)), data)
g!(G, θ) = G .= -1 .* Beluga.gradient_cr(model(fullvec(θ)), data)

# The default DLWGD model is parameterized with a rate for each node in the tree,
# and a DLWGD model with `n` nodes and `k` WGDs can be constructed based on a
# given DLWGD model instance and a vector looking like
# `[[λ1 … λn] ; [μ1 … μn] ; [q1 … qk] ; η]`. To be clear, consider the following
# example code
@show v = asvector(model)
newmodel = model(rand(length(v)))
@show asvector(newmodel)

# In the constant rates model, we assume all duplication rates and all loss
# rates are identical across the tree respectively. The `fullvec(θ)` function
# defined above will construct a full model vector, from which we can construct
# a `DLWGD` instance, from the simpler vector `θ = [log(λ), log(μ)]` that defines
# the constant rates model.

# Now for optimization. Let's try two optimization algorithms, one only using the
# likelihood (using the Nelder-Mead downhill simplex algorithm), and another using
# gradient information (using the LBFGS algorithm)

# Using Nelder-Mead
init = randn(2)
results = optimize(f, init)
@show exp.(results.minimizer)

# Using LBFGS (requires gradients!)
init = randn(2)
results = optimize(f, g!, init)
@show exp.(results.minimizer)

# !!! warning
#     Currently [the gradient seems to only work in `NaN` safe mode](http://www.juliadiff.org/ForwardDiff.jl/stable/user/advanced/#Fixing-NaN/Inf-Issues-1).
#     In order to enable `NaN` safe mode, you should change a line in the ForwardDiff
#     source code. On Linux, and assuming you use julia v1.3, the following should
#     work for most people:
#
#     ```
#     rm -r ~/.julia/compiled/v1.3/ForwardDiff
#     sed -i 's/NANSAFE_MODE_ENABLED = false/NANSAFE_MODE_ENABLED = true/g' \
#     ~/.julia/packages/ForwardDiff/*/src/prelude.jl
#     ```
