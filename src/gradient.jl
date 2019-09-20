#= NOTE: This approach for computing the gradient has quite some overhead
since we compute W for each vector again. It is however the only way to get
ForwardDiff to work with the distributed approach. =#
# Profile arrays (DArray)
gradient(d::PhyloBDP, p::PArray) = mapreduce((x)->gradient(d, x.x), +, p)

# AbstractMatrix implementation
gradient(d::PhyloBDP, X::AbstractMatrix{Int64}) =
    sum(Array(mapslices((x)->gradient(d, x), X, dims=2)'), dims=2)[:,1]

# gradient for a single vector
function gradient(d::PhyloBDP, x::AbstractVector{Int64})
    v = asvector(d)
    f = (u) -> logpdf(d(u), x)
    g = ForwardDiff.gradient(f, v)
    return g[:, 1]
end

# vector based constructor (to core.jl?)
function (d::DuplicationLossWGD)(θ::Vector)
    @unpack tree, value = d
    η = pop!(θ)
    q = [pop!(θ) for i=1:nwgd(d.tree)]
    μ = [pop!(θ) for i=1:length(θ)÷2]
    DuplicationLossWGD(tree, θ, μ, q, η, value.m)
end

# to core.jl?
asvector(d::DuplicationLossWGD) = [d.λ ; d.μ ; d.q ; d.η ]
