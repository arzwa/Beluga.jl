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
