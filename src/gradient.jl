#= NOTE: This approach for computing the gradient has quite some overhead
since we compute W for each vector again. It is however the only way to get
ForwardDiff to work with the distributed approach. =#
# FIXME!! This worked in c948fdbc94a52e51111e95ffbc0d42d69b1b203c (wgds branch)
# but now returns NaNs! Maybe it has something to do with porting all
# computation on a log scale...

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
    return g
end
