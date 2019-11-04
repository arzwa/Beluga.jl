using DistributedArrays


# Profile
# =======
# NOTE not sure if really necessary to have the `xp` field, we could esaily
# adapt the csuros-miklos algorithm to use the x of the `nonwgdchild` instead.
# an alternative, that should be seriously looked at, is to use a dict of
# vectors instead of a matrix

abstract type AbstractProfile end

"""
    Profile{V<:Real}

Struct for a phylogenetic profile of a single family. Geared towards MCMC
applications.
"""
@with_kw mutable struct Profile{T<:Real} <: AbstractProfile
    x ::Vector{Int64}
    xp::Vector{Int64} = deepcopy(x)
    L ::Matrix{T}
    Lp::Matrix{T} = deepcopy(L)
end

const PArray{T} = DArray{Profile{T},1,Array{Profile{T},1}} where T<:Real

Profile(x::Vector{Int64}, n=length(x), m=maximum(x)+1) = Profile(x=x, L=minfs(Float64,m,n))
Profile(X::Matrix{Int64}) = distribute([Profile(X[:,i]) for i=1:size(X)[2]])

logpdf!(d::DLWGD, p::PArray) = mapreduce((x)->logpdf!(x.Lp, x.xp, d), +, p)
logpdf!(n::ModelNode, p::PArray) = mapreduce((x)->logpdf!(x.Lp, x.xp, n), +, p)
logpdfroot(n::ModelNode, p::PArray) = mapreduce((x)->logpdfroot(x.Lp, n), +, p)

set!(p::PArray) = map!(_set!, p, p)
rev!(p::PArray) = map!(_rev!, p, p)

# function _set!(p::Profile)  # slightly faster
#     copyto!(p.x, p.xp)
#     copyto!(p.L, p.Lp)
#     p
# end

function _set!(p::Profile)
    p.x = deepcopy(p.xp)
    p.L = deepcopy(p.Lp)
    p
end

# function _rev!(p::Profile)  # slightly faster
#     copyto!(p.xp, p.x)
#     copyto!(p.Lp, p.L)
#     p
# end

function _rev!(p::Profile)
    p.xp = deepcopy(p.x)
    p.Lp = deepcopy(p.L)
    p
end

extend!(p::PArray, i::Int64) = map!((x)->_extend!(x,i), p, p)
shrink!(p::PArray, i::Int64) = map!((x)->_shrink!(x,i), p, p)

function _extend!(p::Profile{T}, i::Int64) where T<:Real
    p.xp = vcat(p.xp, p.xp[i], p.xp[i])
    p.Lp = hcat(p.Lp, minfs(T, size(p.Lp)[1], 2))
    p
end

function _shrink!(p::Profile{T}, i::Int64) where T<:Real
    p.xp = vcat(p.xp[1:i-1], p.xp[i+2:end])
    p.Lp = hcat(p.Lp[:,1:i-1], p.Lp[:,i+2:end])
    p
end
