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

set!(p::PArray) = map!(_set!, p, p)
rev!(p::PArray) = map!(_rev!, p, p)

function _set!(p::Profile)
    copyto!(p.x, p.xp)
    copyto!(p.L, p.Lp)
    p
end

function _rev!(p::Profile)
    copyto!(p.xp, p.x)
    copyto!(p.Lp, p.L)
    p
end
