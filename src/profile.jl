# Profile
# =======
# NOTE not sure if really necessary to have the `xp` field, we could easily
# adapt the csuros-miklos algorithm to use the x of the `nonwgdchild` instead.

# an alternative, that should be seriously looked at, is to use a dict of
# vectors instead of a matrix

abstract type AbstractProfile{T} end

"""
    Profile{T<:Real}

Struct for a phylogenetic profile of a single family. Geared towards MCMC
applications (temporary storage fields) and parallel applications (using
DArrays). See also `PArray`.
"""
@with_kw mutable struct Profile{T<:Real} <: AbstractProfile{T}
    x ::Vector{Int64}
    xp::Vector{Int64} = deepcopy(x)
    L ::Matrix{T}
    Lp::Matrix{T} = deepcopy(L)
end

const PArray{T} = DArray{Profile{T},1,Array{Profile{T},1}} where T<:Real
PArray() = distribute([Profile(nothing)])

Profile(x::Nothing) = Profile(Int64[], Int64[], zeros(0,0), zeros(0,0))
Profile(x::Vector{Int64}, n=length(x), m=maximum(x)+1) =
    Profile(x=x, L=minfs(Float64,m,n))
Profile(X::Matrix{Int64}) = distribute([Profile(X[:,i]) for i=1:size(X)[2]])


# NOTE: the length hack is quite ugly, maybe nicer to have a type for empty
# (mock) profiles [for sampling from the prior alone in MCMC applications]
logpdf!(d::DLWGD, p::PArray) = length(p[1].x) == 0 ?
    0. : mapreduce((x)->logpdf!(x.Lp, d, x.xp), +, p)

logpdf!(n::ModelNode, p::PArray) = length(p[1].x) == 0 ?
    0. : mapreduce((x)->logpdf!(x.Lp, n, x.xp), +, p)

logpdfroot(n::ModelNode, p::PArray) = length(p[1].x) == 0 ?
    0. : mapreduce((x)->logpdfroot(x.Lp, n), +, p)

gradient(d::DLWGD, p::PArray) = mapreduce((x)->gradient(d, x.xp), +, p)


# Efficient setting/resetting
# copyto! approach is slightly faster, but not compatible with arrays of ≠ dims
set!(p::PArray) = map!(_set!, p, p)
rev!(p::PArray) = map!(_rev!, p, p)

function _set!(p::Profile)
    p.x = deepcopy(p.xp)
    p.L = deepcopy(p.Lp)
    p
end

function _rev!(p::Profile)
    p.xp = deepcopy(p.x)
    p.Lp = deepcopy(p.L)
    p
end

# function _set!(p::Profile)  # slightly faster
#     copyto!(p.x, p.xp)
#     copyto!(p.L, p.Lp)
#     p
# end

# function _rev!(p::Profile)  # slightly faster
#     copyto!(p.xp, p.x)
#     copyto!(p.Lp, p.L)
#     p
# end


# extend/shrink profiles (reversible jump MCMC applications)
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
