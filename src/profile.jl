abstract type AbstractProfile{T} end

# NOTE: april 6 2020: introduced site-patterns approach. Note that this is
# incompatible with family-specific rates if there is no marginalization over
# rate variation.

"""
    Profile{T<:Real}

Struct for a phylogenetic profile of a single family. Geared towards MCMC
applications (temporary storage fields) and parallel applications (using
DArrays). See also `PArray`.
"""
@with_kw mutable struct Profile{T<:Real} <: AbstractProfile{T}
    n ::Int # number of times this profie appears in the data
    x ::Vector{Int64}
    xp::Vector{Int64} = deepcopy(x)
    L ::Matrix{T}
    Lp::Matrix{T} = deepcopy(L)
end

"""
    PArray{T<:Real}

Ditributed array of phylogenetic profiles.
"""
const PArray{T} = DArray{Profile{T},1,Array{Profile{T},1}} where T<:Real
PArray() = distribute([Profile(nothing)])
Base.show(io::IO, P::PArray{T}) where T = write(io, "PArray{$T}($(length(P)))")

Profile(x::Nothing) = Profile(1, Int64[], Int64[], zeros(0,0), zeros(0,0))
Profile(x::Vector{Int64}, n=1, l=length(x), m=maximum(x)+1) =
    Profile(x=x, n=n, L=minfs(Float64,m,l))
Profile(X::Matrix{Int64}) = distribute([Profile(X[:,i]) for i=1:size(X)[2]])
# n is the vector of couts for the site patterns
Profile(X::Matrix{Int64}, n::Vector{Int64}) =
    distribute([Profile(X[:,i], n[i]) for i=1:length(n)])

# NOTE: the length hack is quite ugly, maybe nicer to have a type for empty
# (mock) profiles [for sampling from the prior alone in MCMC applications]
# XXX: There is a problem, we need the init kwarg to have type stability in the
# mapreduce operation, but the init kwarg does not work with the DArray
# mapreduce; maybe worthwhile to get an MWE and report an issue? Or more
# likely its due to some design mistake from my part... As a workaround for now
# we type-annotate the output.
"""
    logpdf!(d::DLWGD, p::PArray{T})
    logpdf!(n::ModelNode, p::PArray{T})

Accumulate the log-likelihood ℓ(λ,μ,q,η|X) in parallel for the phylogenetic
profile matrix. If the first argument is a ModelNode, this will recompute
the dynamic programming matrices starting from that node to save computation.
Assumes (of course) that the phylogenetic profiles are iid from the same DLWGD
model.
"""
logpdf!(d::DLWGD, p::PArray{T}) where T = length(p[1].x) == 0 ?
    zero(T) : mapreduce((x)->x.n*logpdf!(x.Lp, d, x.xp), +, p)::T

logpdf!(n::ModelNode, p::PArray{T}) where T = length(p[1].x) == 0 ?
    zero(T) : mapreduce((x)->x.n*logpdf!(x.Lp, n, x.xp), +, p)::T

logpdfroot(n::ModelNode, p::PArray{T}) where T = length(p[1].x) == 0 ?
    zero(T) : mapreduce((x)->x.n*logpdfroot(x.Lp, n), +, p)::T

"""
    gradient!(d::DLWGD, p::PArray{T})

Accumulate the gradient ∇ℓ(λ,μ,q,η|X) in parallel for the phylogenetic profile
matrix `p`.
"""
gradient(d::DLWGD, p::PArray{T}) where T =
    mapreduce((x)->x.n .* gradient(d, x.xp), +, p)::Vector{T}

gradient_cr(d::DLWGD, p::PArray{T}) where T =
    mapreduce((x)->x.n .* gradient_cr(d, x.xp), +, p)::Vector{T}

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

"""
    addwgds!(m::DLWGD, p::PArray, config::Array)

Add WGDs from array of named tuples e.g. [(lca="ath,cpa", t=rand(), q=rand())]
and update the profile array.
"""
function addwgds!(m::DLWGD, p::PArray, config::Array)
    for x in config
        n = lca_node(m, Symbol.(split(x.lca, ","))...)
        haskey(x, :l) && x.l == 3 ?
            addwgt!(m, n, n[:t]*x.t, x.q) :
            addwgd!(m, n, n[:t]*x.t, x.q)
        length(p[1].x) == 0 ? nothing : extend!(p, n.i)
    end
end

"""
    addwgds!(m::DLWGD, p::PArray, config::String)
    addwgds!(m::DLWGD, p::PArray, config::Dict{Int64,Tuple})

Add WGDs from a (stringified) dictionary (as in the wgds column of the
trace data frame in rjMCMC applications) and update the profile array.
"""
addwgds!(model::DLWGD, p::PArray, config::String) =
    addwgds!(model, p, eval(Meta.parse(wgds)))

function addwgds!(model::DLWGD, p::PArray, wgds::Dict)
    for (k,v) in wgds
        for wgd in v
            n, t = closestnode(model[k], wgd[1])
            addwgd!(model, n, t, wgd[2])
            length(p[1].x) == 0 ? nothing : extend!(p, n.i)
        end
    end
end

function addwgds!(p::PArray, wgds::Dict)
    for (k,v) in sort(wgds)
        length(p[1].x) == 0 ? nothing : extend!(p, v[1])
    end
end
