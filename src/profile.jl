# struct to store the data, enabling recomputation etc. Not that this is a
# fairly tedious interface if partial recomputation is not needed.
abstract type AbstractProfile end

"""
    Profile{V<:Real}

Struct for a phylogenetic profile of a single family.
"""
mutable struct Profile{V<:Real} <: AbstractProfile
    x   ::Vector{Int64}
    xtmp::Vector{Int64}
    L   ::Matrix{V}
    Ltmp::Matrix{V}
end

Profile(T::Type, x::Vector{Int64}, n::Int64) =
    Profile{T}(x, x, minfs(T, n, maximum(x)+1), minfs(T, n, maximum(x)+1))

Profile() = distribute(
    Profile[Profile{Float64}(zeros(0), zeros(0,0), zeros(0,0))])

function Profile(df::DataFrame, tree::Arboreal)
    X = profile(tree, df)
    N, n = size(X)
    D = Profile[Profile(Float64, X[i,:], n) for i=1:N]
    distribute(D), maximum(X)
end

const PArray = DArray{Profile,1,Array{Profile,1}}


# likelihoods
# ===========
"""
    logpdf!(m::PhyloBDP, p::PArray, [node::Int64])

Accumulate the logpdf along a PArray (distributed profile array).
"""
function logpdf!(m::PhyloBDP, p::PArray,
        branches::Vector{Int64}=m.tree.order)
    if length(p[1].x) == 0.  # HACK
        return 0.
    end
    mapreduce((x)->logpdf!(x.Ltmp, m, x.xtmp, branches), +, p)
end

logpdf(m::DuplicationLossWGD{T,V}, p::PArray) where
        {T<:Real,V<:Arboreal} = mapreduce((x)->logpdf(m, x.x), +, p)


# bookkeeping
# ===========
set!(p::PArray) = ppeval(_set!, p)
rev!(p::PArray) = ppeval(_rev!, p)

function _set!(p)
    p[1].L = deepcopy(p[1].Ltmp)  # XXX this must be a deepcopy!!!
    p[1].x = deepcopy(p[1].xtmp)
    0
end

function _rev!(p)
    p[1].Ltmp = deepcopy(p[1].L)  # XXX this must be a deepcopy!!!
    p[1].xtmp = deepcopy(p[1].x)
    0
end
