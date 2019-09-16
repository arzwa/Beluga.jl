# struct to store the data, enabling recomputation etc.

# Should we store the likelihood matrices in a 3D array, or rather, like in
# Whale for instance, keep the 2D matrix separately for each family ?
#=struct ProfileMatrix{V<:Real}
    X::AbstractMatrix{Int64}
    L::AbstractArray{V,3}
end

function ProfileMatrix(df::DataFrame, tree::Arboreal)
    n = nprocs()
    X = distribute(profile(tree, df), dist=(n,1))
    L = distribute(zeros(size(X)..., maximum(X)), dist=(n,1,1))
    Profile{Float64}(X,L)
end=#


"""
    Profile{V<:Real}

Struct for a phylogenetic profile of a single family.
"""
mutable struct Profile{V<:Real}
    x::Vector{Int64}
    L::Matrix{V}
    Ltmp::Matrix{V}
end

function Profile(df::DataFrame, tree::Arboreal)
    X = profile(tree, df)
    D = Array{Profile,1}(undef, size(X)[1])
    n = size(X)[2]
    for i=1:size(X)[1]
        D[i] = Profile{Float64}(X[i,:],
            zeros(n, maximum(X[i,:])+1),
            zeros(n, maximum(X[i,:])+1))
    end
    distribute(D), maximum(X)
end

const PArray = DArray{Profile,1,Array{Profile,1}}

# TODO: will need a conversion function for handling Dual etc.

"""
    logpdf!(m::PhyloBDP, p::PArray, [node::Int64])

Accumulate the logpdf along a PArray (distributed profile array).
"""
function Distributions.logpdf!(m::PhyloBDP, p::PArray, node::Int64=-1)
    branches = node == -1 ? m.tree.order : get_parentbranches(m.tree, node)
    mapreduce((x)->logpdf!(x.Ltmp, m, x.x, branches), +, p)
end

set_tmp!(p::PArray) = map((x)->_set_tmp!(x), p)
set_L!(p::PArray) = map((x)->_set_L!(x), p)
_set_tmp!(p::Profile) = p.Ltmp = p.L
_set_L!(p::Profile) = p.L = p.Ltmp
