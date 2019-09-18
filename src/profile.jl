# struct to store the data, enabling recomputation etc. Not that this is a
# fairly tedious interface if partial recomputation is not needed.
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
    D = Profile[]
    n = size(X)[2]
    for i=1:size(X)[1]
        push!(D, Profile{Float64}(X[i,:],
            zeros(n, maximum(X[i,:])+1),
            zeros(n, maximum(X[i,:])+1)))
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

set_Ltmp!(p::PArray) = ppeval(_set_Ltmp!, p)
set_L!(p::PArray) = ppeval(_set_L!, p)

function _set_Ltmp!(p)
    p[1].Ltmp = p[1].L
    0.
end

function _set_L!(p)
    p[1].L = p[1].Ltmp
    0.
end
