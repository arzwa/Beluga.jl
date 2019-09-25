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

Profile() = distribute(
    Profile[Profile{Float64}(zeros(0), zeros(0,0), zeros(0,0))])

function Profile(df::DataFrame, tree::Arboreal)
    X = profile(tree, df)
    D = Profile[]
    n = size(X)[2]
    for i=1:size(X)[1]
        push!(D, Profile{Float64}(X[i,:],
            minfs(Float64, n, maximum(X[i,:])+1),
            minfs(Float64, n, maximum(X[i,:])+1)))
    end
    distribute(D), maximum(X)
end

const PArray = DArray{Profile,1,Array{Profile,1}}

# TODO: will need a conversion function for handling Dual etc.
# problem: DLModel other Real type than Profiles

"""
    logpdf!(m::PhyloBDP, p::PArray, [node::Int64])

Accumulate the logpdf along a PArray (distributed profile array).
"""
function Distributions.logpdf!(m::PhyloBDP, p::PArray, node::Int64=-1)
    if length(p[1].x) == 0.  # HACK
        return 0.
    end
    branches = node == -1 ? m.tree.order : m.tree.pbranches[node]
    mapreduce((x)->logpdf!(x.Ltmp, m, x.x, branches), +, p)
end

Distributions.logpdf(m::DuplicationLossWGD{T,V}, p::PArray) where
        {T<:Real,V<:Arboreal} = mapreduce((x)->logpdf(m, x.x), +, p)


set_Ltmp!(p::PArray) = ppeval(_set_Ltmp!, p)
set_L!(p::PArray) = ppeval(_set_L!, p)

function _set_Ltmp!(p)
    p[1].Ltmp = deepcopy(p[1].L)  # XXX this must be a deepcopy!!!
    0.
end

function _set_L!(p)
    p[1].L = deepcopy(p[1].Ltmp)
    0.
end
