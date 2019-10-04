
# Profiles for MixtureModeling
# ============================
mutable struct MixtureProfile{V<:Real} <: AbstractProfile
    # this stores likelihood matrices for each cluster
    # may start to eat a lot of memory when many clusters are used...
    x::Vector{Int64}
    L::Array{Matrix{V}}
    Ltmp::Array{Matrix{V}}
    l::Vector{V}
    ltmp::Vector{V}
    z::Int64
end

MixtureProfile(T::Type, x::Vector{Int64}, n::Int64, K::Int64, m=maximum(x)) =
    MixtureProfile{T}(x, minfs(T, n, m), minfs(T, n, m), zeros(K), 1)

function MixtureProfile(df::DataFrame, tree::Arboreal, K::Int64)
    X = profile(tree, df)
    N, n = size(X)
    D = MixtureProfile[MixtureProfile(Float64, X[i,:], n, K) for i=1:N]
    distribute(D), maximum(X)
end

const MPArray = DArray{MixtureProfile,1,Array{MixtureProfile,1}}

getcomponent(p::MPArray, z::Int64) = p[[i for i in 1:length(p) if p[i].z == i]]

# compute logpdf, assuming only component k changed
function Distributions.logpdf!(d::PhyloBDP, p::MPArray, i::Int64, k::Int64)
    if length(p[1].x) == 0.  # HACK
        return 0.
    end
    branches = i == -1 ? m.tree.order : m.tree.pbranches[i]
    mapreduce((x)-> x.z == k ? logpdf!(x, k, branches) : x.logp[x.z], +, p)
end

# compute logpdf for all families for their assignments
function Distributions.logpdf!(d::PhyloBDP, p::MPArray, i::Int64)
    if length(p[1].x) == 0.  # HACK
        return 0.
    end
    branches = i == -1 ? m.tree.order : m.tree.pbranches[i]
    mapreduce((x)-> logpdf!(x, x.z, branches), +, p)
end

# compute logpdfs for cluster k for all families not assigned to k
function logpdf_allother!(d::PhyloBDP, p::MPArray,
        i::Int64, k::Int64)
    if length(p[1].x) == 0.  # HACK
        return 0.
    end
    branches = i == -1 ? m.tree.order : m.tree.pbranches[i]
    mapreduce((x)-> x.z == k ? x.logp[k] : logpdf!(x, k, branches), +, p)
end


function logpdf!(x::MixtureProfile, k::Int64, branches::Array{Int64})
    l = logpdf!(x.Ltmp[k], m, x.x, branches)
    x.ltmp[k] = l
    l
end

set_Ltmp!(p::MPArray, k) = ppeval((x)->_set_Ltmp!(x, k), p)
set_L!(p::MPArray, k) = ppeval((x)->_set_L!(x, k), p)

function _set_Ltmp!(p, k)
    p[1].Ltmp[k] = deepcopy(p[1].L[k])  # XXX this must be a deepcopy!!!
    p[1].ltmp[k] = deepcopy(p[1].l[k])  # XXX this must be a deepcopy!!!
    0.
end

function _set_L!(p, k)
    p[1].L[k] = deepcopy(p[1].Ltmp[k])
    p[1].l[k] = deepcopy(p[1].ltmp[k])
    0.
end
