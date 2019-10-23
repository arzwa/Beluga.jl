
# Nice thing this is unrelated to the implementation of the MixtureChain
# Profiles for MixtureModeling
# ============================
mutable struct MixtureProfile{V<:Real} <: AbstractProfile
    # this stores likelihood matrices for each cluster
    # may start to eat a lot of memory when many clusters are used...
    x::Vector{Int64}
    L::Array{Matrix{V},1}
    Ltmp::Array{Matrix{V},1}
    l::Vector{V}
    ltmp::Vector{V}
    ztrace::Vector{Int64}
end

MixtureProfile(T::Type, x::Vector{Int64}, n::Int64, K::Int64, m=maximum(x)) =
    MixtureProfile{T}(x,
        [minfs(T, n, m+1) for i=1:K],
        [minfs(T, n, m+1) for i=1:K],
        zeros(K), zeros(K), Int64[rand(1:K)])

MixtureProfile(K, N) = distribute(
    MixtureProfile[MixtureProfile(Float64, Int64[], 0, K, -1) for i=1:N])

function MixtureProfile(df::DataFrame, tree::Arboreal, K::Int64)
    X = profile(tree, df)
    N, n = size(X)
    D = MixtureProfile[MixtureProfile(Float64, X[i,:], n, K) for i=1:N]
    distribute(D), maximum(X)
end

const MPArray = DArray{MixtureProfile,1,Array{MixtureProfile,1}}

getcomponent(p::MPArray, z::Int64) = p[[i for i in 1:length(p) if p[i].z == z]]
assignments(p::MPArray) = [pi.z for pi in p]
StatsBase.countmap(p::MPArray) = countmap([x.z for x in p])
StatsBase.counts(p::MPArray, K::Int64) = counts([x.z for x in p], 1:K)

Base.getproperty(p::MixtureProfile, x::Symbol) =
    x == :z ? getfield(p, :ztrace)[end] : getfield(p, x)
Base.setproperty!(p::MixtureProfile, x::Symbol, v) =
    x == :z ? setfield!(p, :ztrace, [p.ztrace ; v]) : setfield!(p, x, v)

# logpdf functions
# ===============
# These are required for the MCMC algorithm for finite mixtures
# compute logpdf, assuming only component k changed
function logpdf!(d::PhyloBDP, p::MPArray, k::Int64, bs::Vector{Int64})
    if length(p[1].x) == 0. ; return 0.; end  # HACK
    mapreduce((x)-> x.z == k ? logpdf!(d, x, k, bs) : x.l[x.z], +, p)
end

# compute logpdf for all families for their assignments
function logpdf!(d::Array{<:PhyloBDP,1}, p::MPArray, bs::Vector{Int64})
    if length(p[1].x) == 0. ; return 0.; end  # HACK
    mapreduce((x)-> logpdf!(d[x.z], x, x.z, bs), +, p)
end

# compute logpdfs for cluster k for all families not assigned to k
function logpdf_allother!(d::PhyloBDP, p::MPArray, k::Int64, bs::Vector{Int64})
    if length(p[1].x) == 0. ; return 0.; end  # HACK
    mapreduce((x)-> x.z == k ? x.l[k] : logpdf!(d, x, k, bs), +, p)
end

function logpdf!(d::PhyloBDP, x::MixtureProfile, k::Int64, bs::Array{Int64})
    l = Distributions.logpdf!(x.Ltmp[k], d, x.x, bs)
    x.ltmp[k] = l
    l
end


# caching functions
# =================
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
