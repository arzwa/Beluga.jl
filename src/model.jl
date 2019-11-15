# NOTE:
# * generalize to incl
# * should evaluate numerical accurracy in some way
# * would prefer to have more tests

# @time set!(d)
#   0.000061 seconds (47 allocations: 4.188 KiB)
# type union in ModelNode does not cause drop in efficiency
# should perhaps develop a type hierarchy to replace the union, but small type
# unions are no real issue https://julialang.org/blog/2018/08/union-splitting

# BelugaNode/ModelNode
# ====================
# would we lose efficiency if we use θ::Dict{Symbol,Union{T,Vector{T}}}?
@with_kw mutable struct BelugaNode{T}
    θ::Dict{Symbol,T}          # parameter vector (λ, μ, q, η, [κ])
    ϵ::Vector{T} = ones(T, 2)  # extinction probabilities
    W::Matrix{T}               # transition probability matrix
    kind::Symbol               # fake node type, for manual dispatch
end

# branch instead of node rates
@with_kw mutable struct BelugaBranch{T}
    θ::Dict{Symbol,T}          # parameter vector (λ, μ, q, η, [κ])
    ϵ::Vector{T} = ones(T, 2)  # extinction probabilities
    W::Matrix{T}               # transition probability matrix
    kind::Symbol               # fake node type, for manual dispatch
end

const ModelNode{T} = Union{TreeNode{BelugaNode{T}},TreeNode{BelugaBranch{T}}} where T

Base.setindex!(n::ModelNode{T}, v::T, s::Symbol) where T = n.x.θ[s] = v
Base.getindex(n::ModelNode, s::Symbol) = n.x.θ[s]
Base.getindex(n::ModelNode, args::Symbol...) = [n.x.θ[s] for s in args]
Base.eltype(n::TreeNode{T}) where T = T

iswgd(n::ModelNode) = n.x.kind == :wgd
iswgdafter(n::ModelNode) = n.x.kind == :wgdafter
isawgd(n::ModelNode) = iswgd(n) || iswgdafter(n)
issp(n::ModelNode) = n.x.kind == :sp
gete(n::ModelNode, i::Int64) = n.x.ϵ[i]
getw(n::ModelNode, i::Int64, j::Int64) = n.x.W[i,j]

function update!(n::TreeNode{BelugaNode{T}}) where T
    setbelow!(n)
    setabove!(n)
end

update!(n::TreeNode{BelugaBranch{T}}) where T = setabove!(n)

function update!(n::ModelNode{T}, θ::Symbol, v::T) where T
    n[θ] = v
    update!(n)
end

function update!(n::ModelNode, θ::NamedTuple)
    for (k, v) in pairs(θ)
        n[k] = v
    end
    update!(n)
end

# Different node kinds (maybe we could use traits?)
rootnode(η::T, λ::T, μ::T, m::Int64, nt::Type) where T<:Real =
    TreeNode(1, nt{T}(
        θ=Dict(:t=>0.,:λ=>λ,:μ=>μ,:η=>η),
        W=zeros(m,m),
        kind=:root))

speciationnode(i::Int64, p::V, λ::T, μ::T, t::T) where {T<:Real,V<:ModelNode{T}} =
    TreeNode(i, eltype(p)(
        θ=Dict(:t=>t,:λ=>λ,:μ=>μ),
        W=zeros(size(p.x.W)),
        kind=:sp), p)

# alternatively we could hack WGDs by adding dimensions in ϵ and W
wgdnode(i::Int64, p::V, q::T, t::T) where {T<:Real,V<:ModelNode{T}} =
    TreeNode(i, eltype(p)(
        θ=Dict(:t=>t,:q=>q),
        W=zeros(size(p.x.W)),
        kind=:wgd), p)

wgdafternode(i::Int64, p::V) where {T<:Real,V<:ModelNode{T}} =
    TreeNode(i, eltype(p)(
        θ=Dict(:t=>0.),
        W=zeros(size(p.x.W)),
        kind=:wgdafter), p)

# assumes n is a wgdnode, returns n if not a wgd node
function nonwgdparent(n::ModelNode)
    while isawgd(n) ; n = n.p ; end; n
end

function nonwgdchild(n::ModelNode)
    while isawgd(n); n = first(n.c); end; n
end


# Model
# =====
struct DuplicationLossWGDModel{T<:Real}
    nodes ::Dict{Int64,<:ModelNode{T}}
    leaves::Dict{Int64,Symbol}
end

const DLWGD{T} = DuplicationLossWGDModel{T}

Base.show(io::IO, d::DLWGD{T}) where T = write(io, "DLWGD{$T}($(length(d)))")
Base.length(d::DLWGD) = length(d.nodes)
Base.getindex(d::DLWGD, i::Int64) = d.nodes[i]
Base.getindex(d::DLWGD, i::Int64, s::Symbol) = d.nodes[i][s]
ne(d::DLWGD) = 2*length(d.leaves) - 2  # number of edges ignoring WGDs

function DuplicationLossWGDModel(nw::String, df::DataFrame, λ, μ, η,
        nt::Type=BelugaNode)
    @unpack t, l = readnw(nw)
    M, m = profile(t, l, df)
    d = DuplicationLossWGDModel(inittree(t, l, η, λ, μ, max(3,m), nt)...)
    set!(d)
    d, M
end

function DuplicationLossWGDModel(nw::String, λ, μ, η, nt::Type=BelugaNode)
    @unpack t, l = readnw(nw)
    d = DuplicationLossWGDModel(inittree(t, l, η, λ, μ, 3, nt)...)
    # NOTE: m should be >= 3, otherwise WGD model breaks
    set!(d)
    d
end

function setrates!(model::DLWGD{T}, x::Matrix{T}) where T
    for (i, n) in sort(model.nodes)
        if isawgd(n) ; break ; end
        n[:λ] = x[1, n.i]
        n[:μ] = x[2, n.i]
    end
    set!(model)
end

function logpdf(d::DLWGD, x::Vector{Int64})
    L = csuros_miklos(x, d[1])
    l = integrate_root(L[:,1], d[1])
    l -= condition_oib(d[1])  #XXX sometimes -Inf-(-Inf) = NaN
    isnan(l) ? -Inf : l
end

function logpdf!(L::Matrix{T}, x::Vector{Int64}, d::DLWGD{T}) where T<:Real
    for n in postwalk(d[1])
        csuros_miklos!(L, x, n)
    end
    l = integrate_root(L[:,1], d[1])
    l -= condition_oib(d[1])  #XXX sometimes -Inf-(-Inf) = NaN
    isnan(l) ? -Inf : l
end

# recompute from `n` upward
function logpdf!(L::Matrix{T}, x::Vector{Int64}, n::ModelNode{T}) where T<:Real
    while !isroot(n)
        csuros_miklos!(L, x, n)
        n = n.p
    end
    csuros_miklos!(L, x, n)
    l = integrate_root(L[:,1], n)
    l -= condition_oib(n)
    isnan(l) ? -Inf : l
end

# when only η has changed at the root, it is wasteful to do `csuros_miklos!`
function logpdfroot(L::Matrix{T}, n::ModelNode{T}) where T<:Real
    l = integrate_root(L[:,1], n)
    l -= condition_oib(n)
    isnan(l) ? -Inf : l
end

function profile(t::TreeNode, l::Dict, df::DataFrame)
    nodes = postwalk(t)
    M = zeros(Int64, size(df)[1], length(nodes))
    for n in nodes
        M[:,n.i] = isleaf(n) ? df[:,Symbol(l[n.i])] : sum([M[:,c.i] for c in n.c])
    end
    return permutedims(M), maximum(M)+1
end

function insertwgd!(d::DLWGD{T}, n::ModelNode{T}, t::T, q::T) where T<:Real
    if isroot(n)
        throw(ArgumentError("Cannot add WGD above root"))
    end
    parent = n.p
    n[:t] - t < 0. ? throw(DomainError("$(n[:t]) - $t < 0.")) : nothing
    i = maximum(keys(d.nodes))+1
    w = wgdnode(i, parent, q, n[:t] - t)
    a = wgdafternode(i+1, w)
    insertwgd!(d, n, w, a)
end

function insertwgd!(d::DLWGD{T}, n::ModelNode{T},
        w::ModelNode{T}, a::ModelNode{T}) where T<:Real
    # NOTE: this function assumes `w` and `a` have their parents already
    # but not children; this is not very elegant
    n.p.c = setdiff(n.p.c, Set([n])) # ??
    push!(n.p, w)
    push!(w, a)
    push!(a, n)
    n.p = a
    n[:t] -= w[:t]
    d.nodes[w.i] = w
    d.nodes[a.i] = a
    setabove!(n)
    return w
end

function removewgd!(d::DLWGD, n::ModelNode, reindex::Bool=true)
    if !iswgd(n)
        throw(ArgumentError("Not a WGD node $i"))
    end
    parent = n.p
    child = first(first(n))
    delete!(first(n), child)
    delete!(n, first(n))
    delete!(parent, n)
    push!(parent, child)
    child.p = parent
    child[:t] += n[:t]
    delete!(d.nodes, n.i)
    delete!(d.nodes, n.i+1)
    reindex ? reindex!(d, n.i+2) : nothing
    setabove!(child)
    return child
end

function reindex!(d::DLWGD, i::Int64)
    for j=i:maximum(keys(d.nodes))
        d.nodes[j-2] = d.nodes[j]
        d[j-2].i = j-2
        delete!(d.nodes, j)
    end
end

function inittree(t::TreeNode, l, η::T, λ::T, μ::T,
        m::Int64, nt::Type) where T<:Real
    d = Dict(1=>rootnode(η, λ, μ, m, nt))
    n = Dict{Int64,Symbol}()
    function walk(x, y)
        if isroot(x)
            x_ = d[1]
        else
            d[x.i] = x_ = speciationnode(x.i, y, λ, μ, x.x)
            push!(y, x_)
        end
        isleaf(x) ? n[x.i] = Symbol(l[x.i]) : [walk(c, x_) for c in x.c]
        return x_
    end
    walk(t, nothing)
    return d, n
end

function set!(d::DLWGD)
    for n in postwalk(d[1])
        set!(n)
    end
end

function set!(n::ModelNode)
    setϵ!(n)
    setW!(n)
end

function setabove!(n::ModelNode)
    while !isnothing(n)
        set!(n)
        n = n.p
    end
end

function setbelow!(n::ModelNode)
    for c in n.c
        set!(c)
    end
end

function setϵ!(n::ModelNode{T}) where T<:Real
    if isleaf(n)
        n.x.ϵ[2] = 0.
    elseif iswgd(n)
        q = n[:q]
        c = first(n.c)
        λ, μ = getλμ(n, c)
        c.x.ϵ[1] = ϵc = gete(c, 2)
        n.x.ϵ[2] = c.x.ϵ[1] = q*ϵc^2 + (1. - q)*ϵc
    else
        n.x.ϵ[2] = one(T)
        for c in n.c
            c.x.ϵ[1] = one(T)
            λ, μ = getλμ(n, c)
            c.x.ϵ[1] = approx1(ep(λ, μ, c[:t], gete(c, 2)))
            n.x.ϵ[2] *= c.x.ϵ[1]
        end
        n.x.ϵ[2] = approx1(n.x.ϵ[2])
    end
end

function setW!(n::ModelNode{T}) where T<:Real
    if isroot(n)
        return
    end
    λ, μ = getλμ(n.p, n)
    ϵ = gete(n, 2)
    if iswgdafter(n)
        q = n.p[:q]
        wstar_wgd!(n.x.W, n[:t], λ, μ, q, ϵ)
    else
        wstar!(n.x.W, n[:t], λ, μ, ϵ)
    end
end

function getλμ(a::TreeNode{BelugaNode{T}}, b::TreeNode{BelugaNode{T}}) where T
    a = isawgd(a) ? nonwgdparent(a) : a
    b = isawgd(b) ? nonwgdchild(b)  : b
    # 0.5*(a[:λ, :μ] .+ b[:λ, :μ]) below is faster
    [(a[:λ]+b[:λ])/2., (a[:μ]+b[:μ])/2.]
end

function getλμ(a::TreeNode{BelugaBranch{T}}, b::TreeNode{BelugaBranch{T}}) where T
    b = isawgd(b) ? nonwgdchild(b)  : b
    [b[:λ], b[:μ]]
end

function branchrates(d::DLWGD)
    r = zeros(2, ne(d)+1)
    r[:,1] = d[1][:λ, :μ]
    for (i,n) in d.nodes
        if issp(n)
            r[:, i] = getλμ(n.p, n)
        end
    end
    r
end

function wstar!(w, t, λ, μ, ϵ)
    # compute w* (Csuros Miklos 2009)
    ϕ = getϕ(t, λ, μ)  # p
    ψ = getψ(t, λ, μ)  # q
    _n = 1. - ψ*ϵ
    ϕp = approx1((ϕ*(1. - ϵ) + (1. - ψ)*ϵ) / _n)
    ψp = approx1(ψ*(1. - ϵ) / _n)
    w[1,1] = 1.
    for m=1:size(w)[1]-1, n=1:m
        w[n+1, m+1] = ψp*w[n+1, m] + (1. - ϕp)*(1. - ψp)*w[n, m]
    end
end

function wstar_wgd!(w, t, λ, μ, q, ϵ)
    # compute w* (Csuros Miklos 2009)
    w[1,1] = 1.
    w[2,2] = ((1. - q) + 2q*ϵ)*(1. - ϵ)
    w[2,3] = q*(1. - ϵ)^2
    mmax = size(w)[1]-1
    for i=1:mmax, j=2:mmax
        w[i+1, j+1] =  w[2,2]*w[i, j] + w[2,3]*w[i, j-1]
    end
end

function csuros_miklos(x, node::ModelNode{T}) where T<:Real
    L = minfs(T, maximum(x)+1, length(x))
    for n in postwalk(node)
        csuros_miklos!(L, x, n)
    end
    L
end

function csuros_miklos!(L::Matrix{T}, x::Vector{Int64},
        node::ModelNode{T}) where T<:Real
    # NOTE: possible optimizations:
    #  - column-major based access B, W, (L ✓)
    #  - matrix operations instead of some loops ~ WGDgc
    @unpack W, ϵ = node.x
    mx = maximum(x)
    if isleaf(node)
        L[x[node.i]+1, node.i] = 0.
    else
        children = [c for c in node.c]
        Mc = [x[c.i] for c in children]
        _M = cumsum([0 ; Mc])
        # @show Mc, _M
        _ϵ = cumprod([1.; [gete(c, 1) for c in node.c]])
        if any(_ϵ .> 1.)  # FIXME:
            _ϵ[_ϵ .> 1.] .= 1.
        end
        B = minfs(eltype(_ϵ), length(Mc), _M[end]+1, mx+1)
        A = minfs(eltype(_ϵ), length(Mc), _M[end]+1)
        for i = 1:length(Mc)
            c = children[i]
            Mi = Mc[i]
            Wc = c.x.W[1:mx+1, 1:mx+1]
            B[i, 1, :] = log.(Wc * exp.(L[1:mx+1, c.i]))

            for t=1:_M[i], s=0:Mi  # this is 0...M[i-1] & 0...Mi
                B[i,t+1,s+1] = s == Mi ? B[i,t,s+1] + log(gete(c, 1)) :
                    logaddexp(B[i,t,s+2], log(gete(c, 1))+B[i,t,s+1])
            end
            if i == 1
                for n=0:_M[i+1]  # this is 0 ... M[i]
                    A[i,n+1] = B[i,1,n+1] - n*log(1. - _ϵ[2])
                end
            else
                # XXX is this loop as efficient as it could?
                for n=0:_M[i+1], t=0:_M[i]
                    s = n-t
                    if s < 0 || s > Mi
                        continue
                    else
                        p = _ϵ[i]
                        if !(zero(p) <= p <= one(p))
                            @error "Invalid extinction probability ($p)"
                            p = one(p)
                        end
                        lp = logpdf(Binomial(n, p), s) +
                            A[i-1,t+1] + B[i,t+1,s+1]
                        A[i,n+1] = logaddexp(A[i,n+1], lp)
                    end
                end
                for n=0:_M[i+1]  # this is 0 ... M[i]
                    A[i,n+1] -= n*log(1. - _ϵ[i+1])
                end
            end
        end
        for n=0:x[node.i]
            # @show node.i, x[node.i], A
            L[n+1, node.i] = A[end, n+1]
        end
    end
end

function integrate_root(L::Vector{T}, n::ModelNode{T}) where T<:Real
    η = n[:η]
    ϵ = log(gete(n, 2))
    p = -Inf
    for i in 2:length(L)
        f = (i-1)*log1mexp(ϵ) + log(η) + (i-2)*log(1. - η)
        f -= i*log1mexp(log(1. - η)+ϵ)
        p = logaddexp(p, L[i] + f)
    end
    return p
end

function condition_oib(n::ModelNode{T}) where T<:Real
    lη = log(n[:η])
    lr = [geometric_extinctionp(log(gete(c, 1)), lη) for c in n.c]
    if lr[1] > zero(T) || lr[2] > zero(T)
        @warn "Invalid probabilities at `condition_oib`, returning -Inf"
        return -Inf
    else
        return log1mexp(lr[1]) + log1mexp(lr[2])
    end
end


# Instantiate a model from a data frame row, assumes a base model to copy
# **without** WGDs! (HACK: not elegant)
function (m::DuplicationLossWGDModel)(row::DataFrameRow)
    model = deepcopy(m)
    for i in getwgds(model)
        @show i
        removewgd!(model, model[i])
    end
    for (i,n) in model.nodes
        for (k,v) in n.x.θ
            s = id(n, k)
            s in names(row) ? n[k] = row[s] : nothing
        end
        update!(n)
    end
    for (k,v) in row[:wgds]
        for wgd in v
            insertwgd!(model, closestnode(model[k], wgd[2])..., wgd[1])
        end
    end
    model
end

function closestnode(n, t)
    t_ = t - n[:t]
    while t_ > 0
        t = t_
        n = n.p
        t_ -= n[:t]
    end
    n, t
end
