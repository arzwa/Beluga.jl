abstract type PhyloBDPModel{T} end

"""
    DLWGD{T<:Real,V<:ModelNode{T}}

Duplication, loss and WGD model. This holds a dictionary for easy access of the
nodes in the probabilistic graphical model and the leaf names.
"""
struct DLWGD{T<:Real,V<:ModelNode{T}} <: PhyloBDPModel{T}
    nodes ::Dict{Int64,V}
    leaves::Dict{Int64,Symbol}
end

Base.show(io::IO, d::DLWGD{T,V}) where {T,V} = write(io, "DLWGD{$T,$V}($(length(d)))")
Base.length(d::DLWGD) = length(d.nodes)
Base.getindex(d::DLWGD, i::Int64) = d.nodes[i]
Base.getindex(d::DLWGD, i::Int64, s::Symbol) = d.nodes[i][s]
ne(d::DLWGD) = 2*length(d.leaves) - 2  # number of edges ignoring WGDs

function DLWGD(nw::String, df::DataFrame, λ, μ, η, nt::Type=Branch)
    @unpack t, l = readnw(nw)
    M, m = profile(t, l, df)
    d = DLWGD(initmodel(t, l, η, λ, μ, max(3, m), nt)...)
    set!(d)
    (model=d, data=Profile(M))
end

function DLWGD(nw::String, λ, μ, η, nt::Type=Branch)
    # NOTE: model without data; m should be >= 3, otherwise WGD model breaks
    @unpack t, l = readnw(nw)
    d = DLWGD(initmodel(t, l, η, λ, μ, 3, nt)...)
    set!(d)
    (model=d, data=PArray())
end

# compute extended phylogenetic profile (i.e. upper bound of number of
# surviving lineages at internal nodes)
function profile(t::TreeNode, l::Dict, df::DataFrame)
    nodes = postwalk(t)
    M = zeros(Int64, size(df)[1], length(nodes))
    for n in nodes
        M[:,n.i] = isleaf(n) ?
            df[:,Symbol(l[n.i])] : sum([M[:,c.i] for c in n.c])
    end
    return permutedims(M), maximum(M)+1
end

"""
    set!(d::DLWGD)

Compute all model internals in postorder.
"""
function set!(d::DLWGD)
    for n in postwalk(d[1])
        set!(n)
    end
end

"""
    setrates!(model::DLWGD{T}, X::Matrix{T})

Set duplication and loss rates for each non-wgd node|branch in the model.
Rates should be provided as a 2 × n matrix, where the columns correspond to
model node indices.
"""
function setrates!(model::DLWGD{T}, X::Matrix{T}) where T
    for (i, n) in sort(model.nodes)
        if isawgd(n) ; break ; end
        n[:λ] = X[1, n.i]
        n[:μ] = X[2, n.i]
    end
    set!(model)
end

"""
    getrates(model::DLWGD{T})

Get the duplication and loss rate matrix (2 × n).
"""
function getrates(model::DLWGD{T}) where T
    X = zeros(T, 2,ne(model)+1)
    for (i, n) in sort(model.nodes)
        if isawgd(n) ; break ; end
        X[1, n.i] = n[:λ]
        X[2, n.i] = n[:μ]
    end
    X
end

function getqs(model::DLWGD{T}) where T
    X = zeros(T, nwgd(model))
    for (j, i) in enumerate(getwgds(model))
        X[j] = model[i, :q]
    end
    X
end

"""
    addwgd!(d::DLWGD, n::ModelNode, t, q)

Insert a WGD node with retention rate `q` at distance `t` above node `n`.
"""
function addwgd!(d::DLWGD{T}, n::ModelNode{T}, t::T, q::T) where T<:Real
    @assert !isroot(n) "Cannot add WGD above root node"
    @assert n[:t] - t > 0. "Invalid WGD time $(n[:t]) - $t < 0."
    parent = n.p
    i = maximum(keys(d.nodes))+1
    w = wgdnode(i, parent, q, n[:t] - t)
    a = wgdafternode(i+1, w)
    addwgd!(d, n, w, a)
end

function addwgt!(d::DLWGD{T}, n::ModelNode{T}, t::T, q::T) where T<:Real
    @assert !isroot(n) "Cannot add WGT above root node"
    @assert n[:t] - t > 0. "Invalid WGT time $(n[:t]) - $t < 0."
    parent = n.p
    i = maximum(keys(d.nodes))+1
    w = wgtnode(i, parent, q, n[:t] - t)
    a = wgdafternode(i+1, w)
    addwgd!(d, n, w, a)
end

function addwgd!(d::DLWGD{T}, n::N, w::N, a::N) where {T,N<:ModelNode{T}}
    # NOTE: assumes `w` and `a` have their parents already but not children
    n.p.c = setdiff(n.p.c, Set([n])) # remove n from its parent's children
    push!(n.p, w)
    push!(w, a)
    push!(a, n)
    n.p = a
    n[:t] -= w[:t]
    d.nodes[w.i] = w
    d.nodes[a.i] = a
    setabove!(nonwgdchild(n))
    return w
end

addwgds!(model::DLWGD, wgds::String) =
    addwgds!(model, eval(Meta.parse(wgds)))

function addwgds!(model::DLWGD, wgds::Dict)
    for (k,v) in wgds
        for wgd in v
            addwgd!(model, closestnode(model[k], wgd[1])..., wgd[2])
        end
    end
end

"""
    removewgd!(d::DLWGD, n::ModelNode, reindex::Bool=true, set::Bool=true)

Remove WGD/T node `n` from the DLWGD model. If `reindex` is true, the model
nodes are reindexed to be consecutive. If `set` is true, the model internals
(transition and extinction probabilities) are recomputed.
"""
function removewgd!(d::DLWGD, n::ModelNode, reindex::Bool=true, set::Bool=true)
    @assert iswgm(n) "Not a WGD/T node $i"
    @assert haskey(d.nodes, n.i) "Node not in model!"
    parent = n.p
    child = first(first(n))
    delete!(first(n), child)
    delete!(n, first(n))
    parent.c = setdiff(parent.c, Set([n]))
    # delete!(parent, n)  # issues! probably because n has changed by now?
    push!(parent, child)
    child.p = parent
    child[:t] += n[:t]
    delete!(d.nodes, n.i)
    delete!(d.nodes, n.i+1)
    reindex ? reindex!(d, n.i+2) : nothing
    set ? setabove!(nonwgdchild(child)) : nothing
    return child
end

function reindex!(d::DLWGD, i::Int64)
    for j=i:maximum(keys(d.nodes))
        d.nodes[j-2] = d.nodes[j]
        d[j-2].i = j-2
        delete!(d.nodes, j)
    end
end

"""
    removewgds(d::DLWGD)

Remove all WGD nodes from the model.
"""
function removewgds!(model::DLWGD)
    for i in getwgds(model)
        removewgd!(model, model[i], true, false)
    end
    set!(model)
end

function randpos(model::DLWGD)
    l = length(model)
    v = zeros(l)
    for (i, n) in model.nodes
        v[i] = n[:t]
    end
    i = sample(1:l, Weights(v))
    t = rand()*model[i][:t]
    model[i], t
end

nwgd(model::DLWGD) = length(getwgds(model))
randwgd(model::DLWGD) = model[rand(getwgds(model))]

function getwgds(model)
    wgds = Int64[]; i = maximum(keys(model.nodes))
    while isawgd(model[i])
        iswgm(model[i]) ? push!(wgds, i) : nothing
        i -= 1
    end
    wgds
end

function initmodel(t::TreeNode, l::Dict,
        η::T, λ::T, μ::T, m::Int64, nt::Type) where T<:Real
    # recursively construct the model by copying a reference tree
    d = Dict(1=>rootnode(η, λ, μ, m, nt))
    n = Dict{Int64,Symbol}()
    function walk(x, y)
        if isroot(x)
            x_ = d[1]
        else
            d[x.i] = x_ = spnode(x.i, y, λ, μ, x.x)
            push!(y, x_)
        end
        isleaf(x) ? n[x.i] = Symbol(l[x.i]) : [walk(c, x_) for c in x.c]
        return x_
    end
    walk(t, nothing)
    return d, n
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

function scattermat_iid(d::DLWGD)
    N = ne(d); M = 2
    Y = zeros(N, M)
    A = zeros(M, M)
    X0 = log.(d[1][:λ, :μ])
    for (i, n) in d.nodes
        if isawgd(n) || isroot(n)
            continue
        else
            Y[i-1,:] = log.(n[:λ, :μ]) - X0
            A += Y[i-1,:]*Y[i-1,:]'
        end
    end
    (A=A, Y=Y, q=M+1, n=N)
end

function setparams!(d::DLWGD, row::DataFrameRow)
    for (i,n) in d.nodes
        for (k,v) in n.x.θ
            s = id(n, k)
            s in names(row) ? n[k] = row[s] : nothing
        end
    end
end

"""
    (m::DLWGD)(row::DataFrameRow)

Instantiate a model based on a row from a trace data frame. This returns a
modified copy of the input model.
"""
function (m::DLWGD)(row::DataFrameRow)
    model = deepcopy(m)
    removewgds!(model)
    setparams!(model, row)
    insertwgds!(model, row[:wgds])
    set!(model)
    model
end

# gives isues with Documenter https://github.com/JuliaDocs/Documenter.jl/issues/1192
# """
#     (model::DLWGD{T,V})(θ::Vector{Y})
#
# Instantiate a model from a vector representation of the model parameterization.
# This returns a modified copy of the input model. θ is expected to have the
# structure [λ1, …, λn, μ1, …, μn, q1, …, qk, η].
# """
function (model::DLWGD{T,V})(θ::Vector{Y}) where {T<:Real,V<:ModelNode{T},Y<:Real}
    n = ne(model)+1
    m = size(model[1].x.W)[1]
    nt = nodetype(model[1])
    λ = θ[1:n]
    μ = θ[n+1:2n]
    q = θ[2n+1:end-1]
    d = Dict(1=>rootnode(θ[end], λ[1], μ[1], m, nt))
    function recursivecopy(x, y)
        if isroot(x)
            x_ = d[1]
        elseif issp(x)
            d[x.i] = x_ = spnode(x.i, y, λ[x.i], μ[x.i], Y(x[:t]))
            push!(y, x_)
        elseif iswgm(x)
            j = x.i % n
            j -= (j - 1) ÷ 2
            d[x.i] = x_ = iswgd(x) ?
                wgdnode(x.i, y, q[j], Y(x[:t])) :
                wgtnode(x.i, y, q[j], Y(x[:t]))
            push!(y, x_)
        elseif iswgdafter(x)
            d[x.i] = x_ = wgdafternode(x.i, y)
            push!(y, x_)
        end
        if !isleaf(x)
            [recursivecopy(c, x_) for c in x.c]
        end
        return x_
    end
    recursivecopy(model[1], nothing)
    newmodel = DLWGD(d, model.leaves)
    set!(newmodel)
    newmodel
end

"""
    logpdf(d::DLWGD, x::Vector{Int64})

Compute the log likelihood under the DLWGD model for a single count vector `x`
ℓ(λ,μ,q,η|x).
"""
function logpdf(d::DLWGD, x::Vector{Int64})
    L = csuros_miklos(d[1], x)
    l = integrate_root(L[:,1], d[1])
    l -= condition_oib(d[1])  #XXX sometimes -Inf-(-Inf) = NaN
    isfinite(l) ? l : -Inf
end

"""
    logpdf!(L::Matrix, d::DLWGD, x::Vector{Int64})

Compute the log likelihood under the DLWGD model for a single count vector `x`
ℓ(λ,μ,q,η|x) and update the dynamic programming matrix (`L`).
"""
function logpdf!(L::Matrix{T}, d::DLWGD{T}, x::Vector{Int64}) where T<:Real
    for n in postwalk(d[1])
        csuros_miklos!(L, n, x)
    end
    l = integrate_root(L[:,1], d[1])
    l -= condition_oib(d[1])  # XXX sometimes -Inf-(-Inf) = NaN
    isfinite(l) ? l : -Inf
end

"""
    logpdf!(L::Matrix, n::ModelNode, x::Vector{Int64})

Compute the log likelihood under the DLWGD model for a single count vector `x`
ℓ(λ,μ,q,η|x) and update the dynamic programming matrix (`L`), only recomputing
the matrix above node `n`.
"""
function logpdf!(L::Matrix{T}, n::ModelNode{T}, x::Vector{Int64}) where T<:Real
    while !isroot(n)
        csuros_miklos!(L, n, x)
        n = n.p
    end
    csuros_miklos!(L, n, x)
    l = integrate_root(L[:,1], n)
    l -= condition_oib(n)
    isfinite(l) ? l : -Inf
end

# when only η has changed at the root, it is wasteful to use `csuros_miklos!`
function logpdfroot(L::Matrix{T}, n::ModelNode{T}) where T<:Real
    l = integrate_root(L[:,1], n)
    l -= condition_oib(n)
    isfinite(l) ? l : -Inf
end

function csuros_miklos(node::ModelNode{T}, x::Vector{Int64}) where T<:Real
    L = minfs(T, maximum(x)+1, length(x))
    for n in postwalk(node)
        csuros_miklos!(L, n, x)
    end
    L
end

function csuros_miklos!(L::Matrix{T},
        node::ModelNode{T}, x::Vector{Int64}, leaf=false) where T<:Real
    # NOTE: possible optimizations:
    #  - column-major based access B, W, (L ✓)
    #  - matrix operations instead of some loops ~ WGDgc
    @unpack W, ϵ = node.x
    mx = maximum(x)
    if isleaf(node) || leaf
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
                            @warn "Invalid extinction probability, set to 1" p
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

# this is a bit superfluous (or the above function is), but is convenient
# currently to have separately. Note that we always keep the L matrix unaffected
# by the root prior or conditioning choice.
function root_vector(L::Vector{T}, n::ModelNode{T}) where T<:Real
    η = n[:η]
    ϵ = log(gete(n, 2))
    p = zeros(length(L))
    for i in 2:length(L)
        f = (i-1)*log1mexp(ϵ) + log(η) + (i-2)*log(1. - η)
        f -= i*log1mexp(log(1. - η)+ϵ)
        p[i] = L[i] + f
    end
    return p
end

function condition_oib(n::ModelNode{T}) where T<:Real
    lη = log(n[:η])
    lr = [geometric_extinctionp(log(gete(c, 1)), lη) for c in n.c]
    if lr[1] > zero(T) || lr[2] > zero(T)
        @warn "Invalid probabilities at `condition_oib`, returning -Inf" lr
        return -Inf
    else
        return log1mexp(lr[1]) + log1mexp(lr[2])
    end
end


# gradient, seems to only work in NaN safe mode http://www.juliadiff.org/
# ForwardDiff.jl/stable/user/advanced/#Fixing-NaN/Inf-Issues-1
# speed previous implementation, plants2.nw, maximum entry 19 = 1.6 ms
"""
    gradient(d::DLWGD, x::Vector)

Compute the gradient of the log likelihood under the DLWGD model for a single
count vector `x`, ∇ℓ(λ,μ,q,η|x).

!!! warning
    Currently the gradient seems to only work in NaN safe mode [github
    issue](http://www.juliadiff.org/ForwardDiff.jl/stable/user/advanced/#Fixing-NaN/Inf-Issues-1)
"""
function gradient(d::DLWGD{T}, x::Vector{Int64}) where T<:Real
    v = asvector(d)
    f = (u) -> logpdf(d(u), x)
    g = ForwardDiff.gradient(f, v)
    return g::Vector{Float64}
end

"""
    asvector(d::DLWGD)

Get a paraeter vector for the DLWGD model, structured as [λ1, …, λn, μ1, …, μn,
q1, …, qk, η].
"""
function asvector(d::DLWGD)
    r = getrates(d)
    q = getqs(d)
    η = d[1, :η]
    vcat(r[1,:], r[2,:], q, η)
end
