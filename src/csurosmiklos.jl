
function csuros_miklos!(L::Matrix{T},
        x::AbstractVector{Int64},
        matrices::CsurosMiklos{T},
        tree::Arboreal,
        branches::Array{Int64}) where T<:Real
    @unpack W, ϵ, m = matrices
    mx = size(L)[2]-1

    for e in branches
        if isleaf(d, e)
            L[e, M[e]+1] = 1.0
        else
            children = childnodes(tree, e)
            Mc = [M[c] for c in children]
            _M = cumsum([0 ; Mc])
            _ϵ = cumprod([1.; [ϵ[c, 1] for c in children]])
            B = zeros(eltype(_ϵ), length(children), _M[end]+1, mx+1)
            A = zeros(eltype(_ϵ), length(children), _M[end]+1)
            for i = 1:length(children)
                c = children[i]
                Mi = Mc[i]
                B[i, 1, :] = W[1:mx+1, 1:mx+1, c] * L[c, :]
                for t=1:_M[i], s=0:Mi  # this is 0...M[i-1] & 0...Mi
                    if s == Mi
                        B[i,t+1,s+1] = B[i,t,s+1] * ϵ[c,1]
                    else
                        B[i,t+1,s+1] = B[i,t,s+2] + ϵ[c,1]*B[i,t,s+1]
                    end
                end
                if i == 1
                    for n=0:_M[i+1]  # this is 0 ... M[i]
                        A[i,n+1] = B[i,1,n+1]/(1. - _ϵ[2])^n
                    end
                else
                    # XXX is this loop as efficient as it could?
                    for n=0:_M[i+1], t=0:_M[i]
                        s = n-t
                        p = _ϵ[i]
                        p = isapprox(p, one(p)) ? one(p) : p  # had some problem
                        s < 0 || s > Mi ? continue : A[i,n+1] += pdf(Binomial(
                            n, p), s) * A[i-1,t+1] * B[i,t+1,s+1]
                    end
                    for n=0:_M[i+1]  # this is 0 ... M[i]
                        A[i,n+1] /= (1. - _ϵ[i+1])^n
                    end
                end
            end
            for n=0:M[e]
                L[e, n+1] = A[end, n+1]
            end
        end
    end
    log.(L)
end


"""
    csuros_miklos(d::PhyloLinearBDP, M::AbstractVector, W::Array{<:Real,3})

Csuros & Miklos (2009) algorithm for computing the likelihood of a profile
conditional on the number of surviving lineages. Verified against WGDgc (Cecile
Ane 2013). This should work for all linear phylogenetic BDPs (i.e DL, DL+WGD,
DL+gain, DL+WGD+gain), provided the W matrix is computed correctly for the
relevant model.
"""
function csuros_miklos(d::PhyloLinearBDP, M::AbstractVector, W::Array{<:Real,3})
    mx = maximum(M)
    L = zeros(eltype(d.ϵ), length(d.tree), mx+1)
    if any((x)->x<0., d.ϵ)
        L .= NaN
        return L
    end
    # Here partial recomputation could be possible when only some rates changed
    for e in d.tree.order
        if isleaf(d, e)
            L[e, M[e]+1] = 1.0
        else
            children = childnodes(d, e)
            Mc = [M[c] for c in children]
            _M = cumsum([0 ; Mc])
            _ϵ = cumprod([1.; [d[c, 1] for c in children]])
            B = zeros(eltype(_ϵ), length(children), _M[end]+1, mx+1)
            A = zeros(eltype(_ϵ), length(children), _M[end]+1)
            for i = 1:length(children)
                c = children[i]
                Mi = Mc[i]
                B[i, 1, :] = W[1:mx+1, 1:mx+1, c] * L[c, :]
                for t=1:_M[i], s=0:Mi  # this is 0...M[i-1] & 0...Mi
                    if s == Mi
                        B[i,t+1,s+1] = B[i,t,s+1] * d[c,1]
                    else
                        B[i,t+1,s+1] = B[i,t,s+2] + d[c,1]*B[i,t,s+1]
                    end
                end
                if i == 1
                    for n=0:_M[i+1]  # this is 0 ... M[i]
                        A[i,n+1] = B[i,1,n+1]/(1. - _ϵ[2])^n
                    end
                else
                    # XXX is this loop as efficient as it could?
                    for n=0:_M[i+1], t=0:_M[i]
                        s = n-t
                        p = _ϵ[i]
                        p = isapprox(p, one(p)) ? one(p) : p  # had some problem
                        s < 0 || s > Mi ? continue : A[i,n+1] += pdf(Binomial(
                            n, p), s) * A[i-1,t+1] * B[i,t+1,s+1]
                    end
                    for n=0:_M[i+1]  # this is 0 ... M[i]
                        A[i,n+1] /= (1. - _ϵ[i+1])^n
                    end
                end
            end
            for n=0:M[e]
                L[e, n+1] = A[end, n+1]
            end
        end
    end
    b = L .< 0
    if any(b)
        # this shouldn't occur...
        @warn "Negative values in 'csuros_miklos' ⇒ set to zero()"
        L[b] .= zero(L[1])
    end
    log.(L)
end
