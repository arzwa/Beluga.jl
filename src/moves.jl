function move_node!(chain, n)
    @unpack data, state, model, props, prior = chain
    v = n[:λ, :μ]
    prop = rand(props[n.i])
    w, r = prop(log.(v))
    update!(n, (λ=exp(w[1]), μ=exp(w[2])))
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + r
    # @show l_, p_, exp.(w), v
    if log(rand()) < hr
        update!(state, n, :λ, :μ)
        state[:logp] = l_
        state[:logπ] = p_
        set!(data)
        prop.accepted += 1
    else
        update!(n, (λ=v[1], μ=v[2]))
        rev!(data)
    end
end

function move_root!(chain, n)
    @unpack data, state, model, props, prior = chain
    prop = props[0][1]
    η = n[:η]
    η_, r = prop(η)
    update!(n, :η, η_)
    l_ = logpdfroot(n, data)
    p_ = logpdf(prior, model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + r
    if log(rand()) < hr
        update!(state, n, :η)
        state[:logp] = l_
        state[:logπ] = p_
        set!(data)
        prop.accepted += 1
    else
        update!(n, :η, η)
        rev!(data)
    end
end

# ove for both retention rate and time of WGD
function move_wgdtime!(chain, n)
    @unpack data, state, model, props, prior = chain
    prop = props[n.i][1]
    child = first(first(n))
    q = n[:q]
    t1 = n[:t]
    t = t2 = child[:t]
    u = rand()
    r = 0.
    if u < 0.66  # move q
        q_, r = prop(q)
        n[:q] = q_  # update from child below
    elseif u > 0.33  # move t
        t = rand()*(t1 + t2)
        n[:t] = t1 + t2 - t  # update from child below
    end
    update!(child, :t, t)
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + r
    if log(rand()) < hr
        update!(state, n, :q)
        state[:logp] = l_
        state[:logπ] = p_
        set!(data)
        u < 0.66 ? prop.accepted += 1 : nothing
    else
        n[:q] = q
        n[:t] = t1
        update!(child, :t, t2)
        rev!(data)
    end
end

function move_wgdrates!(chain::RevJumpChain{T,CoevolRevJumpPrior}, n) where T
    @unpack data, state, model, props, prior = chain
    q = n[:q]
    parent = nonwgdparent(n)
    child = nonwgdchild(n)
    flank = rand([parent, child])
    rates = flank[:λ, :μ]
    v = [q; log.(rates)]
    prop = rand(props[n.i][2:end])
    w, r = prop(v)
    n[:q] = w[1]
    flank[:λ] = exp(w[2])
    flank[:μ] = exp(w[3])
    update!(child)
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + r
    if log(rand()) < hr
        update!(state, n, :q)
        update!(state, flank, :λ, :μ)
        state[:logp] = l_
        state[:logπ] = p_
        set!(data)
        prop.accepted += 1
    else
        n[:q] = q
        flank[:λ] = rates[1]
        flank[:μ] = rates[2]
        update!(child)
        rev!(data)
    end
end

function move_wgdrates!(chain::RevJumpChain{T,IidRevJumpPrior}, n) where T
    @unpack data, state, model, props, prior = chain
    q = n[:q]
    child = nonwgdchild(n)
    rates = child[:λ, :μ]
    v = [q; log.(rates)]
    prop = rand(props[n.i][2:end])
    w, r = prop(v)
    n[:q] = w[1]
    update!(child, (λ=exp(w[2]), μ=exp(w[3])))
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, model)
    hr = l_ + p_ - state[:logp] - state[:logπ] + r
    if log(rand()) < hr
        update!(state, n, :q)
        update!(state, child, :λ, :μ)
        state[:logp] = l_
        state[:logπ] = p_
        set!(data)
        prop.accepted += 1
    else
        n[:q] = q
        update!(child, (λ=rates[1], μ=rates[2]))
        rev!(data)
    end
end


# Reversible jump moves
# =====================
function _move_addwgd!(chain::RevJumpChain{T,IidRevJumpPrior}) where T
    # XXX unpack copies the model or something??
    @unpack data, state, model, props, prior = chain
    n, t = randpos(chain.model)
    wgdnode = insertwgd!(chain.model, n, t, rand(prior.πq)/10.)
    length(data[1].x) == 0 ? nothing : extend!(data, n.i)
    l_ = logpdf!(n, data)
    p_ = logpdf(prior, chain.model)
    # hr = state[:k] == 0 ? log(0.5) : 0.  # see Rabosky
    # if we use the move where a removal or addition is chosen with P=0.5, this
    # is not necessary as q(M0|M1) = q(M1|M0) = 0.5
    hr = l_ + p_ - state[:logp] - state[:logπ]
    if log(rand()) < hr
        state[:logp] = l_
        state[:logπ] = p_
        state[:k] += 1
        s = Symbol("k$(nonwgdchild(wgdnode).i)")
        state[s] = haskey(state, s) ? state[s] + 1 : 1
        update!(state, wgdnode, :q)
        set!(data)
        props[wgdnode.i] = [AdaptiveUnitProposal() ; WgdProposals()]
    else
        removewgd!(chain.model, wgdnode)
        rev!(data)
    end
end


function move_addwgd!(chain::RevJumpChain{T,IidRevJumpPrior}) where T
    # XXX unpack copies the model or something??
    @unpack data, state, model, props, prior = chain
    if logpdf(prior.πK, nwgd(chain.model)+1) == -Inf
        return
    end
    n, t  = randpos(chain.model)
    child = nonwgdchild(n)
    propq = chain.props[0][2]
    propλ = chain.props[0][3]
    λn    = child[:λ]
    q, _  = propq(0.)
    # q = rand(chain.prior.πq)  # sample from prior (doesn't work that well)
    θ, _  = propλ(log(λn))

    child[:λ] = exp(θ)
    wgdnode = insertwgd!(chain.model, n, t, q)
    length(data[1].x) == 0 ? nothing : extend!(data, n.i)
    l_ = logpdf!(child, data)
    p_ = logpdf(prior, chain.model)
    hr = l_ + p_ - state[:logp] - state[:logπ]

    if log(rand()) < hr
        state[:logp] = l_
        state[:logπ] = p_
        state[:k] += 1
        s = Symbol("k$(nonwgdchild(wgdnode).i)")
        state[s] = haskey(state, s) ? state[s] + 1 : 1
        update!(state, wgdnode, :q)
        update!(state, child, :λ)
        set!(data)
        props[wgdnode.i] = [AdaptiveUnitProposal() ; WgdProposals()]
        propλ.accepted += 1
        propq.accepted += 1
    else
        child[:λ] = λn
        removewgd!(chain.model, wgdnode)
        rev!(data)
    end
end


function move_rmwgd!(chain::RevJumpChain{T,IidRevJumpPrior}) where T
    @unpack data, state, model, props, prior = chain
    if nwgd(chain.model) == 0
        return
    end
    propλ = chain.props[0][3]
    wgdnode = randwgd(chain.model)
    wgdafter = first(wgdnode)
    n  = nonwgdchild(wgdnode)
    λn = n[:λ]
    θ, _  = propλ(log(λn))
    θ  = log(λn) - (θ - log(λn))  # HACK to reverse decrease proposal
    n[:λ] = exp(θ)
    child = removewgd!(chain.model, wgdnode, false)

    l_ = logpdf!(n, data)
    p_ = logpdf(prior, chain.model)
    hr = l_ + p_ - state[:logp] - state[:logπ]
    if log(rand()) < hr
        # upon acceptance; shrink and reindex
        length(data[1].x) == 0 ? nothing : shrink!(data, wgdnode.i)
        delete!(props, wgdnode.i)
        delete!(state, id(wgdnode, :q))
        reindex!(chain.model, wgdnode.i+2)
        reindex!(chain.props, wgdnode.i+2)
        set!(data)
        setstate!(state, chain.model)
        s = Symbol("k$(n.i)")
        update!(state, n, :λ)
        state[s] -= 1
        state[:logp] = l_
        state[:logπ] = p_
        propλ.accepted += 1
    else
        n[:λ] = λn
        insertwgd!(chain.model, child, wgdnode, wgdafter)
        rev!(data)
    end
end


function _move_rmwgd!(chain::RevJumpChain{T,IidRevJumpPrior}) where T
    @unpack data, state, model, props, prior = chain
    if nwgd(chain.model) == 0
        return
    end
    # _model = deepcopy(chain.model)  # TODO find solution
    wgdnode = randwgd(chain.model)
    wgdafter = first(wgdnode)
    child = removewgd!(chain.model, wgdnode, false)
    l_ = logpdf!(child, data)
    p_ = logpdf(prior, chain.model)
    hr = l_ + p_ - state[:logp] - state[:logπ]
    if log(rand()) < hr
        # upon acceptance; shrink and reindex
        length(data[1].x) == 0 ? nothing : shrink!(data, wgdnode.i)
        delete!(props, wgdnode.i)
        delete!(state, id(wgdnode, :q))
        reindex!(chain.model, wgdnode.i+2)
        reindex!(chain.props, wgdnode.i+2)
        set!(data)
        setstate!(state, chain.model)
        s = Symbol("k$(nonwgdchild(child).i)")
        state[s] -= 1
        state[:logp] = l_
        state[:logπ] = p_
    else
        # re-inserting instead of the deepcopy above is tricky, since we insert
        # a node at the end, which may be a different index; if we refactor
        # removewgd to not reindex, we could re-insert the original wgdnode
        # upon rejection, re-insert the original node
        insertwgd!(chain.model, child, wgdnode, wgdafter)
        # chain.model = _model
        rev!(data)
    end
end
