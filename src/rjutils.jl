# NOTE: everything is very ugly and will need a rewrite eventually
# finish this implementation; but maybe think about more elegant implementation
# perhaps using the DAG more abstractly or so? The DLModel could be fully
# represented as a DAG with speciation, WGD and root nodes associated with
# their rates

# at most two objects should change during the rj MCMC: the model (holding the
# tree, ϵ and W) and the PArray, with the DP matrices (add/remove dimensions)
# these methods should presumably work on copies

# add WGD node and modify profile array
# add wgd onm branch e in model m and profile array p at time t with ret. rate q
function Beluga.addwgd!(m, p, e, t, q)
    n, n_ = addwgd!(m.tree, e, t, nwgd(m.tree)+1)
    extend!(m.value)
    m.q = [m.q ; 0.]
    m[:q, n] = q      # this will reocompute the necessary stuff
    extend!(p, e, 2)
end

# remove WGD node and modify profile array
# remove wgd node n
# WGD node has index n, after node index n-1
# this is horrible
function removewgd!(m, p, n)
    # delete branches
    @unpack value = m
    @unpack tree = m.tree
    shrink!(p, n)  # modify profile
    shrink!(value, n)
    a = parentnode(tree, n)
    b = childnodes(tree, n-1)[1]
    t1 = parentdist(tree, n)
    t2 = parentdist(tree, b)
    out = tree.nodes[n].out
    for node in [n, n-1, b]
        deletebranch!(tree, tree.nodes[node].in[1])
    end
    for node in [n, n-1]
        deletenode!(tree, node)
    end
    # connect disconnected nodes
    addbranch!(tree, a, b, t2+t1)
    i = m.tree.bindex[n][:q]
    delete!(m.tree.bindex, n)
    delete!(m.tree.bindex, n-1)

    # remove node from bindex
    if i < length(m.q)
        for (k,v) in m.tree.bindex
            haskey(v, :q) && v[:q] > i ? v[:q] -= 1 : (continue)
        end
    end
    m.q = [m.q[1:i-1] ; m.q[i+1:end]]

    # shift node indices if necessary  # HACK ugly!
    if n <= length(m.tree)
        t, bi = reindex(m.tree)
        m.tree.tree = t
        m.tree.bindex = bi
    end

    m.tree.order = postorder(m.tree)
    Beluga.set_parentbranches!(m.tree)
    bs = m.tree.pbranches[b]
    Beluga.get_ϵ!(m, bs)
    Beluga.get_W!(m, bs)
end



# change WGD time
# tbefore is the time between WGD node n and its parentnode (which may be a WGD)
function shiftwgd!(m, n, tbefore)
    @unpack tree = m.tree
    child = childnodes(tree, n-1)[1]
    b1 = tree.branches[tree.nodes[n].in[1]]
    b2 = tree.branches[tree.nodes[child].in[1]]
    l = b1.length + b2.length
    b1.length = tbefore
    b2.length = l - tbefore
    bs = m.tree.pbranches[child]
    Beluga.get_ϵ!(m, bs)
    Beluga.get_W!(m, bs)
end


# Change models
function extend!(v::CsurosMiklos{T}) where T<:Real
    v.ϵ = [v.ϵ ; Beluga.minfs(T, 2, 2)]
    v.W = cat(v.W, Beluga.minfs(T, v.m+1, v.m+1, 2), dims=3)
end

function shrink!(v::CsurosMiklos{T}, i::Int64) where T<:Real
    v.ϵ = [v.ϵ[1:i-1,:] ; v.ϵ[i+2:end,:]]
    v.W = cat(v.W[:,:,1:i-1],v.W[:,:,i+2:end], dims=3)
end


# copy the tree shifting node indices
function reindex(s::SpeciesTree)
    @unpack tree, bindex = s
    T = Tree()
    index   = Dict()
    bindex_ = Dict{Int64,Dict{Symbol,Int64}}()
    i = 1
    for (n, node) in sort(tree.nodes)
        addnode!(T)
        index[n] = i
        bindex_[i] = bindex[n]
        i += 1
    end
    for (n, node) in sort(tree.nodes)
        for o in node.out
            n_ = tree.branches[o].target
            i = index[n]
            j = index[n_]
            addbranch!(T, i, j, parentdist(tree, n_))
        end
    end
    T, bindex_
end


# Change profiles
# to profile eventually
# only change Ltmp; L should be set using set_L, something similar for x?
extend!(p::AbstractArray{P}, e::Int64, dim::Int64) where P<:AbstractProfile =
    map((x)->extend!(x, e, dim), p)

function extend!(p::Profile{T}, e::Int64, dim::Int64) where T<:Real
    m = size(p.L)[2]
    p.xtmp = [p.x ; [p.x[e], p.x[e]]]
    p.Ltmp = [p.Ltmp ; Beluga.minfs(T, dim, m)]
end

# not implemented for general Mixtures!
function extend!(p::MixtureProfile{T}, e::Int64, dim::Int64) where T<:Real
    m = size(p.L[1])[2]
    p.x = [p.x ; [p.x[e], p.x[e]]]  # profile itself should be extended as well!
    p.Ltmp[1] = [p.Ltmp[1] ; Beluga.minfs(T, dim, m)]
end

shrink!(p::AbstractArray{P}, i) where P<:AbstractProfile =
    map((x)->shrink!(x, i), p)

# i is the WGD node
function shrink!(p::Profile{T}, i) where T<:Real
    p.x    = [p.x[1:i-2]      ; p.x[i+1:end]     ]
    p.Ltmp = [p.Ltmp[1:i-2,:] ; p.Ltmp[i+1:end,:]]
end

function shrink!(p::MixtureProfile{T}, i) where T<:Real
    p.x    = [p.x[1:i-2]      ; p.x[i+1:end]     ]
    p.Ltmp = [[p.Ltmp[1][1:i-2,:] ; p.Ltmp[1][i+1:end,:]]]
end
