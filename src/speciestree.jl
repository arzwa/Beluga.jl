# Abstract PhyloCTMC
# ==================
abstract type PhyloCTMC <: DiscreteMultivariateDistribution end
abstract type PhyloBDP <: PhyloCTMC end
abstract type PhyloLinearBDP <: PhyloBDP end

PhyloTrees.isleaf(d::PhyloCTMC, n::Int64) = isleaf(d.tree, n)
PhyloTrees.childnodes(d::PhyloCTMC, n::Int64) = childnodes(d.tree, n)
PhyloTrees.parentdist(d::PhyloCTMC, n::Int64) = parentdist(d.tree, n)

const BranchIndex = Dict{Int64,Dict{Symbol,Int64}}
Base.getindex(d::BranchIndex, i::Int64, s::Symbol) = d[i][s]
Base.setindex!(d::BranchIndex, x::Int64, i::Int64, s::Symbol) = haskey(d, i) ?
    d[i][s] = x : d[i] = Dict(s=>x)


# SpeciesTree type
# ================
"""
    SpeciesTree(tree, leaves, bindex)

This maintains two important indices for inference purposes: (1) the bindex
which links branches to indices in parameter vectors; here the θ parameter is
probably mandatory; (2) pbranches, which links each branch to an array of
'parent branches' (which are not strictly parent branches in the topology).
This array for some branch `i` contains all branches that are affected by a
change in parameters on branch `i` (i.e. those that are not conditionally
independent of `θ` on branch `i`). Note that this is only for `θ` (i.e.
typically (λ, μ) in the birth-death process).
"""
mutable struct SpeciesTree <: Arboreal
    tree::Tree
    leaves::Dict{Int64,Symbol}
    bindex::BranchIndex
    pbranches::Dict{Int64,Array{Int64}}  # branches
    order::Array{Int64,1}
    wgds::Set{Int64}
end

function SpeciesTree(tree::Tree, leaves::Dict{Int64,T}) where
        T<:AbstractString
    s = SpeciesTree(tree,
            Dict(k=>Symbol(v) for (k,v) in leaves),
            defaultidx(tree),
            Dict{Int64,Array{Int64}}(),
            postorder(tree),
            Set{Int64}())
    set_parentbranches!(s)
    return s
end

SpeciesTree(tree::LabeledTree) = SpeciesTree(tree.tree, tree.leaves)
SpeciesTree(treefile::String) = SpeciesTree(readtree(treefile))

defaultidx(tree) = BranchIndex(i=>Dict(:θ=>i) for (i,n) in tree.nodes)

Base.getindex(Ψ::SpeciesTree, i::Int64, s::Symbol) = Ψ.bindex[i, s]

function Base.setindex!(Ψ::SpeciesTree, x::Int64, i::Int64,s::Symbol)
    Ψ.bindex[i,s] = x
    reindex!(Ψ)
    set_parentbranches!(Ψ)
end

Base.length(Ψ::SpeciesTree) = length(Ψ.tree.nodes)
Base.display(Ψ::SpeciesTree) = println("$(typeof(Ψ))($(length(Ψ)))")

leafname(Ψ::SpeciesTree, i::Int64) = Ψ.leaves[i]
leafname(d::PhyloLinearBDP, i::Int64) = leafname(d.tree, i)


# WGD stuff
# =========
# NB: wgd is assumed to be defined by having a q entry in the branch index!
iswgd(Ψ::SpeciesTree, i::Int64) = haskey(Ψ.bindex[i], :q)
iswgdafter(Ψ::SpeciesTree, i::Int64) = isroot(Ψ.tree, i) ?
    false : haskey(Ψ.bindex[parentnode(Ψ, i)], :q)
nwgd(Ψ::SpeciesTree) = length(wgdnodes(Ψ))
qparent(Ψ::SpeciesTree, i::Int64) = Ψ.bindex[parentnode(Ψ, i), :q]
wgdnodes(Ψ::SpeciesTree) = [k for (k,v) in Ψ.bindex if haskey(v, :q)]

function set_wgdrates!(Ψ::SpeciesTree)
    for n in Ψ.order
        if iswgd(Ψ, n) || iswgdafter(Ψ, n)
            Ψ[n, :θ] = Ψ[childnodes(Ψ, n)[1], :θ]
        end
    end
    reindex!(Ψ)
end

function set_constantrates!(Ψ::SpeciesTree, s=:θ)
    for k in keys(Ψ.bindex)
        Ψ[k, s] = 1
    end
end

function reindex!(Ψ::SpeciesTree, s=:θ)
    d = Dict{Int64,Int64}()
    i = 1
    for n in preorder(Ψ)
        !haskey(Ψ.bindex[n], s) ? continue : nothing
        if haskey(d, Ψ[n, s])
            Ψ.bindex[n][s]  = d[Ψ.bindex[n][s]]
        else
            d[Ψ.bindex[n][s]] = i
            Ψ.bindex[n][s] = i
            i += 1
        end
    end
end

hasconstantrates(Ψ::SpeciesTree, s=:θ) =
    all([Ψ.bindex[s, i]==1 for i=1:Ψ.order])

nrates(Ψ::SpeciesTree, s=:θ) = length(unique([v[:θ] for (k,v) in Ψ.bindex]))

function addwgd!(Ψ::SpeciesTree, lca::Vector, t::Number)
    i = nwgd(Ψ) + 1
    n = lca_node(Ψ.tree, Set([k for (k,v) in Ψ.leaves if v in lca]))
    while leafdist(Ψ, parentnode(Ψ, n)) < t
        n = parentnode(Ψ, n)
    end
    tn = leafdist(Ψ, n)
    tbefore = t - tn
    addwgd!(Ψ, n, tbefore, i)
end

# n is a branch id, which is the node at the end of a branch between to splits
# here t is the time before n where the WGD has to be inserted
function addwgd!(Ψ::SpeciesTree, n::Int64, tbefore, i)
    wgdafter = insert_node!(Ψ.tree, n, tbefore)
    wgdnode = insert_node!(Ψ.tree, wgdafter, 0.)
    Ψ.order = postorder(Ψ)
    #@info "Added WGD node $wgdnode and 'after' node $wgdafter"
    Ψ.bindex[wgdnode] = Dict(:q=>i, :θ=>Ψ[n, :θ])
    Ψ.bindex[wgdafter] = Dict(:θ=>Ψ[n, :θ])
    set_parentbranches!(Ψ)
    push!(Ψ.wgds, wgdnode)
    wgdnode, wgdafter
end


# Expanded profile
# ================
function profile(Ψ::SpeciesTree, df::DataFrame)
    M = zeros(Int64, size(df)[1], length(Ψ))
    for n in Ψ.order
        M[:, n] = isleaf(Ψ, n) ? df[:, leafname(Ψ, n)] :
            sum([M[:, c] for c in childnodes(Ψ,n)])
    end
    return M
end


# Example data
# ============
function example_data1()
    s = "(D:18.03,(C:12.06,(B:7.06,A:7.06):4.99):5.97);"
    t = SpeciesTree(read_nw(s)[1:2]...)
    x = DataFrame(:A=>[2],:B=>[2],:C=>[3],:D=>[4])
    M = profile(t, x)
    t, M[1,:]
end

function example_data2()
    s = "(D:18.03,(((C:12.06,(B:7.06,A:7.06):4.99):2.00):0):3.97);"
    t = SpeciesTree(read_nw(s)[1:2]...)
    t.bindex[3, :q] = 1
    x = DataFrame(:A=>[2],:B=>[2],:C=>[3],:D=>[4])
    M = profile(t, x)
    set_wgdrates!(t)
    t, M[1,:]
end


# Useful routines
# ===============
"""
    set_parentbranches!(t::SpeciesTree)

This considers the rate indices!
"""
function set_parentbranches!(s::SpeciesTree)
    for n in s.order
        idx = s[n, :θ]
        nodes = [k for (k,v) in s.bindex if v[:θ] == idx]
        pnodes_ = vcat([get_parentbranches(s, nn) for nn in nodes]...)
        # may repeat nodes, keep always last instance
        pnodes = Int64[]
        for n in reverse(pnodes_)
            !(n in pnodes) ? push!(pnodes, n) : continue
        end
        s.pbranches[n] = reverse(pnodes)
    end
end

function get_parentbranches(s::SpeciesTree, node::Int64)
    node = iswgd(s, node) || iswgdafter(s, node) ?
        non_wgd_child(s, node) : node
    branches = Int64[]
    n = node
    root = findroot(s)
    while n != root
        push!(branches, n)
        n = parentnode(s.tree, n)
    end
    return [branches; [root]]
end

function non_wgd_child(tree::Arboreal, n)
    while outdegree(tree.tree, n) == 1
        n = childnodes(tree.tree, n)[1]
    end
    return n
end

function non_wgd_parent(s::SpeciesTree, n::Int64)
    n == findroot(s) ? (return n) : nothing
    x = parentnode(s, n)
    while iswgd(s, x) || iswgdafter(s, x)
        x = parentnode(s, x)
    end
    return x
end
