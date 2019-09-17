# Species tree for phylogenetic birth-death models
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

"""
    SpeciesTree(tree, leaves, bindex)
"""
struct SpeciesTree <: Arboreal
    tree::Tree
    leaves::Dict{Int64,Symbol}
    bindex::BranchIndex
    order::Array{Int64,1}
end

SpeciesTree(tree::Tree, leaves::Dict{Int64,T}) where T<:AbstractString =
    SpeciesTree(tree, Dict(k=>Symbol(v) for (k,v) in leaves), defaultidx(tree),
        postorder(tree))
SpeciesTree(tree::LabeledTree) = SpeciesTree(tree.tree, tree.leaves)
SpeciesTree(treefile::String) = SpeciesTree(readtree(treefile))

defaultidx(tree) = BranchIndex(i=>Dict(:θ=>i) for (i,n) in tree.nodes)

Base.getindex(Ψ::SpeciesTree, i::Int64, s::Symbol) = Ψ.bindex[i, s]
Base.setindex!(Ψ::SpeciesTree, x::Int64, i::Int64,s::Symbol) = Ψ.bindex[i,s] = x
Base.length(Ψ::SpeciesTree) = length(Ψ.tree.nodes)
Base.display(Ψ::SpeciesTree) = println("$(typeof(Ψ))($(length(Ψ)))")

leafname(Ψ::SpeciesTree, i::Int64) = Ψ.leaves[i]
leafname(d::PhyloLinearBDP, i::Int64) = leafname(d.tree, i)

# WGD stuff
# NB: Wgd is assumed to be defined by having a q entry in the branch index!
iswgd(Ψ::SpeciesTree, i::Int64) = haskey(Ψ.bindex[i], :q)
iswgdafter(Ψ::SpeciesTree, i::Int64) = isroot(Ψ, i) ?
    false : haskey(Ψ.bindex[parentnode(Ψ, i)], :q)
qparent(Ψ::SpeciesTree, i::Int64) = Ψ.bindex[parentnode(Ψ, i), :q]
nwgd(Ψ::SpeciesTree) = length(wgdnodes(Ψ))
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
            Ψ[n, s]  = d[Ψ[n, s]]
        else
            d[Ψ[n, s]] = i
            Ψ[n, s] = i
            i += 1
        end
    end
end

hasconstantrates(Ψ::SpeciesTree, s=:θ) =
    all([Ψ.bindex[s, i]==1 for i=1:Ψ.order])

# expanded profile
function profile(Ψ::SpeciesTree, df::DataFrame)
    M = zeros(Int64, size(df)[1], length(Ψ))
    for n in Ψ.order
        M[:, n] = isleaf(Ψ, n) ? df[:, leafname(Ψ, n)] :
            sum([M[:, c] for c in childnodes(Ψ,n)])
    end
    return M
end

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

"""
    get_parentbranches(t::SpeciesTree, node)

Get the branches that should be recomputed in the dynamic programming matrix,
given that the parameters associated with the branch leading to `node` have
changed. Note: this is not always parent branches alone, but in case of WGD
nodes, also the branch below the WGD node.
"""
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
