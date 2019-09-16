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

Base.length(Ψ::SpeciesTree) = length(Ψ.tree.nodes)
Base.display(Ψ::SpeciesTree) = println("$(typeof(Ψ))($(length(Ψ)))")

leafname(Ψ::SpeciesTree, i::Int64) = Ψ.leaves[i]
leafname(d::PhyloLinearBDP, i::Int64) = leafname(d.tree, i)

iswgd(Ψ::SpeciesTree, i::Int64) = haskey(Ψ.bindex[i], :q)
iswgdafter(Ψ::SpeciesTree, i::Int64) = haskey(Ψ.bindex[parentnode(Ψ, i)], :q)
qparent(Ψ::SpeciesTree, i::Int64) = Ψ.bindex[parentnode(Ψ, i), :q]
nwgd(Ψ::SpeciesTree) = length([k for (k,v) in Ψ.bindex if haskey(v, :q)])

function set_constantrates!(Ψ::SpeciesTree, s=:θ)
    for k in keys(Ψ.bindex)
        Ψ.bindex[k, s] = 1
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
    t, M[1,:]
end

function get_parentbranches(s::Arboreal, node::Int64)
    branches = Int64[]
    n = node
    root = findroot(s)
    while n != root
        push!(branches, n)
        n = parentnode(s.tree, n)
    end
    return [branches; [root]]
end
