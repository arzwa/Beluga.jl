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
end

SpeciesTree(tree::Tree, leaves::Dict{Int64,String}) =
    SpeciesTree(tree, Dict(k=>Symbol(v) for (k,v) in leaves), defaultidx(tree))

defaultidx(tree) = BranchIndex(i=>Dict(:θ=>i) for (i,n) in tree.nodes)

Base.length(Ψ::SpeciesTree) = length(Ψ.tree.nodes)
Base.display(Ψ::SpeciesTree) = println("$(typeof(Ψ))($(length(Ψ)))")

leafname(Ψ::SpeciesTree, i::Int64) = Ψ.leaves[i]
leafname(d::PhyloLinearBDP, i::Int64) = leafname(d.tree, i)

function set_constantrates!(Ψ::SpeciesTree, s=:θ)
    for k in keys(Ψ.bindex)
        Ψ.bindex[k, s] = 1
    end
end 
