# simulation of profiles and trees under (general) BDPs

function Base.rand(m::DLModel)
end

function Base.randtree(m::DLModel)
    Ψ, extant = initialize_sim(m)

    function simulate_tree(e, extant)
        if isleaf(m.tree, e)
            for n in extant
                labels[n] = e
            end
        else
            for f in childnodes(m.tree, e)
                t = distance(m.tree, e, f)
                extant_ = Int64[]
                

        end
    end
end

function initialize_sim(m::PhyloLinearBDP)
    Ψ = RecTree()
    addnode!(Ψ.tree)             # initialize root
    extant = [1]                 # records all lineges that are currently extant
    nroot = rand(m.ρ) + 1        # lineages at the root
    n = 1                        # node counter
    for i in 2:nroot
        source = pop!(extant)
        for j in 1:2
            n += 1
            addnode!(T)
            addbranch!(T, source, n, 1.)  # branch lengths are meaningless
            push!(extant, n)
            Ψ.σ[n] = 1
            Ψ.labels[n] = "duplication"
        end
    end
    return Ψ, extant
end
