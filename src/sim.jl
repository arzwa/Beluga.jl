
# Simulation of profiles and trees from DL model
# ==============================================
Base.rand(m::DuplicationLossWGD) = profile_fromtree(randtree(m), m.tree)

function profile_fromtree(t::RecTree, s::SpeciesTree)
    d = Dict{Symbol,Int64}(x=>0 for x in values(s.leaves))
    merge(d, countmap([s.leaves[t.σ[n]] for n in keys(t.leaves)]))
end

profile_fromtree(t::RecTree) = countmap([t.σ[n] for n in keys(t.leaves)])

function randtree(m::DuplicationLossWGD)
    Ψ, extant = initialize_sim(m)
    i = 1
    function simulate_tree!(e, extant)
        for n in extant
            Ψ.σ[n] = e
            Ψ.labels[n] = Beluga.iswgd(m.tree, e) ? "wgd" : "speciation"
        end
        if isleaf(m.tree, e)
            for n in extant
                Ψ.leaves[n] = "$(m.tree.leaves[e])_$i"
                i += 1
            end
            return
        else
            for f in childnodes(m.tree, e)
                t = distance(m.tree, e, f)
                new_extant = Int64[]
                for u in extant
                    λ = m[:λ, f]
                    μ = m[:μ, f]
                    Ψ, extant_ = dlsim_branch!(Ψ, u, λ, μ, t, e)
                    new_extant = [new_extant ; extant_]
                    if Beluga.iswgd(m.tree, e) && rand() < m[:q, e]
                        T, extant_ = dlsim_branch!(Ψ, u, λ, μ, t, e)
                        new_extant = [new_extant ; extant_]
                    end

                end
                simulate_tree!(f, new_extant)
            end
        end
    end
    simulate_tree!(findroot(m.tree), extant)
    return Ψ
end

function initialize_sim(m::PhyloBDP)
    Ψ = RecTree()
    addnode!(Ψ.tree)             # initialize root
    extant = [1]                 # records all lineges that are currently extant
    nroot = rand(Geometric(m.η)) + 1        # lineages at the root
    n = 1                        # node counter
    for i in 2:nroot
        source = pop!(extant)
        for j in 1:2
            n += 1
            addnode!(Ψ.tree)
            addbranch!(Ψ.tree, source, n, 1.)  # branch lengths are meaningless
            push!(extant, n)
            Ψ.σ[n] = 1
            Ψ.labels[n] = "duplication"
        end
    end
    return Ψ, extant
end


function dlsim_branch!(T::RecTree, u::Int64, λ::Float64, μ::Float64,
        t::Float64, label::Int64)
    W = Exponential(1 / (λ + μ))
    waiting_time = rand(W)
    t -= waiting_time
    if t > 0
        # birth
        if rand() < λ / (λ + μ)
            addnode!(T.tree)
            v = maximum(keys(T.tree.nodes))
            addbranch!(T.tree, u, v, waiting_time)
            T.σ[v] = label
            T.labels[v] = "duplication"
            T, l = dlsim_branch!(T, v, λ, μ, t, label)
            T, r = dlsim_branch!(T, v, λ, μ, t, label)
            return T, [l ; r]
        # death
        else
            addnode!(T.tree)
            v = maximum(keys(T.tree.nodes))
            addbranch!(T.tree, u, v, waiting_time)
            T.σ[v] = label
            T.labels[v] = "loss"
            return T, []
        end
    else
        addnode!(T.tree)
        v = maximum(keys(T.tree.nodes))
        addbranch!(T.tree, u, v, t + waiting_time)
        return T, [v]
    end
end
