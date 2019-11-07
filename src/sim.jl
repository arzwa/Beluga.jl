# simulate data from a DLWGD instance
@with_kw mutable struct GeneTreeNode
    t::Float64 = 0.
    σ::Int64 = 1
    kind::Symbol = :root
end

function simulate!(d::DLWGD)
    root, extant = initialize_sim(d[1])
    simulate_tree!(d[1], extant)
    for (i, n) in enumerate(prewalk(root)); n.i = i; end
    root
end

function profile(n::TreeNode{GeneTreeNode}, d::DLWGD)
    x = Dict(s=>0 for s in values(d.leaves))
    for node in postwalk(n)
        if node.x.kind == :leaf
            x[d.leaves[node.x.σ]] += 1
        end
    end
    x
end

function initialize_sim(root::ModelNode)
    nroot = rand(Geometric(root[:η])) + 1
    r = TreeNode(1, GeneTreeNode())
    extant = [r]
    n = 1         # node counter
    for i in 2:nroot
        source = pop!(extant)
        for j in 1:2
            n += 1
            node = TreeNode(n, GeneTreeNode(kind=:dup), source)
            push!(source, node)
            push!(extant, node)
        end
    end
    return r, extant
end

function simulate_tree!(node::ModelNode, extant)
    if iswgdafter(node)
        return simulate_tree!(first(node), extant)
    end
    for n in extant
        n.x.σ = node.i
        if !isroot(node)
            n.x.kind = iswgd(node) ? :wgd : :sp
        end
        if isleaf(node)
            n.x.kind = :leaf
        end
    end
    if !isleaf(node)
        for c in node.c
            t = c[:t]
            new_extant = TreeNode[]
            for u in extant
                λ, μ = getλμ(nonwgdparent(node), nonwgdchild(c))
                extant_  = dlsim_branch!(u, λ, μ, t, node.i)
                new_extant = [new_extant ; extant_]
                if iswgd(node) && rand() < node[:q]
                    extant_ = dlsim_branch!(u, λ, μ, t, node.i)
                    new_extant = [new_extant; extant_]
                end
            end
            simulate_tree!(c, new_extant)
        end
    end
end

function dlsim_branch!(u::TreeNode, λ, μ, t, j)
    W = Exponential(1. / (λ + μ))
    w = rand(W)  # waiting time
    t -= w
    if t > 0.
        if rand() < λ / (λ + μ)  # birth
            v = TreeNode(0, GeneTreeNode(t=w, kind=:dup, σ=j), u)
            push!(u, v)
            l = dlsim_branch!(v, λ, μ, t, j)
            r = dlsim_branch!(v, λ, μ, t, j)
            return [l ; r]
        else  # death
            v = TreeNode(0, GeneTreeNode(t=w, kind=:loss, σ=j), u)
            push!(u, v)
            return TreeNode[]
        end
    else
        v = TreeNode(0, GeneTreeNode(t=w+t, kind=:sp, σ=j), u)
        push!(u, v)
        return [v]
    end
end
