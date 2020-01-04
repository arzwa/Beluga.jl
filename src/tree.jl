# Minimal tree ipleentation
# NOTE: we used OrderedSet as this allows deterministic iteration over
# childnodes while allowing constant-time insertion and deletion
mutable struct TreeNode{T}
    i::Int64                        # index
    x::T                            # can hold anything, typically a distance
    p::Union{TreeNode{T},Nothing}   # parent node
    c::OrderedSet{TreeNode{T}}      # child nodes
end

const Clade{T} = OrderedSet{TreeNode{T}} where T
newclade(T::Type) = Clade{T}()
TreeNode(T::Type, i::Int64=1) = TreeNode{T}(i, zero(T), nothing, newclade(T))
TreeNode(i::Int64, x::T) where T = TreeNode{T}(i, x, nothing, newclade(T))
TreeNode(i::Int64, x::T, p::TreeNode{T}) where T = TreeNode{T}(i, x, p, newclade(T))
isroot(n::TreeNode) = isnothing(n.p)
isleaf(n::TreeNode) = length(n.c) == 0

# Base extensions
Base.getindex(t::TreeNode, i::Int64) = t.c[i]
Base.length(t::TreeNode) = length(t.c)
Base.show(io::IO, t::TreeNode{T}) where T = write(io, "TreeNode{$T}($(t.i); $(length(t)) children)")
Base.delete!(t::TreeNode, x::TreeNode) = delete!(t.c, x)
Base.push!(t::TreeNode, x::TreeNode) = push!(t.c, x)
Base.first(t::TreeNode) = first(t.c)

# Recursive traversals
function postwalk(t::TreeNode{T}) where T
    ns = TreeNode{T}[]
    function walk!(n)
        if !isleaf(n)
            for c in n.c
                walk!(c)
            end
        end
        push!(ns, n)
    end
    walk!(t)
    ns
end

function prewalk(t::TreeNode{T}) where T
    ns = TreeNode{T}[]
    function walk!(n)
        push!(ns, n)
        if !isleaf(n)
            for c in n.c
                walk!(c)
            end
        end
    end
    walk!(t)
    ns
end

# io
"""
    readnw(nw_str::String)

Read a phylogenetic tree from a newick string.
"""
function readnw(nw_str::String)
    l = Dict(); v = Dict()
    stack = []; i = 1; x = 0.; sv = 0.; j=1
    while i < length(nw_str)
        if nw_str[i] == '('  # add an internal node to stack & tree
            push!(stack, TreeNode(Float64, j)); j += 1
        elseif nw_str[i] == ')'  # add branch + collect data on node
            target = pop!(stack)
            source = stack[end]
            target.p = source
            target.x = x
            push!(source.c, target)
            sv, x, i = get_node_info(nw_str, i+1)
            v[source] = sv
        elseif nw_str[i] == ','  # add branch
            target = pop!(stack)
            source = stack[end]
            target.p = source
            target.x = x
            v[target] = sv
            push!(source.c, target)
        else  # store leaf name and get branch length
            push!(stack, TreeNode(Float64, j)); j+=1
            leaf, i = get_leaf_name(nw_str, i)
            sv, x, i = get_node_info(nw_str, i)
            l[stack[end].i] = leaf
        end
        i += 1
    end
    (t=stack[end], l=l, v=v)
end

function get_leaf_name(nw_str::String, i::Int64)
    j = i
    while (nw_str[j] != ':') & (nw_str[j] != ',') & (nw_str[j] != ')')
        j += 1
    end
    leaf = nw_str[i:j-1]
    return leaf, j
end

function get_node_info(nw_str, i)
    # get everything up to the next comma or semicolon
    substr = nw_str[i:end]
    substr = split(substr, ',')[1]
    substr = split(substr, ')')[1]
    if substr == ";"
        return -1.0, -1.0, i
    end
    substr = split(substr, ';')[1]
    if occursin(":", substr)
        sv, bl = split(substr, ':')
        if sv == ""  # only branch length
            sv = 0.0
            bl = parse(Float64, bl)
        else  # both (B)SV and branch length are there
            sv = parse(Float64, sv)
            bl = parse(Float64, bl)
        end
    else
        # nothing there
        bl = sv = -1.
    end
    return sv, bl, i + length(substr) -1
end
